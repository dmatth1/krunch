"""
GPU adaptive head MUST byte-equal the CPU reference at every step.

This is the foundation invariant for AC roundtrip: encoder runs on
GPU, decoder runs on GPU, both must produce identical bias trajectories
to the CPU reference Nacrith uses, otherwise the AC bytes diverge.

Per docs/TIER_3_OPTIMIZATION.md NEXT-3.
"""

import numpy as np
import pytest
import torch

from krunch.codec.adaptive_head import (
    AdaptiveHead, encode_with_adaptive_head, DEFAULT_LR,
)
from krunch.codec.adaptive_head_gpu import (
    AdaptiveHeadGPU, encode_with_adaptive_head_gpu,
)


def _rand_probs(T, V, seed):
    rng = np.random.default_rng(seed)
    p = rng.random((T, V)).astype(np.float64)
    return p / p.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Single-step parity
# ---------------------------------------------------------------------------

def test_adjust_single_step_matches_cpu():
    """One adjust() call on GPU must byte-equal numpy at fp64."""
    V = 1024
    p = _rand_probs(1, V, seed=1)[0]

    cpu_h = AdaptiveHead(V)
    cpu_out = cpu_h.adjust(p)

    gpu_h = AdaptiveHeadGPU(V, batch_size=1, device="cuda")
    p_gpu = torch.from_numpy(p).cuda()
    gpu_out = gpu_h.adjust(p_gpu).cpu().numpy()

    assert np.array_equal(cpu_out, gpu_out), \
        f"max diff: {np.abs(cpu_out - gpu_out).max():.3e}"


def test_update_single_step_matches_cpu():
    V = 256
    p = _rand_probs(1, V, seed=2)[0]
    sym = 17

    cpu_h = AdaptiveHead(V)
    cpu_adj = cpu_h.adjust(p)
    cpu_h.update(sym, cpu_adj)

    gpu_h = AdaptiveHeadGPU(V, batch_size=1, device="cuda")
    gpu_adj = gpu_h.adjust(torch.from_numpy(p).cuda())
    gpu_h.update(sym, gpu_adj)

    assert np.array_equal(cpu_h.bias, gpu_h.bias[0].cpu().numpy())


# ---------------------------------------------------------------------------
# Trajectory parity (the load-bearing test)
# ---------------------------------------------------------------------------

def test_full_trajectory_matches_cpu_at_realistic_vocab():
    """Run T=128 tokens at V=50277 (production vocab). At every step
    the adjusted-probs vector and the bias must byte-equal."""
    T, V = 128, 50277
    probs = _rand_probs(T, V, seed=42)
    rng = np.random.default_rng(42)
    tokens = rng.integers(0, V, size=T)

    cpu_out, cpu_bias = encode_with_adaptive_head(probs, tokens)

    probs_gpu = torch.from_numpy(probs).cuda()
    tokens_gpu = torch.from_numpy(tokens).cuda()
    gpu_out_t, gpu_bias_t = encode_with_adaptive_head_gpu(
        probs_gpu, tokens_gpu, device="cuda"
    )
    gpu_out = gpu_out_t.cpu().numpy()
    gpu_bias = gpu_bias_t.cpu().numpy()

    # Byte-exact at fp64 — every step.
    assert cpu_out.shape == gpu_out.shape == (T, V)
    diff_mask = (cpu_out != gpu_out).any(axis=1)
    assert not diff_mask.any(), (
        f"GPU/CPU diverge starting at step {int(diff_mask.argmax())}; "
        f"max abs diff there: "
        f"{np.abs(cpu_out[diff_mask.argmax()] - gpu_out[diff_mask.argmax()]).max():.3e}"
    )
    assert np.array_equal(cpu_bias, gpu_bias)


# ---------------------------------------------------------------------------
# Batched B>1 sanity (decoder shape)
# ---------------------------------------------------------------------------

def test_batched_decoder_each_row_matches_independent_cpu_run():
    """Decoder runs B=n_chunks bias rows in lockstep. Each row's
    trajectory must equal a standalone CPU run with that row's tokens."""
    B, T, V = 4, 64, 1024
    rng = np.random.default_rng(123)
    probs = _rand_probs(T * B, V, seed=123).reshape(T, B, V)
    tokens = rng.integers(0, V, size=(T, B))

    # GPU batched.
    h_gpu = AdaptiveHeadGPU(V, batch_size=B, device="cuda")
    probs_gpu = torch.from_numpy(probs).cuda()
    tokens_gpu = torch.from_numpy(tokens).cuda()
    gpu_adj = torch.empty((T, B, V), dtype=torch.float64, device="cuda")
    for t in range(T):
        gpu_adj[t] = h_gpu.adjust(probs_gpu[t])
        h_gpu.update(tokens_gpu[t], gpu_adj[t])

    # CPU per-row reference.
    for b in range(B):
        cpu_out, cpu_bias = encode_with_adaptive_head(probs[:, b, :], tokens[:, b])
        gpu_out_b = gpu_adj[:, b, :].cpu().numpy()
        gpu_bias_b = h_gpu.bias[b].cpu().numpy()
        assert np.array_equal(cpu_out, gpu_out_b), (
            f"row {b}: GPU diverges from CPU"
        )
        assert np.array_equal(cpu_bias, gpu_bias_b)


# ---------------------------------------------------------------------------
# Reset + cross-call independence
# ---------------------------------------------------------------------------

def test_reset_brings_gpu_back_to_zero():
    h = AdaptiveHeadGPU(100, batch_size=1, device="cuda")
    p = torch.full((100,), 0.01, dtype=torch.float64, device="cuda")
    h.update(0, h.adjust(p))
    assert (h.bias != 0).any()
    h.reset()
    assert (h.bias == 0).all()


# ---------------------------------------------------------------------------
# Performance smoke
# ---------------------------------------------------------------------------

def test_realistic_vocab_runs_in_reasonable_time():
    """T=256 tokens, V=50277, B=1: should be < 250 ms on A10G class.
    (Bound is generous — we mostly want to catch a 10× regression.)"""
    import time
    T, V = 256, 50277
    rng = np.random.default_rng(7)
    probs = _rand_probs(T, V, seed=7)
    tokens = rng.integers(0, V, size=T)

    h = AdaptiveHeadGPU(V, batch_size=1, device="cuda")
    probs_gpu = torch.from_numpy(probs).cuda()
    tokens_gpu = torch.from_numpy(tokens).cuda()

    torch.cuda.synchronize()
    t0 = time.time()
    for t in range(T):
        adj = h.adjust(probs_gpu[t])
        h.update(int(tokens_gpu[t].item()), adj)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    assert elapsed < 0.25, f"adaptive head took {elapsed*1000:.1f} ms"

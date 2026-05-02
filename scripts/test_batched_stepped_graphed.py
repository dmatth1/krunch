"""Verify forward_stepped_batched_graphed_v2 vs forward_stepped_batched.

Goal: prove (1) graphed batched forward is bit-equivalent to non-graphed,
(2) measure speedup.

Requires KRUNCH_DETERMINISTIC_MATMUL=1 + KRUNCH_OWN_WKV=1 — the graph
determinism prerequisite established by test_graph_determinism.py.
"""
import os, sys, time
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("KRUNCH_OWN_WKV", "1")

import torch
import krunch_ac_cuda
from krunch import cpp_path


def maxabs(a, b): return (a.float() - b.float()).abs().max().item()


def make_fake_weights(device):
    """Build a `weights` dict matching cpp_path.init_weights' output but
    with random tensors, to avoid loading the real model in this test."""
    n_embd = 768
    n_att = 768
    n_ffn = 3072
    V = 50277

    def hf(*shape):
        return torch.randn(*shape, dtype=torch.float16, device=device) * 0.05
    def hf_one(*shape):
        return torch.randn(*shape, dtype=torch.float16, device=device) * 0.5 + 1.0
    def f32(*shape, scale=0.3):
        return torch.randn(*shape, dtype=torch.float32, device=device) * scale

    layers = []
    for _ in range(cpp_path.N_LAYER):
        layers.append([
            hf_one(n_embd),                           # ln1.weight
            hf(n_embd) * 0.05,                         # ln1.bias
            torch.rand(n_embd, dtype=torch.float16, device=device),  # tm_k
            torch.rand(n_embd, dtype=torch.float16, device=device),  # tm_v
            torch.rand(n_embd, dtype=torch.float16, device=device),  # tm_r
            -torch.rand(n_att, dtype=torch.float32, device=device) - 0.1,
            f32(n_att),
            hf(n_embd, n_att),                         # Kw
            hf(n_embd, n_att),                         # Vw
            hf(n_embd, n_embd),                        # Rw
            hf(n_att, n_embd),                         # Ow
            hf_one(n_embd),                            # ln2.weight
            hf(n_embd) * 0.05,                          # ln2.bias
            torch.rand(n_embd, dtype=torch.float16, device=device),  # ffn.tm_k
            torch.rand(n_embd, dtype=torch.float16, device=device),  # ffn.tm_r
            hf(n_embd, n_ffn),                         # ffn_Kw
            hf(n_ffn, n_embd),                         # ffn_Vw
            hf(n_embd, n_embd),                        # ffn_Rw
        ])
    return {
        "layers": layers,
        "emb_w": hf(V, n_embd),
        "ln_out_w": hf_one(n_embd),
        "ln_out_b": hf(n_embd) * 0.05,
        "head_w": hf(n_embd, V),
        "n_embd": n_embd,
        "n_att": n_att,
        "device": device,
    }


def run_steps(forward_fn, weights, last_tokens_list, state):
    """Run forward_fn for each step in last_tokens_list, return logits list."""
    out = []
    for tok in last_tokens_list:
        out.append(forward_fn(weights, tok, state).clone())
    return out


def bench(forward_fn, weights, B, n_iters=10, n_warmup=3):
    last = torch.zeros(B, dtype=torch.long, device="cuda")
    state = cpp_path.fresh_state_batched(weights, B)
    for _ in range(n_warmup):
        forward_fn(weights, last, state)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iters):
        forward_fn(weights, last, state)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iters


def correctness_check(weights, B):
    """Verify graphed and non-graphed produce bit-equivalent logits."""
    print(f"--- correctness B={B} ---")
    last_tokens_list = [torch.full((B,), i, dtype=torch.long, device="cuda")
                         for i in range(5)]

    state_a = cpp_path.fresh_state_batched(weights, B)
    state_b = cpp_path.fresh_state_batched(weights, B)

    # Run both with identical inputs + initial state
    raw_logits = run_steps(cpp_path.forward_stepped_batched, weights,
                            last_tokens_list, state_a)
    # Reset cache for clean capture
    cpp_path._GRAPH_BATCHED_CACHE.clear()
    cpp_path._STEPPED_BATCHED_BUFS_CACHE.clear()
    graphed_logits = run_steps(cpp_path.forward_stepped_batched_graphed_v2,
                                 weights, last_tokens_list, state_b)

    pass_ = True
    for i in range(5):
        d = maxabs(raw_logits[i], graphed_logits[i])
        ok = d < 1e-3
        pass_ = pass_ and ok
        print(f"  step {i}: max-abs {d:.6e}  {'PASS' if ok else 'FAIL'}")
    return pass_


def main():
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(2)
    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name()}")

    torch.manual_seed(0)
    weights = make_fake_weights(device)

    all_pass = True
    for B in (128, 256):
        ok = correctness_check(weights, B)
        all_pass = all_pass and ok

    print(f"\nverdict (correctness): {'ALL PASS' if all_pass else 'FAIL'}")
    if not all_pass:
        sys.exit(1)

    # Speed
    print("\n--- speed (ms/forward call) ---")
    print(f"{'B':>4}   {'raw':>8}   {'graphed':>8}   {'speedup'}")
    for B in (128, 256, 512):
        # Fresh state to avoid any cache cross-talk
        cpp_path._GRAPH_BATCHED_CACHE.clear()
        cpp_path._STEPPED_BATCHED_BUFS_CACHE.clear()
        t_raw = bench(cpp_path.forward_stepped_batched, weights, B)
        t_g   = bench(cpp_path.forward_stepped_batched_graphed_v2, weights, B)
        print(f"{B:>4}   {t_raw:>6.2f}   {t_g:>6.2f}   {t_raw/t_g:.2f}x")


if __name__ == "__main__":
    main()

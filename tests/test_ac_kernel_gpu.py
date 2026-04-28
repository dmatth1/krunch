"""
GPU kernel correctness: byte-identical to CPU reference.

Only runs on a CUDA host. Requires krunch_ac.cuda extension built
(cd krunch_ac/cuda && python setup.py build_ext --inplace).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

try:
    import krunch_ac_cuda  # noqa: F401
except ImportError:
    pytest.skip("krunch_ac_cuda extension not built", allow_module_level=True)

from krunch_ac.cpu_reference import encode as cpu_encode
from krunch_ac.gpu_encode import probs_to_cdf_gpu


def _rand_probs(N, V, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.random((N, V)).astype(np.float32)
    p /= p.sum(axis=1, keepdims=True)
    return p


def _gpu_encode_single_batch(probs_np, symbols_np):
    """Encode one batch on GPU. Returns (gpu_bytes, cdf_np_uint32)
    so the CPU reference can encode the SAME CDF for byte-equality
    comparison (numpy fp32 and torch fp32 can disagree in the last
    bit on reductions, so we must compare encoders on identical
    CDFs, not on independent CDF constructions)."""
    device = "cuda"
    probs = torch.from_numpy(probs_np).to(device)
    symbols = torch.from_numpy(symbols_np.astype(np.int32)).to(device)

    cdf = probs_to_cdf_gpu(probs).contiguous()
    output_buf = torch.zeros(len(symbols_np) * 8 + 64, dtype=torch.uint8, device=device)
    state = torch.zeros(4, dtype=torch.uint32, device=device)
    state[1] = 0xFFFFFFFF

    krunch_ac_cuda.encode_step(cdf, symbols, output_buf, state)
    krunch_ac_cuda.encode_finalize(output_buf, state)
    torch.cuda.synchronize()

    bit_offset = int(state[3].item())
    n_bytes = (bit_offset + 7) // 8
    gpu_bytes = bytes(output_buf[:n_bytes].cpu().numpy())
    cdf_np = cdf.cpu().numpy().astype(np.uint32)
    return gpu_bytes, cdf_np


@pytest.mark.parametrize("N,V,seed", [
    (1, 4, 0),
    (8, 32, 1),
    (256, 256, 2),
    (1024, 1024, 3),
    (256, 50277, 4),
])
def test_kernel_byte_identical_to_reference(N, V, seed):
    p = _rand_probs(N, V, seed=seed)
    rng = np.random.default_rng(seed + 100)
    cum = np.cumsum(p, axis=1)
    u = rng.random((N, 1)).astype(np.float32)
    sym = (u < cum).argmax(axis=1).astype(np.int32)

    bs_gpu, cdf_np = _gpu_encode_single_batch(p, sym)
    bs_cpu = cpu_encode(cdf_np, sym)  # CPU encode using the SAME CDF

    assert bs_cpu == bs_gpu, (
        f"GPU kernel diverges from CPU reference at N={N},V={V},seed={seed}\n"
        f"  cpu: {bs_cpu[:32].hex()}... ({len(bs_cpu)} bytes)\n"
        f"  gpu: {bs_gpu[:32].hex()}... ({len(bs_gpu)} bytes)")


def test_multi_step_state_persists():
    """Encode 3 batches sequentially with persistent state == one big
    batch encoded by the CPU reference using the GPU-computed CDFs."""
    N_per, V = 100, 256
    seqs = [_rand_probs(N_per, V, seed=i) for i in range(3)]
    rng = np.random.default_rng(42)
    syms = [rng.integers(0, V, size=N_per).astype(np.int32) for _ in range(3)]

    device = "cuda"
    output_buf = torch.zeros(N_per * 3 * 8 + 64, dtype=torch.uint8, device=device)
    state = torch.zeros(4, dtype=torch.uint32, device=device)
    state[1] = 0xFFFFFFFF
    cdf_chunks = []
    for p, s in zip(seqs, syms):
        probs = torch.from_numpy(p).to(device)
        symbols = torch.from_numpy(s).to(device)
        cdf = probs_to_cdf_gpu(probs).contiguous()
        cdf_chunks.append(cdf.cpu().numpy().astype(np.uint32))
        krunch_ac_cuda.encode_step(cdf, symbols, output_buf, state)
    krunch_ac_cuda.encode_finalize(output_buf, state)
    torch.cuda.synchronize()
    n_bytes = (int(state[3].item()) + 7) // 8
    bs_gpu = bytes(output_buf[:n_bytes].cpu().numpy())

    # CPU reference: encode all three using the same GPU-computed CDFs.
    cdf_all = np.concatenate(cdf_chunks, axis=0)
    s_all = np.concatenate(syms, axis=0).astype(np.int32)
    bs_ref = cpu_encode(cdf_all, s_all)

    assert bs_ref == bs_gpu, "multi-step GPU encode diverges from one-shot CPU on same CDFs"


# ---------------------------------------------------------------------------
# Decode kernel: round-trip via GPU encode -> GPU decode
# ---------------------------------------------------------------------------

def _gpu_decode_one_chunk(bs, cdfs_np):
    """Decode N symbols on GPU using the same CDFs as the encoder.
    cdfs_np: (N, V+1) uint32 numpy. Returns (N,) int32 numpy of decoded
    symbols."""
    device = "cuda"
    N, Vp1 = cdfs_np.shape
    # Pad bitstream to allow over-read on the last token's renorm.
    bs_padded = bs + b"\x00" * 64
    input_buf = torch.from_numpy(np.frombuffer(bs_padded, dtype=np.uint8).copy()).to(device)
    state = torch.zeros(4, dtype=torch.uint32, device=device)
    out_sym = torch.empty(1, dtype=torch.int32, device=device)

    krunch_ac_cuda.decode_init(input_buf, state)
    cdfs_gpu = torch.from_numpy(cdfs_np.astype(np.int32)).to(device)

    decoded = np.empty(N, dtype=np.int32)
    for i in range(N):
        cdf_row = cdfs_gpu[i].contiguous()
        krunch_ac_cuda.decode_step(cdf_row, input_buf, state, out_sym)
        decoded[i] = int(out_sym.item())
    return decoded


@pytest.mark.parametrize("N,V,seed", [
    (1, 4, 0), (8, 32, 1), (256, 256, 2), (1024, 1024, 3), (256, 50277, 4),
])
def test_decode_kernel_roundtrip(N, V, seed):
    p = _rand_probs(N, V, seed=seed)
    rng = np.random.default_rng(seed + 100)
    cum = np.cumsum(p, axis=1)
    u = rng.random((N, 1)).astype(np.float32)
    sym = (u < cum).argmax(axis=1).astype(np.int32)

    bs_gpu, cdf_np = _gpu_encode_single_batch(p, sym)
    decoded = _gpu_decode_one_chunk(bs_gpu, cdf_np)
    assert (decoded == sym).all(), (
        f"GPU decode diverges from input symbols at N={N},V={V},seed={seed}\n"
        f"first mismatch: idx={(decoded != sym).argmax()} "
        f"expected={sym[(decoded != sym).argmax()]} got={decoded[(decoded != sym).argmax()]}")

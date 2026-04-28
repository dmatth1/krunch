"""
GPU encode path: probs (on GPU) → integer CDF (on GPU) → CUDA range
coder kernel → bitstream bytes.

The probs→CDF step uses pure torch ops so no second custom kernel
is needed. Only the serial range encode runs in our CUDA code.

Importable on Mac (without CUDA) — the kernel import is deferred
until encode() is actually called.
"""

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from krunch_ac.cdf import T as CDF_T


def probs_to_cdf_gpu(probs):
    """
    GPU probs → int32 CDF. Mirrors krunch_ac.cdf.probs_to_cdf bit-for-bit
    (same MIN_PROB=1, same deficit-distribution rule).

    Returns int32 (not uint16) because cdf[:, V] == T == 65536, which
    doesn't fit uint16. 200 KB extra per row × 1024 rows = 200 MB GPU
    memory — affordable, and avoids a needless dtype dance.
    """
    assert torch is not None, "torch required"
    assert probs.is_cuda and probs.dim() == 2

    N, V = probs.shape
    assert V < CDF_T, f"vocab {V} must be < T={CDF_T}"

    p = probs.to(torch.float32)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-30)

    counts = (p * float(CDF_T - V)).floor().to(torch.int32) + 1  # MIN_PROB
    deficit = (CDF_T - counts.sum(dim=1)).to(torch.int32)         # (N,)

    argmax = p.argmax(dim=1)
    counts.scatter_add_(1, argmax.unsqueeze(1), deficit.unsqueeze(1))

    cdf = torch.zeros((N, V + 1), dtype=torch.int32, device=probs.device)
    cdf[:, 1:] = torch.cumsum(counts, dim=1).to(torch.int32)
    return cdf


_kernel = None


def _load_kernel():
    global _kernel
    if _kernel is None:
        import krunch_ac_cuda  # built via cuda/setup.py on the GPU host
        _kernel = krunch_ac_cuda
    return _kernel


def encode_chunk_gpu(probs_iter, symbols_iter, max_output_bytes):
    """
    Stream-encode a chunk's batches on GPU.

    Args:
        probs_iter: iterator yielding (B, V) float CUDA tensors per batch.
        symbols_iter: iterator yielding (B,) int32 CUDA tensors per batch.
        max_output_bytes: upper bound for the encoded bitstream length.
            Caller should over-provision (e.g., chunk_size_bytes for the
            worst case where the LM compresses nothing).

    Returns:
        bytes — the encoded bitstream, length = ceil(state.bit_offset / 8).
    """
    assert torch is not None
    krunch_ac_cuda = _load_kernel()

    device = None
    output_buf = None
    state = None

    for probs, symbols in zip(probs_iter, symbols_iter):
        if device is None:
            device = probs.device
            output_buf = torch.zeros(max_output_bytes, dtype=torch.uint8, device=device)
            state = torch.zeros(4, dtype=torch.uint32, device=device)
            # Initial range coder state: low=0, high=0xFFFFFFFF.
            state[0] = 0
            state[1] = 0xFFFFFFFF
            state[2] = 0
            state[3] = 0
        cdf = probs_to_cdf_gpu(probs).contiguous()
        krunch_ac_cuda.encode_step(cdf, symbols.contiguous(), output_buf, state)

    if state is None:
        return b""
    krunch_ac_cuda.encode_finalize(output_buf, state)

    bit_offset = int(state[3].item())
    n_bytes = (bit_offset + 7) // 8
    return output_buf[:n_bytes].cpu().numpy().tobytes()

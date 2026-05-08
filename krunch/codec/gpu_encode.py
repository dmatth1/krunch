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

from krunch.codec.cdf import T as CDF_T


def probs_to_cdf_gpu(probs):
    """
    GPU probs → int32 CDF. Mirrors krunch.codec.cdf.probs_to_cdf bit-for-bit
    (same MIN_PROB=1, same deficit-distribution rule).

    Returns int32 because cdf[:, V] == T == 2^24, which doesn't fit
    uint16 (and barely fits int24, so int32 is the natural choice).
    int32 CDFs at vocab=50K use ~200 KB/row.
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



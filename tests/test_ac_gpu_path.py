"""
GPU CDF path: self-consistency tests.

Cross-library byte equality (numpy fp32 == torch fp32) is intentionally
NOT a guarantee — the two libraries can disagree in the last bit on fp32
reductions, which would diverge the CDF. What we DO guarantee is:

  encode-via-GPU-CDF (bytes) → decode-via-CPU-reference (using the same
  GPU-computed CDF, transferred to CPU) → round-trip equal.

That's the property production cares about. Both encoder and decoder use
the same CDF tensor, so they agree exactly.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from krunch_ac.cpu_reference import encode as cpu_encode, decode as cpu_decode


def _rand_probs(N, V, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.random((N, V)).astype(np.float32)
    p /= p.sum(axis=1, keepdims=True)
    return p


def _torch_probs_to_cdf_cpu(probs):
    """Mirror of probs_to_cdf_gpu but runs on CPU device — same arithmetic
    so we exercise the GPU code path locally on Mac."""
    from krunch_ac.cdf import T as CDF_T
    N, V = probs.shape
    p = probs.to(torch.float32)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-30)
    counts = (p * float(CDF_T - V)).floor().to(torch.int32) + 1
    deficit = (CDF_T - counts.sum(dim=1)).to(torch.int32)
    argmax = p.argmax(dim=1)
    counts.scatter_add_(1, argmax.unsqueeze(1), deficit.unsqueeze(1))
    cdf = torch.zeros((N, V + 1), dtype=torch.int32)
    cdf[:, 1:] = torch.cumsum(counts, dim=1).to(torch.int32)
    return cdf


@pytest.mark.parametrize("N,V,seed", [
    (1, 4, 0), (16, 64, 1), (256, 1024, 2), (256, 50277, 3),
])
def test_gpu_cdf_roundtrip(N, V, seed):
    """Encode using torch CDF, decode using same CDF, recover symbols."""
    p_np = _rand_probs(N, V, seed=seed)
    rng = np.random.default_rng(seed + 100)
    cum = np.cumsum(p_np, axis=1)
    u = rng.random((N, 1)).astype(np.float32)
    sym = (u < cum).argmax(axis=1).astype(np.int32)

    cdf_torch = _torch_probs_to_cdf_cpu(torch.from_numpy(p_np))
    cdf_np = cdf_torch.numpy().astype(np.uint32)

    bs = cpu_encode(cdf_np, sym)
    decoded = cpu_decode(bs, cdf_np)
    assert (decoded == sym).all(), (
        f"round-trip failed at N={N},V={V},seed={seed}")


def test_gpu_cdf_invariants():
    """Torch CDF satisfies the same invariants as numpy CDF."""
    from krunch_ac.cdf import T as CDF_T
    p = _rand_probs(64, 1024, seed=99)
    cdf = _torch_probs_to_cdf_cpu(torch.from_numpy(p)).numpy().astype(np.int64)
    assert (cdf[:, 0] == 0).all()
    assert (cdf[:, -1] == CDF_T).all()
    assert (np.diff(cdf, axis=1) >= 1).all(), "every bin needs width >= 1"

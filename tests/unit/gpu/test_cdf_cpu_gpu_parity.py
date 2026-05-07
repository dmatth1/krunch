"""probs_to_cdf (CPU reference) vs probs_to_cdf_gpu — bit-exact match.

The CPU reference (`krunch.codec.cdf.probs_to_cdf`) is the spec; the
GPU kernel must produce byte-identical output for AC roundtrip to hold
when the encoder is on GPU and the decoder reads the same CDF on CPU
(or vice-versa).

Both implementations are deterministic (fixed-point, no floating-point
reduction order issue) so byte-equality is the right contract.
"""
import pytest
import torch

from krunch.codec.cdf import probs_to_cdf, T as CDF_T


@pytest.fixture
def probs():
    """Realistic-shape probs over the RWKV vocab."""
    torch.manual_seed(0)
    raw = torch.randn(4, 50277, dtype=torch.float16, device="cuda")
    p = torch.softmax(raw.float(), dim=-1)
    p = p.clamp(min=1e-9)
    return (p / p.sum(dim=-1, keepdim=True)).to(torch.float32)


def test_gpu_cdf_matches_cpu_reference_per_row(probs):
    """For each row, GPU CDF must equal CPU CDF byte-for-byte."""
    from krunch.codec.gpu_encode import probs_to_cdf_gpu

    gpu_cdfs = probs_to_cdf_gpu(probs).cpu()
    cpu_cdfs = []
    for row in probs.cpu().numpy():
        cpu_cdfs.append(probs_to_cdf(row))
    cpu_cdfs = torch.tensor(cpu_cdfs, dtype=torch.int32)
    assert torch.equal(gpu_cdfs, cpu_cdfs), \
        "GPU CDF diverged from CPU reference — AC roundtrip would break"


def test_cpu_cdf_invariants(probs):
    """CDF[0]=0, CDF[V]=T, monotonically non-decreasing."""
    for row in probs.cpu().numpy():
        cdf = probs_to_cdf(row)
        assert cdf[0] == 0
        assert cdf[-1] == CDF_T
        assert (cdf[1:] >= cdf[:-1]).all(), "CPU CDF not monotonic"


def test_cdf_uniform_distribution_within_one_lsb():
    """Uniform probs → CDF intervals all within ±1 of T/V (apart from
    one residue bin allocated by the kernel)."""
    V = 50277
    p = torch.full((1, V), 1.0 / V, dtype=torch.float32, device="cuda")
    from krunch.codec.gpu_encode import probs_to_cdf_gpu
    cdf = probs_to_cdf_gpu(p)[0]
    diffs = (cdf[1:] - cdf[:-1]).cpu().numpy()
    expected = CDF_T // V
    # All but at most one bin should be at expected width (the kernel
    # dumps rounding residue into one bin).
    n_at_expected = int((diffs == expected).sum())
    assert n_at_expected >= V - 1, f"expected >= V-1 uniform bins, got {n_at_expected}"

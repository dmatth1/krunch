"""softmax_cdf{s_per_row, _one_row} — kernel correctness.

Encoder uses the per-row form (T x V); decoder uses the one-row form
(V,). They MUST produce bit-identical output for the same logits, or
the AC bitstream won't roundtrip.
"""
import os
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import pytest
import torch

from krunch.cpp_path import softmax_cdfs_per_row, softmax_cdf_one_row


@pytest.fixture(scope="module")
def logits():
    """Realistic-shape logits: small batch over the RWKV vocab."""
    torch.manual_seed(0)
    return torch.randn(8, 50277, dtype=torch.float16, device="cuda")


def test_one_row_matches_per_row_bit_exact(logits):
    """For each row, softmax_cdf_one_row(logits[t]) must equal
    softmax_cdfs_per_row(logits)[t] byte-for-byte."""
    cdfs = softmax_cdfs_per_row(logits)  # [T, V+1]
    for t in range(logits.size(0)):
        single = softmax_cdf_one_row(logits[t])  # [V+1]
        assert torch.equal(single, cdfs[t]), f"row {t} diverged"


def test_cdf_starts_at_zero_ends_at_T(logits):
    """A valid CDF over a discrete distribution starts at 0 and ends at
    the AC kernel's fixed-point ceiling (CDF_T)."""
    from krunch.codec.cdf import T as CDF_T
    cdfs = softmax_cdfs_per_row(logits)
    assert (cdfs[:, 0] == 0).all().item()
    assert (cdfs[:, -1] == CDF_T).all().item()


def test_cdf_monotonic_nondecreasing(logits):
    """CDF must be monotonically non-decreasing across the vocab."""
    cdfs = softmax_cdfs_per_row(logits)
    diffs = cdfs[:, 1:] - cdfs[:, :-1]
    assert (diffs >= 0).all().item()


def test_one_row_handles_extreme_logits():
    """Confident predictions (one logit dominates) should still produce
    a valid CDF — no NaN, no overflow, sum = CDF_T."""
    from krunch.codec.cdf import T as CDF_T
    logits = torch.full((50277,), -1e3, dtype=torch.float16, device="cuda")
    logits[42] = 1000.0  # token 42 is "nearly certain"
    cdf = softmax_cdf_one_row(logits)
    assert torch.isfinite(cdf).all().item()
    assert cdf[0].item() == 0
    assert cdf[-1].item() == CDF_T
    diffs = cdf[1:] - cdf[:-1]
    assert (diffs >= 0).all().item()


def test_per_row_handles_uniform_logits():
    """All-zeros logits → uniform distribution. The det_softmax_cdf
    kernel allocates rounding residue to one bin, so we check
    (a) total bin counts sum to CDF_T, (b) almost all bins are at the
    expected uniform width, and (c) no bin is zero (would break AC)."""
    from krunch.codec.cdf import T as CDF_T
    V = 50277
    logits = torch.zeros(2, V, dtype=torch.float16, device="cuda")
    cdfs = softmax_cdfs_per_row(logits)
    diffs = (cdfs[:, 1:] - cdfs[:, :-1]).float()
    assert diffs.sum(dim=1).tolist() == [CDF_T, CDF_T]
    expected = CDF_T // V  # integer truncation matches kernel
    # The kernel sets all-but-one bin to `expected` and dumps the residue
    # into the remaining bin. Verify that's the shape.
    n_at_expected = (diffs == expected).sum(dim=1)
    assert (n_at_expected >= V - 1).all().item(), \
        f"expected V-1 bins at uniform width, got n_at_expected={n_at_expected.tolist()}"
    assert (diffs > 0).all().item(), "every bin must have nonzero count for AC"

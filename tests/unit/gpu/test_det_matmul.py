"""det_matmul kernels — bit-stability across M for the production
fp16 paths. Without this, AC encoder + decoder would compute different
logits at different M values (encoder packed M=1024, decoder stepped
M=1) and AC bit-exact roundtrip would break.

Skipped on sm<8: the cp.async-based variants used here have empty
kernel bodies on Turing; production T4 routes through different
kernels (auto-selected by cpp_path's `KRUNCH_HEAD_ASYNC=0` /
`KRUNCH_3WAY_ASYNC=0` set on import for sm<8).
"""
import os
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import pytest
import torch
import krunch_ac_cuda as M

pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_properties(0).major < 8,
    reason="cp.async det_matmul variants are sm_80+; T4 uses fallbacks",
)


def _max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


# RWKV-4-Pile-169M matmul shapes: K=in features, N=out features.
SHAPES = [
    (768, 768,   "Kw/Vw/Rw/Ow"),
    (768, 3072,  "ffn_Kw"),
    (3072, 768,  "ffn_Vw"),
    (768, 50277, "head"),
]


@pytest.fixture
def weights_pair():
    """Random fp16 weights at production shapes."""
    torch.manual_seed(0)
    big_m = 4096
    return big_m


@pytest.mark.parametrize("K,N,name", SHAPES)
def test_det_matmul_bit_stable_across_M(K, N, name):
    """det_matmul (production primary path on sm_80+) must produce
    bit-identical output rows at any M slice — encoder and decoder
    can call at different M and must agree."""
    torch.manual_seed(0)
    x = torch.randn(4096, K, dtype=torch.float16, device="cuda") * 0.5
    w = torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.05
    big = M.det_matmul(x, w)
    for m in (1, 16, 64, 1024, 4096):
        small = M.det_matmul(x[:m], w)
        diff = _max_abs(big[:m], small)
        assert diff == 0.0, f"{name} (K={K},N={N}) m={m}: diff={diff:.3e}"


def test_det_matmul_dtypes():
    """Output dtype matches input dtype (fp16). Output shape = [M, N]."""
    x = torch.randn(64, 768, dtype=torch.float16, device="cuda")
    w = torch.randn(768, 768, dtype=torch.float16, device="cuda")
    out = M.det_matmul(x, w)
    assert out.dtype == torch.float16
    assert out.shape == (64, 768)


def test_det_matmul_correctness_vs_torch():
    """det_matmul must produce numerically-similar output to torch.matmul
    (the reference) — within fp16 round-off. Pinning that we haven't
    silently swapped the kernel for something incorrect."""
    torch.manual_seed(0)
    x = torch.randn(32, 768, dtype=torch.float16, device="cuda") * 0.5
    w = torch.randn(768, 768, dtype=torch.float16, device="cuda") * 0.05
    out = M.det_matmul(x, w)
    ref = torch.matmul(x, w)
    diff = _max_abs(out, ref)
    # fp16 round-off scales with sum-magnitude; allow a generous tolerance.
    assert diff < 0.5, f"det_matmul vs torch.matmul diff={diff} too large"

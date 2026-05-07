"""cuBLAS pinned-algo bit-stability across M for RWKV-4 matmul shapes.

Without algo pinning, cuBLAS picks different kernels at different M
values, producing bit-distinct outputs that break AC roundtrip.
`set_cublas_pinned_algo(99)` (CUBLAS_GEMM_ALGO0_TENSOR_OP) is what
production uses; this test pins that the chosen algo is bit-stable on
every RWKV-4-Pile-169M matmul shape.
"""
import os
os.environ.setdefault("KRUNCH_CUBLAS_PINNED", "1")
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import pytest
import torch
import krunch_ac_cuda as M


def _max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


@pytest.fixture(scope="module", autouse=True)
def _pin_default_algo():
    M.set_cublas_pinned_algo(99)  # CUBLAS_GEMM_ALGO0_TENSOR_OP — production default
    yield


@pytest.mark.parametrize("K,N,name", [
    (768,    768, "Kw/Vw/Rw/Ow"),
    (768,   3072, "ffn_Kw"),
    (3072,   768, "ffn_Vw"),
    (768,  50277, "head"),
])
def test_bit_stable_across_M(K, N, name):
    """For each layer matmul shape, the pinned algo must produce
    byte-identical rows for `det_matmul_cublas_pinned(x[:m], w)` at
    several values of m vs the full-batch slice."""
    torch.manual_seed(0)
    x = torch.randn(4096, K, dtype=torch.float16, device="cuda") * 0.5
    w = torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.05
    big = M.det_matmul_cublas_pinned(x, w)
    for m in (1, 16, 64, 1024, 4096):
        small = M.det_matmul_cublas_pinned(x[:m], w)
        diff = _max_abs(big[:m], small)
        assert diff == 0.0, f"{name} (K={K},N={N}) m={m}: diff={diff:.3e}"

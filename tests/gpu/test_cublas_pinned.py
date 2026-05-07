"""Verify cuBLAS pinned algo produces bit-identical output across M,
using the ACTUAL gemm_fp16 path (not the WMMA pybind which my
earlier version of this test accidentally exercised).

Optionally sweep over algo IDs so we can find one that's bit-stable
on every target arch (Turing T4, Ampere A10G/A100, Hopper H100).

Run with:
    python test_cublas_pinned.py            # uses default algo
    SWEEP=1 python test_cublas_pinned.py    # tries every TC algo
"""
import os
os.environ.setdefault("KRUNCH_CUBLAS_PINNED", "1")
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import torch
import krunch_ac_cuda as M


def maxabs(a, b):
    return (a.float() - b.float()).abs().max().item()


def shape_stable(K: int, N: int):
    """Returns max abs diff across M values; 0.0 means bit-stable."""
    torch.manual_seed(0)
    x = torch.randn(4096, K, dtype=torch.float16, device="cuda") * 0.5
    w = torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.05
    big = M.det_matmul_cublas_pinned(x, w)
    worst = 0.0
    for m_test in [1, 16, 64, 1024, 4096]:
        small = M.det_matmul_cublas_pinned(x[:m_test], w)
        d = maxabs(big[:m_test], small)
        worst = max(worst, d)
    return worst


def test_one_algo(algo_id: int) -> bool:
    """Run shape-stability for all RWKV-4 shapes at algo_id; return True
    if all are bit-stable."""
    M.set_cublas_pinned_algo(algo_id)
    cases = [
        (768, 768, "Kw/Vw/Rw/Ow"),
        (768, 3072, "ffn_Kw    "),
        (3072, 768, "ffn_Vw    "),
        (768, 50277, "head      "),
    ]
    print(f"\nalgo={algo_id}:")
    all_ok = True
    for K, N, name in cases:
        try:
            d = shape_stable(K, N)
        except RuntimeError as e:
            print(f"  {name}: RuntimeError ({str(e)[:80]})")
            return False
        ok = d == 0.0
        all_ok = all_ok and ok
        print(f"  {name}: max_diff={d:.6e} {'✓' if ok else '✗'}")
    return all_ok


def main():
    if os.environ.get("SWEEP") == "1":
        # CUBLAS_GEMM_ALGO0_TENSOR_OP=99 ... ALGO15_TENSOR_OP=114
        # Plus DEFAULT_TENSOR_OP=99 and DEFAULT=-1
        good = []
        for algo_id in range(99, 115):
            ok = test_one_algo(algo_id)
            if ok:
                good.append(algo_id)
        print(f"\n=== bit-stable algos: {good} ===")
    else:
        algo_id = int(os.environ.get("ALGO", "99"))  # CUBLAS_GEMM_ALGO0_TENSOR_OP
        ok = test_one_algo(algo_id)
        print("\nALL PASS" if ok else "\nSOMETHING FAILED")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

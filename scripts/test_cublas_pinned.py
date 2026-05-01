"""Verify cuBLAS pinned algo produces bit-identical output across M.

If this passes, we can replace det_matmul_tc with cuBLAS for layer
matmuls — same bit-exactness guarantee, much faster. If it fails,
we need to either pick a different algo or stay on WMMA.

Tests at the actual RWKV-4-Pile-169M shapes:
  K=768  N=768  (Kw, Vw, Rw, Ow)
  K=768  N=3072 (ffn_Kw)
  K=3072 N=768  (ffn_Vw)
  K=768  N=50277 (head — non-aligned N, important to check)
"""
import os
os.environ.setdefault("KRUNCH_CUBLAS_PINNED", "1")
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import torch
import krunch_ac_cuda as M


def maxabs(a, b):
    return (a.float() - b.float()).abs().max().item()


def test_shape_stability(K: int, N: int, name: str):
    torch.manual_seed(0)
    # Build a [4096, K] input + [K, N] weight; verify
    # det_matmul(input, weight) at any M ≤ 4096 produces the SAME
    # output[m, n] as a single big call.
    x = torch.randn(4096, K, dtype=torch.float16, device="cuda") * 0.5
    w = torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.05

    big = M.det_matmul(x, w)

    diffs = []
    for m_test in [1, 16, 64, 1024, 4096]:
        small = M.det_matmul(x[:m_test], w)
        d = maxabs(big[:m_test], small)
        diffs.append((m_test, d))
        marker = "✓" if d == 0 else "✗"
        print(f"  {name} M={m_test:>4}: diff={d:.6e} {marker}")

    return all(d == 0 for _, d in diffs)


def main():
    cases = [
        (768, 768,  "Kw/Vw/Rw/Ow"),
        (768, 3072, "ffn_Kw    "),
        (3072, 768, "ffn_Vw    "),
        (768, 50277, "head      "),
    ]
    all_ok = True
    for K, N, name in cases:
        ok = test_shape_stability(K, N, name)
        all_ok = all_ok and ok

    print("\nALL PASS — cuBLAS pinned algo is shape-stable" if all_ok
          else "\nSOMETHING FAILED — algo is NOT shape-stable; pick another or stay on WMMA")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

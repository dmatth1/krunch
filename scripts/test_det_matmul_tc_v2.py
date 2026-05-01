"""Verify det_matmul_tc_v2 (32×32 tile) is bit-stable across M.
Also benches it vs the 16×16 v1 kernel at production shapes.
"""
import os
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import time
import torch
import krunch_ac_cuda as M


def maxabs(a, b):
    return (a.float() - b.float()).abs().max().item()


def shape_stable(K: int, N: int):
    torch.manual_seed(0)
    x = torch.randn(4096, K, dtype=torch.float16, device="cuda") * 0.5
    w = torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.05
    big = M.det_matmul_tc_v2(x, w)
    worst = 0.0
    for m_test in [1, 8, 16, 32, 64, 128, 1024, 4096]:
        small = M.det_matmul_tc_v2(x[:m_test], w)
        d = maxabs(big[:m_test], small)
        worst = max(worst, d)
        marker = "✓" if d == 0 else "✗"
        print(f"    M={m_test:>4}: diff={d:.6e} {marker}")
    return worst


def bench(K, N, M_size, n_iter=20):
    torch.manual_seed(0)
    x = torch.randn(M_size, K, dtype=torch.float16, device="cuda") * 0.5
    w = torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.05
    # Warmup
    for _ in range(3):
        _ = M.det_matmul(x, w); _ = M.det_matmul_tc_v2(x, w)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        _ = M.det_matmul(x, w)
    torch.cuda.synchronize()
    t_v1 = (time.time() - t0) / n_iter * 1000
    t0 = time.time()
    for _ in range(n_iter):
        _ = M.det_matmul_tc_v2(x, w)
    torch.cuda.synchronize()
    t_v2 = (time.time() - t0) / n_iter * 1000
    print(f"    M={M_size:>5} K={K:>4} N={N:>5}: v1={t_v1:6.3f}ms v2={t_v2:6.3f}ms speedup={t_v1/t_v2:.2f}x")


def main():
    cases = [
        (768, 768),
        (768, 3072),
        (3072, 768),
        (768, 50277),
    ]
    print("=== Shape-stability v2 ===")
    all_ok = True
    for K, N in cases:
        print(f"  K={K} N={N}:")
        ok = shape_stable(K, N) == 0.0
        all_ok = all_ok and ok

    print("\n=== Bench v1 (16x16) vs v2 (32x32) ===")
    for K, N in cases:
        for M_size in [128, 1024, 6360]:
            try:
                bench(K, N, M_size)
            except RuntimeError as e:
                print(f"    M={M_size} K={K} N={N}: SKIP ({str(e)[:50]})")

    print("\n" + ("ALL PASS shape-stability" if all_ok else "SHAPE-STABILITY FAILED"))
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

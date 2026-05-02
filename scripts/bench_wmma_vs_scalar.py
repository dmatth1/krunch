"""Bench WMMA vs scalar GEMV at v2-relevant shapes.

Validates the assumption that WMMA-padded matmul beats per-thread scalar
GEMV even at M=1 (where 15/16 of WMMA tile is wasted padding). If yes,
lever 4 v3 redesign (replace v2's scalar GEMV inner loops with WMMA
fragments) is the gate-closing path. If not, lever 4 is dead and we
should switch to instance bump.

Shapes tested match the layer matmuls in RWKV-4-Pile-169M:
  KVR/Ow/ffn_R: M=?, K=768, N=768
  ffn_K:        M=?, K=768, N=3072
  ffn_V:        M=?, K=3072, N=768
M values: 1 (single-stream), 16, 64, 128 (production batched), 1024
(compress packed).
"""
import os, sys, time
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")

import torch
import krunch_ac_cuda


def bench_one(M, K, N, n_iters=100, n_warmup=10):
    """Return ms/call for the routed det_matmul (uses WMMA via det_matmul_tc
    for layer shapes; multi-warp WMMA via det_matmul_tc_mw for head shape)."""
    A = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.05
    W = torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.05

    def fn():
        return krunch_ac_cuda.det_matmul(A, W, out_dtype=torch.float16)

    for _ in range(n_warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters): fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_iters * 1000
    return ms


def bench_torch(M, K, N, n_iters=100, n_warmup=10):
    """torch.matmul (uses cuBLAS HGEMM internally; M-shape-dependent algo)."""
    A = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.05
    W = torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.05

    def fn():
        return torch.matmul(A, W)

    for _ in range(n_warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters): fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_iters * 1000
    return ms


def estimate_scalar_gemv(M, K, N):
    """Theoretical scalar GEMV time at full T4 utilization. Per output:
    K muladds. Total: M*K*N muladds. T4 fp32 peak: 8 TFLOPS = 8e9 muladds/ms.
    """
    muladds = M * K * N
    return muladds / 8e9 * 1000  # ms


def main():
    print(f"{'M':>6} {'K':>6} {'N':>6} | {'WMMA(det_matmul) ms':>20} | "
          f"{'cuBLAS HGEMM ms':>16} | {'scalar floor ms':>15}")
    print("-" * 88)
    for M in [1, 16, 64, 128, 1024]:
        for K, N in [(768, 768), (768, 3072), (3072, 768)]:
            wmma_ms = bench_one(M, K, N)
            cublas_ms = bench_torch(M, K, N)
            scalar_floor = estimate_scalar_gemv(M, K, N)
            print(f"{M:>6} {K:>6} {N:>6} | "
                  f"{wmma_ms:>20.4f} | {cublas_ms:>16.4f} | {scalar_floor:>15.4f}")
        print()


if __name__ == "__main__":
    main()

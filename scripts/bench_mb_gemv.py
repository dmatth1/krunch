"""Bench multi-block GEMV kernels vs torch's `@` (cuBLAS) and report
correctness diffs + relative speed across the 3 RWKV-4 shapes.

Pre-req: rebuild krunch_ac_cuda with mb_gemv.cu added.
"""
import time
import torch

import krunch_ac_cuda

N_WARMUP = 16
N_ITERS = 1000


def bench(label, fn, n_iters=N_ITERS):
    for _ in range(N_WARMUP): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters): fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_iters
    print(f"BENCH {label} per-call-us={dt*1e6:.2f}", flush=True)
    return dt


def test_shape(K, N, fn_kernel, label):
    device = "cuda"
    torch.manual_seed(K + N)
    x = torch.randn(K, dtype=torch.float16, device=device) * 0.1
    W = torch.randn(K, N, dtype=torch.float16, device=device) * 0.1
    y_kernel = torch.empty(N, dtype=torch.float16, device=device)
    y_torch = (x.float() @ W.float()).half()  # fp32 acc reference

    fn_kernel(x, W, y_kernel)
    torch.cuda.synchronize()

    diff = (y_kernel.float() - y_torch.float()).abs().max().item()
    rel = (y_kernel.float() - y_torch.float()).abs().mean().item()
    print(f"DIFF {label} K={K} N={N} max_abs={diff:.5f} mean_abs={rel:.5f}", flush=True)

    # Bench kernel
    t_kernel = bench(f"kernel_{label}", lambda: fn_kernel(x, W, y_kernel))
    # Bench torch @
    t_torch = bench(f"torch_at_{label}", lambda: (x @ W))
    # Bench torch fp32 @ (matches our accum dtype)
    print(f"SPEEDUP kernel/torch_at_{label} = {t_torch/t_kernel:.2f}x", flush=True)


def main():
    print("--- 768 x 768 (K, V, R, Out, ffn_R) ---")
    test_shape(768, 768, krunch_ac_cuda.mb_gemv_768x768, "768x768")

    print("--- 768 x 3072 (FFN K) ---")
    test_shape(768, 3072, krunch_ac_cuda.mb_gemv_768x3072, "768x3072")

    print("--- 3072 x 768 (FFN V) ---")
    test_shape(3072, 768, krunch_ac_cuda.mb_gemv_3072x768, "3072x768")


if __name__ == "__main__":
    main()

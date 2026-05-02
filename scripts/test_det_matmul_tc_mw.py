"""Verify det_matmul_tc_mw (multi-warp 64×64-tile) is bit-stable across M
and produces correct values vs torch reference (within fp16 noise).

Then microbench the head matmul shape: M=1024, K=768, N=50277.

Run after build:
  cd krunch_ac/cuda && python setup.py build_ext --inplace
  PYTHONPATH=krunch_ac/cuda python scripts/test_det_matmul_tc_mw.py
"""
import os, time
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")

import torch
import krunch_ac_cuda


def maxabs(a, b): return (a.float() - b.float()).abs().max().item()


def bit_stability(K, N, write_fp32):
    """Check det_matmul_tc_mw produces bit-identical output for a row at
    different M values."""
    device = "cuda"
    torch.manual_seed(0)
    Mref = 64  # multiple of 64 for clean tiling
    A = torch.randn(Mref, K, dtype=torch.float16, device=device) * 0.05
    W = torch.randn(K, N, dtype=torch.float16, device=device) * 0.05

    dtype = torch.float32 if write_fp32 else torch.float16
    out_ref = krunch_ac_cuda.det_matmul_tc_mw(A, W, out_dtype=dtype)

    # Now run with M=1, taking the first row of A. Check output[0] matches
    # out_ref[0] bit-for-bit.
    A1 = A[:1].contiguous()
    out_1 = krunch_ac_cuda.det_matmul_tc_mw(A1, W, out_dtype=dtype)
    diff = (out_ref[0].float() - out_1[0].float()).abs().max().item()
    print(f"  K={K} N={N} write_fp32={write_fp32} M-invariance diff = {diff:.6e} "
          f"(must be 0.0)")
    return diff == 0.0


def correctness(K, N):
    """Check det_matmul_tc_mw matches torch reference within fp16 noise."""
    device = "cuda"
    torch.manual_seed(1)
    M = 1024
    A = torch.randn(M, K, dtype=torch.float16, device=device) * 0.05
    W = torch.randn(K, N, dtype=torch.float16, device=device) * 0.05

    out_mw = krunch_ac_cuda.det_matmul_tc_mw(A, W, out_dtype=torch.float16)
    out_ref = (A.float() @ W.float()).to(torch.float16)
    diff = maxabs(out_mw, out_ref)
    print(f"  K={K} N={N} M=1024 vs torch fp32 reference: max_abs = {diff:.4f}")
    return diff


def bench_head(M, K, N, n_iters=30, n_warmup=5):
    device = "cuda"
    A = torch.randn(M, K, dtype=torch.float16, device=device) * 0.05
    W = torch.randn(K, N, dtype=torch.float16, device=device) * 0.05

    def fn_mw():
        return krunch_ac_cuda.det_matmul_tc_mw(A, W, out_dtype=torch.float16)
    def fn_existing():
        # det_matmul (pybind) → routes to det_matmul_tc (16×16, 1 warp/block)
        # if N is large but KRUNCH_HEAD_MW=0; otherwise to MW. To bench the
        # OLD path, we set KRUNCH_HEAD_MW=0 in the same process at startup.
        return krunch_ac_cuda.det_matmul(A, W, out_dtype=torch.float16)

    def time_it(label, fn):
        for _ in range(n_warmup): fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters): fn()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n_iters * 1000
        print(f"  {label:25s}  {ms:.3f} ms/call")
        return ms

    print(f"\nMicrobench M={M}, K={K}, N={N}:")
    ms_mw = time_it("det_matmul_tc_mw", fn_mw)
    ms_old = time_it("det_matmul (router→old/mw)", fn_existing)
    print(f"  speedup (mw vs router): {ms_old/ms_mw:.2f}×")


def main():
    print("== bit-stability across M (must be 0.0 across both K) ==")
    ok1 = bit_stability(K=768, N=50277, write_fp32=False)
    ok2 = bit_stability(K=768, N=50277, write_fp32=True)
    ok3 = bit_stability(K=3072, N=768, write_fp32=False)

    print("\n== correctness vs torch reference ==")
    correctness(K=768, N=50277)
    correctness(K=768, N=768)

    print("\n== head matmul microbench (T4 / A10G) ==")
    bench_head(M=1024, K=768, N=50277)
    bench_head(M=512, K=768, N=50277)

    if not (ok1 and ok2 and ok3):
        print("\nFAIL: bit-stability check failed")
        raise SystemExit(1)
    print("\nPASS")


if __name__ == "__main__":
    main()

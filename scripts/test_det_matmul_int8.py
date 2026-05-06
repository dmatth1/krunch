"""
Unit test + microbench for `det_matmul_int8` kernel.

Three checks:
  1. CORRECTNESS: int8 kernel result == eager dequant + fp16 matmul,
     within fp16 noise. Verifies the inner-loop dequant + WMMA math.
  2. M-INVARIANCE: int8 kernel at M=large produces same first-N-rows
     as M=small (sliced). Required for AC encoder + decoder symmetry.
  3. MICROBENCH: int8 vs det_matmul (fp16) at our matmul shapes.
     Layer matmul (M=B=16 stepped, M=T=1024 packed) and head matmul
     shape are the relevant cases.

Run on GPU instance via Docker:
  docker run --rm --gpus all \\
    -v /tmp:/tmp -v /tmp/inference.py:/app/krunch/inference.py:ro \\
    -v /tmp/cpp_path.py:/app/krunch/cpp_path.py:ro \\
    --entrypoint /opt/conda/bin/python \\
    ghcr.io/dmatth1/krunch:latest \\
    /tmp/test_det_matmul_int8.py
"""
import os
import sys
import time

os.environ.setdefault("RWKV_CUDA_ON", "1")
os.environ.setdefault("RWKV_JIT_ON", "1")


def quantize_per_channel_uint8(w_fp16):
    """Same recipe as cpp_path._quantize_dequant_uint8_per_input_channel
    but returns the int8 buffers + scale + offset (instead of dequant fp16)."""
    import torch
    assert w_fp16.dim() == 2
    w_f = w_fp16.to(torch.float32)
    mn = w_f.min(dim=1, keepdim=True).values
    mx = w_f.max(dim=1, keepdim=True).values
    scale = ((mx - mn) / 255).clamp(min=1e-8)
    w_q = ((w_f - mn) / scale).round().clamp(0, 255).to(torch.uint8)
    return w_q.contiguous(), scale.squeeze(1).to(torch.float16).contiguous(), \
           mn.squeeze(1).to(torch.float16).contiguous()


def main():
    import torch
    import krunch_ac_cuda

    if not torch.cuda.is_available():
        sys.exit("CUDA required")

    device = "cuda"
    torch.manual_seed(42)

    # Use realistic-magnitude weights (RWKV layer weights are roughly
    # ~N(0, 0.05) at the att/ffn matmul shapes).
    def make_inputs(M, K, N):
        x = torch.randn(M, K, dtype=torch.float16, device=device) * 0.5
        w_fp16 = (torch.randn(K, N, device=device) * 0.05).to(torch.float16)
        w_q, scale, offset = quantize_per_channel_uint8(w_fp16)
        return x, w_fp16, w_q, scale, offset

    def reference_dequant_matmul(x, w_q, scale, offset):
        """Eager: dequant W, then fp16 matmul (no WMMA, just torch)."""
        w_d = (w_q.to(torch.float32) * scale.to(torch.float32).unsqueeze(1)
               + offset.to(torch.float32).unsqueeze(1)).to(torch.float16)
        return (x.to(torch.float32) @ w_d.to(torch.float32)).to(torch.float16)

    def kernel_int8(x, w_q, scale, offset):
        return krunch_ac_cuda.det_matmul_int8(x, w_q, scale, offset)

    print("=" * 60)
    print("STEP 1: CORRECTNESS — int8 kernel vs reference (eager dequant)")
    print("=" * 60)
    failed = 0
    for (M, K, N) in [(16, 768, 768), (128, 768, 768), (1024, 768, 768),
                       (16, 768, 3072), (1024, 768, 3072),
                       (1024, 3072, 768), (16, 3072, 768)]:
        x, w_fp16, w_q, scale, offset = make_inputs(M, K, N)
        torch.cuda.synchronize()
        y_kernel = kernel_int8(x, w_q, scale, offset)
        torch.cuda.synchronize()
        y_ref = reference_dequant_matmul(x, w_q, scale, offset)

        diff = (y_kernel.to(torch.float32) - y_ref.to(torch.float32)).abs()
        rel_diff = diff / (y_ref.abs().to(torch.float32) + 1e-3)
        max_abs = diff.max().item()
        # fp16 matmul has ~1e-3 noise floor at our magnitudes.
        ok = max_abs < 0.5
        status = "✓" if ok else "✗"
        print(f"  {status}  M={M:5d} K={K:4d} N={N:4d}  "
              f"max_abs={max_abs:.4f}  mean_abs={diff.mean().item():.4f}  "
              f"max_rel={rel_diff.max().item():.4f}")
        if not ok:
            failed += 1

    if failed:
        sys.exit(f"FAIL: {failed} correctness checks exceeded tolerance")

    print("\n" + "=" * 60)
    print("STEP 2: M-INVARIANCE — encoder M=large vs decoder M=small")
    print("=" * 60)
    # The bit-stability gate: y[m, n] for any (m, n) must NOT depend on M.
    # AC encoder runs M=T (large packed) and decoder runs M=B (small stepped);
    # if these diverge bit-wise the AC roundtrip breaks.
    for (K, N) in [(768, 768), (768, 3072), (3072, 768)]:
        x_large = torch.randn(1024, K, dtype=torch.float16, device=device) * 0.5
        w_fp16 = (torch.randn(K, N, device=device) * 0.05).to(torch.float16)
        w_q, scale, offset = quantize_per_channel_uint8(w_fp16)
        # Compare: kernel(x_large) vs kernel(x_large[:16]) row-slice
        y_large = kernel_int8(x_large, w_q, scale, offset)
        y_small = kernel_int8(x_large[:16].contiguous(), w_q, scale, offset)
        diff = (y_large[:16].to(torch.float32) - y_small.to(torch.float32)).abs()
        max_abs = diff.max().item()
        # Must be EXACTLY 0 — any diff means encoder/decoder produce
        # different bytes at the same (m, n) and AC breaks.
        ok = max_abs == 0.0
        status = "✓" if ok else "✗"
        print(f"  {status}  K={K:4d} N={N:4d}  "
              f"max_abs(M=1024[:16] - M=16) = {max_abs}")
        if not ok:
            print(f"    → BIT-STABILITY VIOLATION; AC roundtrip would break")
            failed += 1

    if failed:
        sys.exit(f"FAIL: {failed} M-invariance checks failed")

    print("\n" + "=" * 60)
    print("STEP 3: MICROBENCH — int8 vs fp16 at production shapes")
    print("=" * 60)

    def bench(fn, repeat=50):
        for _ in range(5):  # warmup
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(repeat):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1000 / repeat  # ms / call

    print(f"  shape (M, K, N)         fp16 ms   int8 ms   speedup")
    for (M, K, N) in [(16, 768, 768), (128, 768, 768),
                       (1024, 768, 768), (1024, 768, 3072),
                       (1024, 3072, 768), (1024, 768, 50272)]:
        # 50272 is N=50277 rounded down to multiple of 8 for det_matmul_tc_async
        if K not in (768, 3072): continue
        x, w_fp16, w_q, scale, offset = make_inputs(M, K, N)
        ms_fp16 = bench(lambda: krunch_ac_cuda.det_matmul(x, w_fp16))
        ms_int8 = bench(lambda: kernel_int8(x, w_q, scale, offset))
        speedup = ms_fp16 / ms_int8 if ms_int8 > 0 else 0
        print(f"  M={M:5d} K={K:4d} N={N:5d}    "
              f"{ms_fp16:6.3f}    {ms_int8:6.3f}    {speedup:.2f}×")

    print("\n=== ALL CORRECTNESS + M-INVARIANCE TESTS PASS ===")


if __name__ == "__main__":
    main()

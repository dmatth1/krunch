"""
Unit test + microbench for `det_matmul_w8a8` and `quantize_per_row_int8`.

Three checks (Phase 2A Step 2 — meta-lesson #4):
  1. CORRECTNESS: W8A8 result matches reference (eager dequant matmul)
     within fp16 noise.
  2. M-INVARIANCE: encoder M=large packed result equals decoder M=small
     stepped result for the same first rows. Required for AC roundtrip.
  3. MICROBENCH: int8 WMMA kernel vs fp16 det_matmul at our shapes.
     Decision gate: if W8A8 isn't ≥ 1.5× faster than fp16 at our
     production shapes, don't bother integrating.

Run on GPU instance via Docker (after building the extension):
  docker run --rm --gpus all \\
    -v /tmp/krunch_ac/cuda/krunch_ac_cuda.cpython-311-x86_64-linux-gnu.so:/app/krunch_ac_cuda.cpython-311-x86_64-linux-gnu.so:ro \\
    -v /tmp/test_det_matmul_w8a8.py:/tmp/test_det_matmul_w8a8.py \\
    --entrypoint /opt/conda/bin/python \\
    ghcr.io/dmatth1/krunch:latest \\
    /tmp/test_det_matmul_w8a8.py
"""
import os
import sys
import time

os.environ.setdefault("RWKV_CUDA_ON", "1")
os.environ.setdefault("RWKV_JIT_ON", "1")


def quantize_weight_per_output_channel_int8(w_fp16):
    """Per-output-channel symmetric int8 quant. Matches
    cpp_path._quantize_dequant_int8_per_output_channel but returns
    the int8 buffers + scale (instead of dequant fp16)."""
    import torch
    assert w_fp16.dim() == 2
    w_f = w_fp16.to(torch.float32)
    max_abs = w_f.abs().max(dim=0, keepdim=True).values  # [1, N]
    scale = (max_abs / 127).clamp(min=1e-8)              # [1, N]
    w_q = (w_f / scale).round().clamp(-127, 127).to(torch.int8)
    return w_q.contiguous(), scale.squeeze(0).to(torch.float16).contiguous()


def main():
    import torch
    import krunch_ac_cuda

    if not torch.cuda.is_available():
        sys.exit("CUDA required")

    device = "cuda"
    torch.manual_seed(42)

    def make_inputs(M, K, N):
        # Realistic-magnitude weights (~N(0, 0.05)) and activations
        # (~N(0, 0.5) — RWKV layer activations after LN+premix).
        x_fp16 = torch.randn(M, K, dtype=torch.float16, device=device) * 0.5
        w_fp16 = (torch.randn(K, N, device=device) * 0.05).to(torch.float16)
        w_q, scale_w = quantize_weight_per_output_channel_int8(w_fp16)
        return x_fp16, w_fp16, w_q, scale_w

    def reference_w8a8(x_fp16, w_q, scale_w):
        """Eager: per-row int8 quantize x, then int32 matmul, then scale."""
        # Per-row act quant
        max_abs = x_fp16.to(torch.float32).abs().max(dim=1, keepdim=True).values
        scale_x = (max_abs / 127).clamp(min=1e-8)              # [M, 1]
        x_q = (x_fp16.to(torch.float32) / scale_x).round().clamp(-127, 127)
        # Promote to int32 for matmul
        x_q_i32 = x_q.to(torch.int32)
        w_q_i32 = w_q.to(torch.int32)
        # Use fp32 path for the accumulator (mathematically identical to int32
        # acc since values fit in int32 range here):
        acc = (x_q_i32.to(torch.float32) @ w_q_i32.to(torch.float32))
        # Dequant: scale_x[m] * scale_w[n] * acc
        out = scale_x * scale_w.unsqueeze(0).to(torch.float32) * acc
        return out.to(torch.float16), scale_x.squeeze(1).to(torch.float16)

    print("=" * 60)
    print("STEP 1: CORRECTNESS — W8A8 kernel vs reference")
    print("=" * 60)
    failed = 0
    for (M, K, N) in [(16, 768, 768), (128, 768, 768), (1024, 768, 768),
                       (16, 768, 3072), (1024, 768, 3072),
                       (1024, 3072, 768), (16, 3072, 768)]:
        x_fp16, w_fp16, w_q, scale_w = make_inputs(M, K, N)
        # Run kernel: quantize_per_row_int8 then det_matmul_w8a8
        x_q_kernel, scale_x_kernel = krunch_ac_cuda.quantize_per_row_int8(x_fp16)
        y_kernel = krunch_ac_cuda.det_matmul_w8a8(
            x_q_kernel, scale_x_kernel, w_q, scale_w)
        torch.cuda.synchronize()
        # Reference
        y_ref, scale_x_ref = reference_w8a8(x_fp16, w_q, scale_w)

        # Check scale_x matches
        scale_diff = (scale_x_kernel.to(torch.float32) -
                       scale_x_ref.to(torch.float32)).abs().max().item()
        # Check output matches (allow some noise from int32 acc precision
        # differences vs fp32 reference)
        diff = (y_kernel.to(torch.float32) - y_ref.to(torch.float32)).abs()
        rel = diff / (y_ref.abs().to(torch.float32) + 1e-3)
        max_abs = diff.max().item()
        # fp16 noise + small int32-vs-fp32 acc diff
        ok_scale = scale_diff < 0.001
        ok_y = max_abs < 0.5
        status = "✓" if (ok_scale and ok_y) else "✗"
        print(f"  {status}  M={M:5d} K={K:4d} N={N:4d}  "
              f"scale_diff={scale_diff:.5f}  y_max_abs={max_abs:.4f}  "
              f"y_max_rel={rel.max().item():.4f}")
        if not (ok_scale and ok_y):
            failed += 1

    if failed:
        sys.exit(f"FAIL: {failed} correctness checks exceeded tolerance")

    print("\n" + "=" * 60)
    print("STEP 2: M-INVARIANCE — encoder M=large vs decoder M=small")
    print("=" * 60)
    for (K, N) in [(768, 768), (768, 3072), (3072, 768)]:
        x_large = torch.randn(1024, K, dtype=torch.float16, device=device) * 0.5
        w_fp16 = (torch.randn(K, N, device=device) * 0.05).to(torch.float16)
        w_q, scale_w = quantize_weight_per_output_channel_int8(w_fp16)

        # Quantize full + sliced; both must produce identical first-16-rows.
        # Note: scale_x[m] depends on row m's values, NOT on M, so identical.
        x_q_large, sx_large = krunch_ac_cuda.quantize_per_row_int8(x_large)
        x_q_small, sx_small = krunch_ac_cuda.quantize_per_row_int8(
            x_large[:16].contiguous())
        sx_diff = (sx_large[:16].to(torch.float32) -
                    sx_small.to(torch.float32)).abs().max().item()
        # Quant int8 must also match (same fp16 input + same scale → same q)
        xq_diff = (x_q_large[:16].to(torch.int32) -
                    x_q_small.to(torch.int32)).abs().max().item()

        y_large = krunch_ac_cuda.det_matmul_w8a8(
            x_q_large, sx_large, w_q, scale_w)
        y_small = krunch_ac_cuda.det_matmul_w8a8(
            x_q_small, sx_small, w_q, scale_w)
        y_diff = (y_large[:16].to(torch.float32) -
                   y_small.to(torch.float32)).abs().max().item()

        ok = (sx_diff == 0.0 and xq_diff == 0 and y_diff == 0.0)
        status = "✓" if ok else "✗"
        print(f"  {status}  K={K:4d} N={N:4d}  "
              f"scale_x_diff={sx_diff}  X_q_diff={xq_diff}  y_diff={y_diff}")
        if not ok:
            print(f"    → BIT-STABILITY VIOLATION; AC roundtrip would break")
            failed += 1

    if failed:
        sys.exit(f"FAIL: {failed} M-invariance checks failed")

    print("\n" + "=" * 60)
    print("STEP 3: MICROBENCH — W8A8 vs fp16 at production shapes")
    print("=" * 60)

    def bench(fn, repeat=50):
        for _ in range(5):  # warmup
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(repeat):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1000 / repeat

    print(f"  shape (M, K, N)         fp16 ms   w8a8 ms   speedup   notes")
    for (M, K, N) in [(16, 768, 768), (128, 768, 768),
                       (1024, 768, 768), (1024, 768, 3072),
                       (1024, 3072, 768), (1024, 768, 50272)]:
        if K not in (768, 3072): continue
        x_fp16, w_fp16, w_q, scale_w = make_inputs(M, K, N)
        # Pre-quantize x once so we measure matmul, not quant overhead
        x_q_pre, sx_pre = krunch_ac_cuda.quantize_per_row_int8(x_fp16)

        ms_fp16 = bench(lambda: krunch_ac_cuda.det_matmul(x_fp16, w_fp16))
        # W8A8 with PRE-quantized x (fair comparison: just the matmul cost)
        ms_w8a8 = bench(lambda: krunch_ac_cuda.det_matmul_w8a8(
            x_q_pre, sx_pre, w_q, scale_w))
        # W8A8 with quant + matmul (realistic production cost)
        def quant_and_matmul():
            xq, sx = krunch_ac_cuda.quantize_per_row_int8(x_fp16)
            return krunch_ac_cuda.det_matmul_w8a8(xq, sx, w_q, scale_w)
        ms_w8a8_full = bench(quant_and_matmul)

        sp_matmul = ms_fp16 / ms_w8a8 if ms_w8a8 > 0 else 0
        sp_full = ms_fp16 / ms_w8a8_full if ms_w8a8_full > 0 else 0
        note = "(matmul / matmul+quant)"
        print(f"  M={M:5d} K={K:4d} N={N:5d}    "
              f"{ms_fp16:6.3f}    {ms_w8a8:6.3f}    "
              f"{sp_matmul:.2f}× / {sp_full:.2f}×")

    print("\n=== ALL CORRECTNESS + M-INVARIANCE TESTS PASS ===")


if __name__ == "__main__":
    main()

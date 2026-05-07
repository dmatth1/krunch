"""
Phase 2B precursor probe (2026-05-06): is `torch._int_mm` M-stable
on A10G across our W8A8 shapes?

If yes:
  - Drop-in replace `det_matmul_w8a8_tc_async` with `_int_mm + scale epilogue`.
  - No CUTLASS work needed — cuBLAS-class perf with the M-stability we lack.
  - Phase 2B reduces to a few hours of integration.

If no:
  - Confirms cuBLAS int8 path has the same M-instability as fp16 cuBLAS
    (already documented as the killer in TIER_3_OPTIMIZATION.md §8 and
    §12 retro). CUTLASS StreamK with kDeterministic reduction becomes the
    only path forward; budget ~1-2 days of real CUDA work for the
    Phase 2B kernel.

Test method (mirrors scripts/test_det_matmul_w8a8.py M-invariance check):
  for each (K, N) ∈ {(768,768), (768,3072), (3072,768)}:
    build random int8 X[1024, K], int8 W[K, N]
    y_large = _int_mm(X[:1024], W)        # encoder shape
    for M_small in [1, 16, 32, 128]:
      y_small = _int_mm(X[:M_small], W)   # decoder shape
      check y_large[:M_small] == y_small (must be 0.0 abs diff)
  Also microbench vs det_matmul_w8a8_tc_async at our production shapes
  to size the perf upside if it lands.
"""
import os
import sys
import time

os.environ.setdefault("RWKV_CUDA_ON", "1")
os.environ.setdefault("RWKV_JIT_ON", "1")


def main():
    import torch
    if not torch.cuda.is_available():
        sys.exit("CUDA required")

    device = "cuda"
    torch.manual_seed(42)

    # Probe 1: does torch._int_mm even exist on this build?
    has_int_mm = hasattr(torch, "_int_mm")
    print(f"torch._int_mm available: {has_int_mm}")
    if not has_int_mm:
        print("FAIL: torch._int_mm not exposed in this PyTorch build.")
        print("Try: torch.ops.aten._int_mm (private namespace)")
        try:
            torch.ops.aten._int_mm
            print("  → torch.ops.aten._int_mm IS available")
        except Exception as e:
            sys.exit(f"  → not available either: {e}")
        return

    # Probe 2: M-invariance check across our production shapes
    print("\n" + "=" * 60)
    print("M-INVARIANCE — encoder M=large vs decoder M=small (bit-exact required)")
    print("=" * 60)

    failed = 0
    for (K, N) in [(768, 768), (768, 3072), (3072, 768)]:
        x_large = torch.randint(-127, 128, (1024, K),
                                 dtype=torch.int8, device=device)
        w = torch.randint(-127, 128, (K, N),
                          dtype=torch.int8, device=device)
        y_large = torch._int_mm(x_large, w)  # [1024, N] int32
        # NOTE: torch._int_mm requires M > 16 — fails at M=1 with
        # "self.size(0) needs to be greater than 16". Our decoder runs at
        # M=1 every step, so torch._int_mm is fundamentally unusable for
        # the decoder path regardless of M-stability. We test M ∈ {17, 32,
        # 128} only to characterize what could in principle be used for
        # encoder-only — but encoder/decoder must agree bit-for-bit, so
        # mixing kernels breaks AC roundtrip.
        for M_small in [17, 32, 128]:
            x_small = x_large[:M_small].contiguous()
            y_small = torch._int_mm(x_small, w)
            diff = (y_large[:M_small].to(torch.int64) -
                    y_small.to(torch.int64)).abs().max().item()
            ok = (diff == 0)
            status = "✓" if ok else "✗"
            print(f"  {status}  K={K:4d} N={N:4d}  M_small={M_small:3d}  diff={diff}")
            if not ok:
                failed += 1

    if failed:
        print(f"\n  ✗ FAIL: {failed} M-invariance violations.")
        print("  → torch._int_mm uses M-dependent kernels (same as cuBLAS fp16).")
        print("  → Phase 2B needs full CUTLASS StreamK kDeterministic kernel.")
    else:
        print("\n  ✓ PASS: torch._int_mm is M-stable on this hardware.")
        print("  → Phase 2B reduces to integration: replace det_matmul_w8a8_tc_async")
        print("    with torch._int_mm + epilogue scaling (per-row * per-col).")

    # Probe 3: microbench vs current det_matmul_w8a8_tc_async if available
    print("\n" + "=" * 60)
    print("MICROBENCH — torch._int_mm vs det_matmul_w8a8_tc_async")
    print("=" * 60)

    try:
        import krunch_ac_cuda
        has_kac = hasattr(krunch_ac_cuda, "det_matmul_w8a8")
    except ImportError:
        has_kac = False
        print("krunch_ac_cuda not importable; bench torch._int_mm only.")

    def bench(fn, repeat=50):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeat):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000 / repeat

    print(f"  {'shape (M,K,N)':<22}  {'_int_mm ms':>11}  "
          f"{'w8a8_async ms':>15}  {'speedup':>9}")
    for (M, K, N) in [(32, 768, 768), (1024, 768, 768),
                       (1024, 768, 3072), (1024, 3072, 768)]:
        x_q = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=device)
        w_q = torch.randint(-127, 128, (K, N), dtype=torch.int8, device=device)
        sx = torch.ones(M, dtype=torch.float16, device=device) * 0.01
        sw = torch.ones(N, dtype=torch.float16, device=device) * 0.001

        ms_native = bench(lambda: torch._int_mm(x_q, w_q))
        if has_kac:
            ms_w8a8 = bench(lambda: krunch_ac_cuda.det_matmul_w8a8(
                x_q, sx, w_q, sw))
            sp = ms_w8a8 / ms_native if ms_native > 0 else 0
            print(f"  M={M:5d} K={K:4d} N={N:5d}  {ms_native:11.4f}  "
                  f"{ms_w8a8:15.4f}  {sp:9.2f}×")
        else:
            print(f"  M={M:5d} K={K:4d} N={N:5d}  {ms_native:11.4f}  "
                  f"{'n/a':>15}  {'n/a':>9}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()

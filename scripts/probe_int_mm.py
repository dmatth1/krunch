"""
Phase 2B reconnaissance: torch._int_mm vs det_matmul_w8a8_tc_async.

Two questions to answer before committing to integration work:

  1. Is torch._int_mm output bit-stable across M ∈ {17, 32, 64, 161,
     256, 1024} for the production W8A8 inputs (per-row int8
     activations × per-col int8 weights)?

  2. How does the (_int_mm + torch scaling) pipeline's wall time
     compare to launch_det_matmul_w8a8_tc_async at the same shapes?

If (1) holds + (2) shows ≥1.15×, Phase 2B is worth implementing.
If either fails, pivot to NEXT-4 / A1.

Usage (on a g5.xlarge A10G):

    python3 scripts/probe_int_mm.py

Per docs/TIER_3_OPTIMIZATION.md NEXT plan after the 2026-05-08
post-cleanup phase profile.
"""

import time

import torch
import krunch_ac_cuda  # noqa: F401  pull in the kernel module


def _quant_per_row_int8(x_fp16):
    """Mirror cpp_path's per-row int8 quant (matches
    launch_quantize_per_row_int8). Returns (X_q [M,K] int8, scale [M] fp16)."""
    abs_max = x_fp16.abs().amax(dim=1)
    scale = abs_max / 127.0
    scale = scale.clamp_min(1e-8)  # avoid div-by-0 on all-zero rows
    X_q = (x_fp16 / scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    return X_q, scale.to(torch.float16)


def _quant_per_col_int8(w_fp16):
    """Mirror init_weights' per-col int8 quant for W8A8 weights.
    Returns (W_q [K,N] int8, scale [N] fp16)."""
    abs_max = w_fp16.abs().amax(dim=0)
    scale = abs_max / 127.0
    scale = scale.clamp_min(1e-8)
    W_q = (w_fp16 / scale.unsqueeze(0)).round().clamp(-127, 127).to(torch.int8)
    return W_q, scale.to(torch.float16)


def _gemm_via_int_mm(Xq, sx, Wq, sw, out_dtype=torch.float16):
    """torch._int_mm pipeline. Bytes are deterministic on a given
    PyTorch+CUDA build but may DIFFER from det_matmul_w8a8_tc_async.
    Both encoder + decoder using THIS path is bit-exact by construction."""
    int_acc = torch._int_mm(Xq, Wq)  # [M, N] int32
    out = int_acc.to(torch.float32)
    out = out * sx.unsqueeze(1).to(torch.float32)
    out = out * sw.unsqueeze(0).to(torch.float32)
    return out.to(out_dtype)


def _gemm_via_kernel(Xq, sx, Wq_int8, sw, M, K, N, out_dtype=torch.float16):
    """Production W8A8 kernel path (det_matmul_w8a8_tc_async on sm_80+)."""
    out = torch.empty(M, N, dtype=out_dtype, device="cuda")
    write_fp32 = 1 if out_dtype == torch.float32 else 0
    krunch_ac_cuda.det_matmul_w8a8(
        Xq.contiguous(), sx.contiguous(),
        Wq_int8.contiguous(), sw.contiguous(),
        out, write_fp32, M, K, N,
    )
    return out


def _bench(fn, n_warmup=10, n_iters=100):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / n_iters


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name} sm_{p.major}{p.minor}, "
          f"{p.multi_processor_count} SMs")

    # Production layer-matmul shapes: K∈{768,3072}, N∈{768,3072},
    # M varies between encoder packed (1024) and decoder batched (~B chunks).
    Ms = [17, 32, 64, 128, 161, 256, 512, 1024]
    shapes = [
        ("KVR/Ow", 768, 768),
        ("ffn_K",  768, 3072),
        ("ffn_V",  3072, 768),
    ]

    torch.manual_seed(42)

    print("\n==== M-stability probe (output diff at fp16 across M) ====")
    print("Want: per-row output for sym M is identical across M values.")
    print("If different, _int_mm output isn't M-stable in our usage; "
          "Phase 2B requires extra care.\n")
    for name, K, N in shapes:
        print(f"-- {name} (K={K}, N={N}) --")
        # Build a single big input + weight; sub-slice to test M-stability.
        x_big = torch.randn(max(Ms), K, dtype=torch.float16, device="cuda")
        w_big = torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.02
        Wq, sw = _quant_per_col_int8(w_big)
        # int_mm reference: take row 0 at full M=max(Ms) → reference for top row
        ref_M = max(Ms)
        Xq_ref, sx_ref = _quant_per_row_int8(x_big[:ref_M])
        ref_out = _gemm_via_int_mm(Xq_ref, sx_ref, Wq, sw,
                                     out_dtype=torch.float16)
        for M in Ms:
            Xq_M, sx_M = _quant_per_row_int8(x_big[:M])
            out_M = _gemm_via_int_mm(Xq_M, sx_M, Wq, sw,
                                      out_dtype=torch.float16)
            # Compare row 0 (same input, different M slice) to ref row 0.
            # Quant output for x[:M] uses the per-row max-abs of THAT slice;
            # since we use row 0's own data only, sx[0] is identical, so
            # the row 0 output should be identical across M.
            row0_ref = ref_out[0]
            row0_M = out_M[0]
            equal = torch.equal(row0_ref, row0_M)
            max_abs = float((row0_ref.float() - row0_M.float()).abs().max())
            print(f"  M={M:4d}: row0 byte-equal={equal}  max_abs={max_abs:.3e}")
        # Also compare row 0 of int_mm vs row 0 of det_matmul_w8a8 kernel.
        Xq_ker, sx_ker = _quant_per_row_int8(x_big[:1024])
        ker_out = _gemm_via_kernel(Xq_ker, sx_ker, Wq, sw,
                                     1024, K, N, out_dtype=torch.float16)
        equal_ref = torch.equal(ker_out[0], ref_out[0])
        max_abs_ref = float((ker_out[0].float() - ref_out[0].float()).abs().max())
        print(f"  vs det_matmul_w8a8 kernel @ M=1024: "
              f"byte-equal={equal_ref} max_abs={max_abs_ref:.3e}\n")

    print("\n==== Wall-time bench: _int_mm + scaling vs det_matmul_w8a8_tc_async ====")
    print(f"{'shape':<32} {'M':>5} {'kernel ms':>12} {'_int_mm ms':>12} {'speedup':>10}")
    for name, K, N in shapes:
        x_big = torch.randn(max(Ms), K, dtype=torch.float16, device="cuda")
        w_big = torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.02
        Wq, sw = _quant_per_col_int8(w_big)
        for M in [17, 161, 256, 1024]:
            Xq, sx = _quant_per_row_int8(x_big[:M])
            t_kernel = _bench(lambda: _gemm_via_kernel(
                Xq, sx, Wq, sw, M, K, N, out_dtype=torch.float16))
            t_intmm = _bench(lambda: _gemm_via_int_mm(
                Xq, sx, Wq, sw, out_dtype=torch.float16))
            speedup = t_kernel / t_intmm if t_intmm > 0 else float("inf")
            tag = f"{name} K={K} N={N}"
            print(f"{tag:<32} {M:>5} {t_kernel:>12.4f} "
                  f"{t_intmm:>12.4f} {speedup:>10.2f}x")


if __name__ == "__main__":
    main()

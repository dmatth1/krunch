"""uint8 weight matmul microbench. Requires sm_80+ (A10G/A100/L40S).

Compares fp16, bf16, uint8 paths on RWKV-4 layer matmul shapes.
Tells us whether the uint8 lift (predicted 1.5-2×) materializes.

Quantization scheme: per-input-channel asymmetric (matches
rwkv-cpp-accelerated). Validated 0.75-1.25% rel-RMSE on real RWKV-4
weights — see calibrate_uint8_weights.py.
"""
import os, sys
import torch
import krunch_ac_cuda


def time_kernel(fn, n_iters=30, n_warmup=5):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iters


def maxabs(a, b): return (a.float() - b.float()).abs().max().item()


def quantize_per_input(W: torch.Tensor):
    W_f = W.float()
    w_min = W_f.min(dim=1).values
    w_max = W_f.max(dim=1).values
    scale = (w_max - w_min) / 255.0
    scale = scale.clamp(min=1e-12)
    offset = w_min
    Q = ((W_f - offset.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 255).to(torch.uint8)
    return Q.contiguous(), scale.to(torch.float16).contiguous(), offset.to(torch.float16).contiguous()


def shape_bench(M, K, N, label, device):
    torch.manual_seed(0)
    x_f32 = torch.randn(M, K, dtype=torch.float32, device=device) * 0.1
    W_f32 = torch.randn(K, N, dtype=torch.float32, device=device) * 0.05
    x_h = x_f32.half()
    W_h = W_f32.half()
    x_b = x_f32.to(torch.bfloat16)
    W_b = W_f32.to(torch.bfloat16)
    W_u8, sc, off = quantize_per_input(W_h)

    # Sanity dequant check
    W_dq = W_u8.float() * sc.float().unsqueeze(1) + off.float().unsqueeze(1)
    quant_err = maxabs(W_dq, W_h)

    y_h = krunch_ac_cuda.det_matmul(x_h, W_h)
    y_b = krunch_ac_cuda.det_matmul_bf16(x_b, W_b)
    y_u = krunch_ac_cuda.det_matmul_uint8(x_h, W_u8, sc, off)
    y_ref = (x_f32 @ W_f32)

    diff_h = maxabs(y_h, y_ref)
    diff_b = maxabs(y_b, y_ref)
    diff_u = maxabs(y_u, y_ref)
    diff_uh = maxabs(y_u, y_h)

    t_h = time_kernel(lambda: krunch_ac_cuda.det_matmul(x_h, W_h))
    t_b = time_kernel(lambda: krunch_ac_cuda.det_matmul_bf16(x_b, W_b))
    t_u = time_kernel(lambda: krunch_ac_cuda.det_matmul_uint8(x_h, W_u8, sc, off))

    print(f"  {label} M={M} K={K} N={N}  (W quant max-abs: {quant_err:.3e})")
    print(f"    fp16:  {t_h:.3f} ms   diff vs fp32 ref: {diff_h:.3e}")
    print(f"    bf16:  {t_b:.3f} ms   diff vs fp32 ref: {diff_b:.3e}   speedup vs fp16: {t_h/t_b:.2f}x")
    print(f"    uint8: {t_u:.3f} ms   diff vs fp32 ref: {diff_u:.3e}   speedup vs fp16: {t_h/t_u:.2f}x")
    print(f"    uint8 vs fp16 max-abs: {diff_uh:.3e}")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(2)
    cap = torch.cuda.get_device_capability()
    print(f"Device: {torch.cuda.get_device_name()}  capability sm_{cap[0]}{cap[1]}")
    if cap[0] < 8:
        print("ERROR: cp.async + bf16 require sm_80+. T4 won't work.")
        sys.exit(2)
    device = "cuda"

    print("\n=== RWKV-4 layer matmul shapes (fp16 vs bf16 vs uint8) ===")
    print("\nCompress (packed encoder, M=1024):")
    for M, K, N, lbl in [
        (1024, 768, 768,  "Kw/Vw/Rw/Ow"),
        (1024, 768, 3072, "ffn_Kw"),
        (1024, 3072, 768, "ffn_Vw"),
    ]:
        shape_bench(M, K, N, lbl, device)

    print("\nDecompress (stepped, M=128):")
    for M, K, N, lbl in [
        (128, 768, 768,  "Kw/Vw/Rw/Ow"),
        (128, 768, 3072, "ffn_Kw"),
        (128, 3072, 768, "ffn_Vw"),
    ]:
        shape_bench(M, K, N, lbl, device)


if __name__ == "__main__":
    main()

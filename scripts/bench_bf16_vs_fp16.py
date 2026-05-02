"""bf16 vs fp16 microbench. Requires sm_80+ (A10G/A100/L40S).

Measures speed + numerical drift across RWKV-4 layer matmul shapes.
Tells us whether the bf16 swap is (a) faster, (b) numerically close
enough to be neutral on ratio, and (c) whether bytes will differ.

Bytes will differ from current fp16 codec by definition (different
dtype throughout). The question is whether the resulting CDFs are
close enough that compressed-size moves <1%, in which case bf16 is
worth shipping as v2 model_id `RWKV169M_bf16`.
"""
import os, sys, time
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


def shape_bench(M, K, N, label, device):
    # Inputs share storage but cast to fp16 / bf16 (tiny rounding diff
    # in inputs themselves, but inherent to dtype swap).
    torch.manual_seed(0)
    x_f32 = torch.randn(M, K, dtype=torch.float32, device=device) * 0.1
    W_f32 = torch.randn(K, N, dtype=torch.float32, device=device) * 0.05
    x_h = x_f32.half()
    W_h = W_f32.half()
    x_b = x_f32.to(torch.bfloat16)
    W_b = W_f32.to(torch.bfloat16)

    # fp16 path (det_matmul routes to det_matmul_tc_async on N%8==0)
    y_h = krunch_ac_cuda.det_matmul(x_h, W_h)
    y_b = krunch_ac_cuda.det_matmul_bf16(x_b, W_b)
    # Reference: fp32 torch matmul (ground truth)
    y_ref = (x_f32 @ W_f32)

    diff_h = maxabs(y_h, y_ref)
    diff_b = maxabs(y_b, y_ref)
    diff_hb = maxabs(y_h, y_b)

    t_h = time_kernel(lambda: krunch_ac_cuda.det_matmul(x_h, W_h))
    t_b = time_kernel(lambda: krunch_ac_cuda.det_matmul_bf16(x_b, W_b))

    print(f"  {label} M={M} K={K} N={N}")
    print(f"    fp16: {t_h:.3f} ms   max-abs vs fp32 ref: {diff_h:.3e}")
    print(f"    bf16: {t_b:.3f} ms   max-abs vs fp32 ref: {diff_b:.3e}")
    print(f"    bf16/fp16 speed: {t_h/t_b:.2f}x   bf16 vs fp16 max-abs: {diff_hb:.3e}")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(2)
    cap = torch.cuda.get_device_capability()
    print(f"Device: {torch.cuda.get_device_name()}  capability sm_{cap[0]}{cap[1]}")
    if cap[0] < 8:
        print("ERROR: bf16 WMMA requires sm_80+ (A10G/A100/L40S/H100). T4 sm_75 won't work.")
        sys.exit(2)
    device = "cuda"

    print("\n=== RWKV-4 layer matmul shapes ===")
    print("\nCompress (packed encoder, M=1024):")
    for M, K, N, lbl in [
        (1024, 768, 768,  "Kw/Vw/Rw/Ow"),
        (1024, 768, 3072, "ffn_Kw"),
        (1024, 3072, 768, "ffn_Vw"),
    ]:
        shape_bench(M, K, N, lbl, device)

    print("\nDecompress (stepped, M=128 cross-chunk batch):")
    for M, K, N, lbl in [
        (128, 768, 768,  "Kw/Vw/Rw/Ow"),
        (128, 768, 3072, "ffn_Kw"),
        (128, 3072, 768, "ffn_Vw"),
    ]:
        shape_bench(M, K, N, lbl, device)


if __name__ == "__main__":
    main()

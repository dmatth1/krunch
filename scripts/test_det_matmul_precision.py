"""Compare det_matmul kernel output precision vs cuBLAS for the
exact shapes used in RWKV-4-Pile-169M, against an fp64 ground-truth
reference. If det_matmul is meaningfully less accurate than cuBLAS
for a given shape, that explains the ratio degradation observed in
the chunk benchmark.
"""
import torch
import krunch_ac_cuda as M


def report(label, y, ref, raw_x, raw_W):
    fp32_ref = ref.float()
    diff = (y.float() - fp32_ref).abs()
    rel = diff / (fp32_ref.abs() + 1e-6)
    print(f"  {label}: abs_max={diff.max().item():.4e} "
          f"abs_mean={diff.mean().item():.4e} "
          f"rel_max={rel.max().item():.4e} "
          f"rel_mean={rel.mean().item():.4e}")


def main():
    device = "cuda"
    torch.manual_seed(0)
    shapes = [
        ("KW (att)",     1, 768, 768),
        ("KW.T (out)",   1, 768, 768),
        ("FFN K",        1, 768, 3072),
        ("FFN V",        1, 3072, 768),
        ("HEAD",         1, 768, 50277),
        ("HEAD T=32",   32, 768, 50277),
        ("HEAD T=1024", 1024, 768, 50277),
    ]
    for name, Mn, K, N in shapes:
        x = (torch.randn(Mn, K, dtype=torch.float16, device=device)) * 0.5
        W = (torch.randn(K, N, dtype=torch.float16, device=device)) * 0.05

        # fp64 reference (ground truth)
        ref64 = (x.double() @ W.double())

        # det_matmul output (fp16)
        y_det = M.det_matmul(x, W)

        # cuBLAS via PyTorch @ (also fp16 output)
        y_cublas = (x @ W)

        print(f"{name}  M={Mn} K={K} N={N}")
        report("det_matmul ", y_det, ref64, x, W)
        report("cuBLAS @   ", y_cublas, ref64, x, W)
        print()


if __name__ == "__main__":
    main()

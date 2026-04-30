"""Direct test: deterministic matmul kernel must be M-invariant.

y_step[0]  = det_matmul(x[0:1], W)
y_pack[0]  = det_matmul(x,      W)[0:1]
If kernel is shape-invariant, these are bit-identical (max abs diff = 0).
For comparison we also run cuBLAS (which is known to drift).
"""
import torch
import krunch_ac_cuda as M

def main():
    device = "cuda"
    torch.manual_seed(0)
    for (Mn, K, N) in [(64, 768, 768), (64, 768, 3072), (64, 3072, 768)]:
        x = torch.randn(Mn, K, dtype=torch.float16, device=device)
        W = torch.randn(K, N, dtype=torch.float16, device=device) * 0.05

        # Deterministic kernel
        y_pack = M.det_matmul(x, W)              # [Mn, N]
        y_step = M.det_matmul(x[0:1], W)         # [1, N]
        d_det = (y_pack[0:1].float() - y_step.float()).abs().max().item()

        # cuBLAS reference (expected to drift)
        c_pack = (x @ W)
        c_step = (x[0:1] @ W)
        d_cublas = (c_pack[0:1].float() - c_step.float()).abs().max().item()

        # Numerical accuracy of det vs fp32 ref
        ref = (x.float() @ W.float()).half()
        d_acc = (y_pack.float() - ref.float()).abs().max().item()

        print(f"M={Mn} K={K} N={N}: "
              f"det_M_invariant_max_abs={d_det:.6f}  "
              f"cublas_M_drift_max_abs={d_cublas:.6f}  "
              f"det_vs_fp32ref_max_abs={d_acc:.6f}",
              flush=True)

        ok = "PASS" if d_det == 0.0 else "FAIL"
        print(f"  -> {ok} (M-invariance)", flush=True)

if __name__ == "__main__":
    main()

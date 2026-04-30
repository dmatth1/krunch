"""Verify det_softmax_cdf:
1. Bit-identical to per-row Python loop (probs_to_cdf_gpu(softmax(.))).
2. Bit-identical between [T, V] batched call and [1, V] single call.
3. Faster than the per-row Python loop on a realistic (T=1024, V=50277).
"""
import time
import torch
import krunch_ac_cuda as M
from krunch_ac.gpu_encode import probs_to_cdf_gpu
from krunch_ac.cdf import T as CDF_T

def main():
    device = "cuda"
    V = 50277
    Ts = [1, 32, 1024]
    torch.manual_seed(0)

    for T in Ts:
        logits = torch.randn(T, V, dtype=torch.float16, device=device) * 4.0

        # Reference: per-row Python (softmax + probs_to_cdf_gpu)
        ref_rows = []
        for t in range(T):
            p = torch.softmax(logits[t:t+1].float(), dim=-1)
            ref_rows.append(probs_to_cdf_gpu(p).contiguous())
        ref = torch.cat(ref_rows, dim=0)

        # Kernel: batched
        out = M.det_softmax_cdf(logits, CDF_T)

        # Equality
        diff = (out.long() - ref.long()).abs()
        max_diff = diff.max().item()
        n_rows_diff = ((out != ref).any(dim=-1)).sum().item()
        print(f"T={T:>4}: max_int_diff={max_diff}  rows_diff={n_rows_diff}/{T}")

        # Single-row vs batched for the first row of T>1 case
        if T > 1:
            single = M.det_softmax_cdf(logits[0:1], CDF_T)
            row_diff = (single.long() - out[0:1].long()).abs().max().item()
            print(f"       batched[0] vs single([1,V]) diff: {row_diff}")

        # Timing comparison
        if T == 1024:
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(3):
                _ = M.det_softmax_cdf(logits, CDF_T)
            torch.cuda.synchronize()
            t_kernel = (time.time() - t0) / 3

            t0 = time.time()
            for _ in range(3):
                rows = []
                for t in range(T):
                    p = torch.softmax(logits[t:t+1].float(), dim=-1)
                    rows.append(probs_to_cdf_gpu(p).contiguous())
                _ = torch.cat(rows, dim=0)
            torch.cuda.synchronize()
            t_python = (time.time() - t0) / 3
            print(f"  T=1024 timing: kernel={t_kernel*1000:.1f}ms "
                  f"per-row-python={t_python*1000:.1f}ms "
                  f"speedup={t_python/t_kernel:.1f}×")


if __name__ == "__main__":
    main()

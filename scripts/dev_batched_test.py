"""Test batched RWKV-4 forward correctness + throughput on the live instance."""

import os
import sys
import time

import torch


def main():
    sys.path.insert(0, "/home/ubuntu")
    sys.path.insert(0, "/home/ubuntu/krunch_ac/cuda")
    os.environ.setdefault("RWKV_JIT_ON", "1")
    os.environ.setdefault("RWKV_CUDA_ON", "1")

    from rwkv.model import RWKV
    from krunch.batched_rwkv4 import forward_batched, init_state_batched

    m = RWKV(model="/tmp/krunch/models/RWKV-4-Pile-169M-20220807-8023",
             strategy="cuda fp16", verbose=False)
    print(f"loaded v{m.version}, n_layer={m.args.n_layer}, n_embd={m.args.n_embd}")

    # 1) Correctness: batched B=1 should match unbatched single-stream
    tokens = list(range(64))  # 64 token sequence
    logits_unb, state_unb = m.forward(tokens, None, full_output=True)
    if not isinstance(logits_unb, torch.Tensor):
        logits_unb = torch.as_tensor(logits_unb, device="cuda")

    logits_b, state_b = forward_batched(m, [tokens], None, full_output=True)
    print(f"unbatched logits shape: {logits_unb.shape}")
    print(f"batched logits shape:   {logits_b.shape}")

    diff = (logits_unb.float() - logits_b[0].float()).abs()
    print(f"max abs diff (B=1 vs unbatched): {diff.max().item():.6f}")
    print(f"mean abs diff: {diff.mean().item():.6f}")

    rel = diff / (logits_unb.float().abs().mean() + 1e-6)
    print(f"max rel diff: {rel.max().item():.6f}")

    # 2) Throughput sweep
    tokens = list(range(1024))
    for B in [1, 2, 4, 8]:
        batches = [tokens] * B
        # warmup
        torch.cuda.synchronize()
        for _ in range(2):
            logits, state = forward_batched(m, batches, None, full_output=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        N = 5
        for _ in range(N):
            logits, state = forward_batched(m, batches, None, full_output=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / N
        total_tokens = B * 1024
        print(f"B={B}: {dt*1000:.1f}ms/call, {total_tokens/dt:.0f} tok/s aggregate "
              f"({1024/dt:.0f} tok/s per stream, scaling={1024/dt/95000:.2f}x)")


if __name__ == "__main__":
    main()

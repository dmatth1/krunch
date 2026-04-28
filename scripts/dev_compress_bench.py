"""
Developer benchmark: compress one chunk of WildChat with the full
inference path on the host (no Docker), report per-stage timings
and ratio. Run on the GPU dev instance via:

    PYTHONPATH=/home/ubuntu:/home/ubuntu/krunch_ac/cuda /opt/pytorch/bin/python3 \
      scripts/dev_compress_bench.py --sample /tmp/sample.bin --chunk-mb 1 --warmup 2
"""

import argparse
import os
import sys
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--chunk-mb", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--n-chunks", type=int, default=2)
    ap.add_argument("--model-dir", default="/tmp/krunch/models")
    args = ap.parse_args()

    os.environ.setdefault("RWKV_JIT_ON", "1")
    os.environ.setdefault("RWKV_CUDA_ON", "1")

    sys.path.insert(0, "/home/ubuntu")
    sys.path.insert(0, "/home/ubuntu/krunch_ac/cuda")
    import krunch_ac_cuda
    from krunch_ac.gpu_encode import probs_to_cdf_gpu

    from rwkv.model import RWKV
    from tokenizers import Tokenizer

    print(f"loading model from {args.model_dir} ...")
    t0 = time.time()
    model_path = os.path.join(args.model_dir, "RWKV-4-Pile-169M-20220807-8023")
    m = RWKV(model=model_path, strategy="cuda fp16", verbose=False)
    tok = Tokenizer.from_file(os.path.join(args.model_dir, "20B_tokenizer.json"))
    print(f"  model+tokenizer loaded in {time.time()-t0:.1f}s")

    chunk_size = args.chunk_mb * 1024 * 1024
    raw = open(args.sample, "rb").read(chunk_size * (args.warmup + args.n_chunks))

    SEQ_BATCH = 1024
    BOS = 0

    compiled_cdf = torch.compile(probs_to_cdf_gpu)

    def compress_chunk(data: bytes) -> bytes:
        text = data.decode("utf-8", errors="replace")
        tokens = tok.encode(text).ids
        n = len(tokens)
        full_input = [BOS] + tokens[:-1]
        tokens_arr_gpu = torch.as_tensor(tokens, dtype=torch.int32, device="cuda")
        cap = max(len(data) * 2, 64 << 10)
        out = torch.zeros(cap, dtype=torch.uint8, device="cuda")
        st = torch.zeros(4, dtype=torch.uint32, device="cuda"); st[1] = 0xFFFFFFFF
        rwkv_state = None
        pos = 0
        # Keep it simple synchronous — pipelining was tried and gave
        # marginal gains on this hardware (210 KB/s vs 197 sync). The
        # forward pass dominates and AC stages don't overlap as much
        # as expected, likely from torch caching-allocator host work
        # inside the loop. Real wins are now in the forward path.
        for i in range(0, n, SEQ_BATCH):
            batch = full_input[i:i + SEQ_BATCH]
            logits, rwkv_state = m.forward(batch, rwkv_state, full_output=True)
            if not isinstance(logits, torch.Tensor):
                logits = torch.as_tensor(logits, device="cuda")
            B = logits.size(0)
            with torch.no_grad():
                # softmax in fp32 (fp16 softmax is fine for RWKV but the
                # tail-token probabilities lose precision, hurting ratio).
                probs = torch.softmax(logits.float(), dim=-1)
                cdf = compiled_cdf(probs).contiguous()
            sym = tokens_arr_gpu[pos:pos + B].contiguous()
            krunch_ac_cuda.encode_step(cdf, sym, out, st)
            pos += B
        krunch_ac_cuda.encode_finalize(out, st)
        torch.cuda.synchronize()
        bit_offset = int(st[3].item())
        nb = (bit_offset + 7) // 8
        return bytes(out[:nb].cpu().numpy()), [], n

    # Warmup
    print(f"warmup ({args.warmup} chunks)...")
    for w in range(args.warmup):
        chunk = raw[w * chunk_size:(w + 1) * chunk_size]
        bs, _, _ = compress_chunk(chunk)
        print(f"  warmup chunk {w+1}: {len(bs)} bytes / {len(chunk)} bytes "
              f"= ratio {len(bs)/len(chunk):.4f}")

    # Steady-state measurement
    print(f"\nmeasured ({args.n_chunks} chunks)...")
    total_in = 0; total_out = 0; total_t = 0.0; all_batch_ms = []
    for c in range(args.n_chunks):
        chunk = raw[(args.warmup + c) * chunk_size:(args.warmup + c + 1) * chunk_size]
        torch.cuda.synchronize(); t_c = time.perf_counter()
        bs, batch_ms, ntok = compress_chunk(chunk)
        torch.cuda.synchronize(); dt_c = time.perf_counter() - t_c
        total_in += len(chunk); total_out += len(bs); total_t += dt_c
        all_batch_ms.extend(batch_ms)
        print(f"  chunk {c+1}: {len(chunk)} -> {len(bs)} "
              f"(ratio {len(bs)/len(chunk):.4f}) in {dt_c:.2f}s ({len(chunk)/dt_c/1024:.0f} KB/s) "
              f"{ntok} tokens, {len(batch_ms)} batches")

    print(f"\n=== Summary ===")
    print(f"input:        {total_in/1024:.0f} KB")
    print(f"output:       {total_out/1024:.0f} KB")
    print(f"ratio:        {total_out/total_in:.4f}")
    print(f"compress:     {total_in/total_t/1024:.0f} KB/s")
    if all_batch_ms:
        print(f"per-batch ms: median={np.median(all_batch_ms):.1f} "
              f"p90={np.percentile(all_batch_ms, 90):.1f} "
              f"max={max(all_batch_ms):.1f}")


if __name__ == "__main__":
    main()

"""
Experiment: does aggregate compress throughput scale by running N
chunks in parallel via Python threads, each on its own CUDA stream?

If yes -> ship multi-threaded compress as the v1.1 throughput win.
If no -> need true batched WKV kernel (multi-day kernel rewrite).
"""

import argparse
import os
import sys
import threading
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--chunk-mb", type=int, default=1)
    ap.add_argument("--n-chunks", type=int, default=4)
    ap.add_argument("--n-threads", type=int, default=4)
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

    print(f"loading model ...")
    t0 = time.time()
    model_path = os.path.join(args.model_dir, "RWKV-4-Pile-169M-20220807-8023")
    m = RWKV(model=model_path, strategy="cuda fp16", verbose=False)
    tok = Tokenizer.from_file(os.path.join(args.model_dir, "20B_tokenizer.json"))
    print(f"  loaded in {time.time()-t0:.1f}s")

    chunk_size = args.chunk_mb * 1024 * 1024
    raw = open(args.sample, "rb").read(chunk_size * args.n_chunks)
    chunks_data = [raw[i*chunk_size:(i+1)*chunk_size] for i in range(args.n_chunks)]

    SEQ_BATCH = 1024
    BOS = 0
    compiled_cdf = torch.compile(probs_to_cdf_gpu)

    # The rwkv WKV kernel doesn't respect torch's stream context, so
    # all threads share the default stream. Concurrency comes from
    # overlapping Python work (allocator, kernel launches) across
    # threads while the GPU serializes kernels — the GIL is released
    # during CUDA calls, so threads make progress in parallel.
    def compress_chunk(data):
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
        for i in range(0, n, SEQ_BATCH):
            batch = full_input[i:i + SEQ_BATCH]
            logits, rwkv_state = m.forward(batch, rwkv_state, full_output=True)
            if not isinstance(logits, torch.Tensor):
                logits = torch.as_tensor(logits, device="cuda")
            B = logits.size(0)
            with torch.no_grad():
                probs = torch.softmax(logits.float(), dim=-1)
                cdf = compiled_cdf(probs).contiguous()
            sym = tokens_arr_gpu[pos:pos + B].contiguous()
            krunch_ac_cuda.encode_step(cdf, sym, out, st)
            pos += B
        krunch_ac_cuda.encode_finalize(out, st)
        torch.cuda.synchronize()
        bit_offset = int(st[3].item())
        nb = (bit_offset + 7) // 8
        return bytes(out[:nb].cpu().numpy()), n

    # Warmup compile path
    print("warmup ...")
    bs, _ = compress_chunk(chunks_data[0])
    print(f"  ratio={len(bs)/chunk_size:.4f}")

    # Sequential baseline
    print(f"\nSequential ({args.n_chunks} chunks, 1 thread):")
    t0 = time.perf_counter()
    out_bytes_seq = 0
    for ch in chunks_data:
        bs, _ = compress_chunk(ch)
        out_bytes_seq += len(bs)
    torch.cuda.synchronize()
    dt_seq = time.perf_counter() - t0
    in_bytes = chunk_size * args.n_chunks
    print(f"  in={in_bytes/1024:.0f}KB out={out_bytes_seq/1024:.0f}KB "
          f"ratio={out_bytes_seq/in_bytes:.4f} time={dt_seq:.2f}s "
          f"throughput={in_bytes/dt_seq/1024:.0f}KB/s")

    # Threaded: N threads, share default stream, GIL releases on CUDA calls
    print(f"\nThreaded ({args.n_chunks} chunks, {args.n_threads} threads):")
    results = [None] * args.n_chunks
    def worker(ci):
        bs, n = compress_chunk(chunks_data[ci])
        results[ci] = (bs, n)
    t0 = time.perf_counter()
    threads = []
    for ci in range(args.n_chunks):
        t = threading.Thread(target=worker, args=(ci,))
        threads.append(t); t.start()
    for t in threads: t.join()
    torch.cuda.synchronize()
    dt_thr = time.perf_counter() - t0
    out_bytes_thr = sum(len(r[0]) for r in results)
    print(f"  in={in_bytes/1024:.0f}KB out={out_bytes_thr/1024:.0f}KB "
          f"ratio={out_bytes_thr/in_bytes:.4f} time={dt_thr:.2f}s "
          f"throughput={in_bytes/dt_thr/1024:.0f}KB/s")

    speedup = dt_seq / dt_thr
    print(f"\nSpeedup: {speedup:.2f}× ({args.n_chunks} chunks, {args.n_threads} threads)")
    print(f"Aggregate throughput: {in_bytes/dt_thr/1024:.0f} KB/s "
          f"({'PASSES' if in_bytes/dt_thr/1024 > 300 else 'misses'} 300 KB/s gate)")


if __name__ == "__main__":
    main()

"""
1 MB WildChat compress + decompress throughput bench with bit-exact roundtrip check.

Run on GPU instance:
  PYTHONPATH=/tmp/krunch python3 /tmp/krunch/scripts/bench_1mb_roundtrip.py \
    --sample /tmp/sample.bin [--mb 1] [--warmup 1]

Reports: compress KB/s, decompress KB/s, ratio, byte_exact.
Uses the production path (matches krunch CLI cli.py):
  - chunking.compute_chunk_size for dynamic chunk size
    (target ~4× batch_B chunks, floored at 64 KB)
  - per-chunk engine.compress_chunk (packed T=1024 forward; batched
    stepped compress is 5× slower per cli.py:53 comment)
  - engine.decompress_chunks_batched for batched decode
    (B auto from SM count via cpp_path.compute_decompress_batch)
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("KRUNCH_OWN_WKV", "1")
os.environ.setdefault("KRUNCH_CPP_PATH", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")
os.environ.setdefault("RWKV_JIT_ON", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True, help="Path to WildChat binary sample")
    ap.add_argument("--mb", type=float, default=1.0, help="MB to read from sample")
    # Warmup amortizes JIT + CUDA-graph capture + cuBLAS handle init etc.
    # Production CLI pays these once at engine.load() and forever after; the
    # bench needs an explicit warmup pass to match that steady state.
    # Don't pass --warmup 0 unless you specifically want cold-start numbers.
    ap.add_argument("--warmup", type=int, default=1, help="Warmup compress+decompress rounds (default 1 — match production steady state)")
    args = ap.parse_args()
    if args.warmup < 1:
        print("WARNING: --warmup 0 measures cold-start (JIT + cuBLAS init + "
              "graph capture) cost as part of throughput. Production never "
              "pays this. Use --warmup 1 for production-comparable numbers.",
              flush=True)

    n_bytes = int(args.mb * 1024 * 1024)
    with open(args.sample, "rb") as f:
        raw = f.read(n_bytes)
    if len(raw) < n_bytes:
        sys.exit(f"Sample too small: got {len(raw)} bytes, need {n_bytes}")
    print(f"Sample: {len(raw)} bytes ({len(raw)/1024:.0f} KB)", flush=True)

    from krunch.inference import engine
    from krunch.chunking import compute_chunk_size, _split_utf8_safe
    from krunch import cpp_path

    print("Loading model...", flush=True)
    engine.load()
    print("Model loaded.", flush=True)

    chunk_size = compute_chunk_size(len(raw))
    chunks = _split_utf8_safe(raw, chunk_size)
    print(f"Chunking: chunk_size={chunk_size} ({chunk_size//1024} KB), "
          f"n_chunks={len(chunks)}", flush=True)

    # Decompress batch B is auto-picked from SM count + n_chunks.
    B_dec = cpp_path.compute_decompress_batch(n_chunks=len(chunks))
    print(f"Decompress batch B={B_dec} (auto from SM count, clamped to n_chunks)",
          flush=True)

    def one_roundtrip(label: str):
        import torch
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # Per-chunk packed compress (matches production cli.py default; the
        # batched-stepped compress path is 5× slower).
        # D3 (Phase 0b): use compress_chunks so tokenization is batched
        # via tokenizer.encode_batch — same per-chunk bytes, faster.
        compressed_chunks = engine.compress_chunks(chunks)
        torch.cuda.synchronize()
        t_compress = time.perf_counter() - t0

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        # Cross-chunk batched decompress (B auto-picked from SM count).
        decompressed_chunks = engine.decompress_chunks_batched(compressed_chunks)
        torch.cuda.synchronize()
        t_decompress = time.perf_counter() - t1

        # Aggregate — strip mini-headers (8 B each) from compressed total to get
        # the raw AC bitstream size (matches the historical ratio definition).
        ac_bytes = sum(len(c) - 8 for c in compressed_chunks)
        ratio = ac_bytes / len(raw)
        compress_kbs = len(raw) / 1024 / t_compress
        decompress_kbs = len(raw) / 1024 / t_decompress
        decompressed = b"".join(decompressed_chunks)
        byte_exact = (decompressed == raw)

        print(
            f"[{label}] compress {compress_kbs:.1f} KB/s  "
            f"decompress {decompress_kbs:.1f} KB/s  "
            f"ratio {ratio:.4f}  "
            f"byte_exact={byte_exact}  "
            f"({t_compress:.2f}s / {t_decompress:.2f}s)",
            flush=True,
        )
        return compress_kbs, decompress_kbs, ratio, byte_exact

    for i in range(args.warmup):
        one_roundtrip(f"warmup-{i+1}")

    compress_kbs, decompress_kbs, ratio, byte_exact = one_roundtrip("measured")

    result = {
        "input_bytes": len(raw),
        "n_chunks": len(chunks),
        "chunk_size": chunk_size,
        "decompress_batch": B_dec,
        "ratio": round(ratio, 5),
        "compress_kb_s": round(compress_kbs, 1),
        "decompress_kb_s": round(decompress_kbs, 1),
        "byte_exact": byte_exact,
        "gates": {
            "ratio_lte_0_11": ratio <= 0.11,
            "compress_kb_s_gte_200": compress_kbs >= 200,
            "decompress_kb_s_gte_200": decompress_kbs >= 200,
            "byte_exact": byte_exact,
        },
    }
    result["all_gates_pass"] = all(result["gates"].values())

    print("\n=== RESULT ===", flush=True)
    print(json.dumps(result, indent=2), flush=True)

    out_path = os.environ.get("RESULT_PATH", "/tmp/bench_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Written to {out_path}", flush=True)


if __name__ == "__main__":
    main()

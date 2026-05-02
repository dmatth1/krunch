"""Decompress E2E throughput at varying cross-chunk batch B.

Forward microbench showed B=128→512 gives 2× per-token throughput on A10G.
This validates whether E2E decompress (forward + AC decode + Python loop)
also scales 2×, or whether other overhead caps it.

Run:
  PYTHONPATH=... python bench_decompress_at_b.py [--mb N]

Compresses N MB of synthetic text (looped paragraph) into 64KB chunks
once, then decompresses at KRUNCH_DECOMPRESS_BATCH ∈ {128, 256, 512},
reporting KB/s for each.
"""
import argparse, os, sys, time
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("KRUNCH_OWN_WKV", "1")
os.environ.setdefault("KRUNCH_CPP_PATH", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")
os.environ.setdefault("RWKV_JIT_ON", "1")


def make_text_blob(mb: int) -> bytes:
    base = (
        b"Krunch is a neural compression codec for text built around "
        b"a small RWKV-4 language model and a custom GPU arithmetic coder. "
        b"It targets large datasets where ratio matters more than latency. "
        b"The reference implementation runs on a single NVIDIA GPU and "
        b"parallelizes across machines using whatever batch system the "
        b"caller already runs. Compression is deterministic; the decoder "
        b"reproduces the encoder bitstream byte-for-byte across hardware. "
    )
    target = mb * 1024 * 1024
    out = bytearray()
    while len(out) < target:
        out.extend(base)
    return bytes(out[:target])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=8)
    ap.add_argument("--bs", type=str, default="128,256,512")
    args = ap.parse_args()
    target_bs = [int(b) for b in args.bs.split(",")]

    from krunch.inference import InferenceEngine
    eng = InferenceEngine()
    eng.load()

    # Chunk blob into 64KB pieces, compress each
    blob = make_text_blob(args.mb)
    CHUNK = 64 * 1024
    chunks = [blob[i:i+CHUNK] for i in range(0, len(blob), CHUNK)]
    print(f"Sample: {len(blob)/1e6:.1f} MB → {len(chunks)} × 64KB chunks")

    t0 = time.time()
    encoded = [eng.compress_chunk(c) for c in chunks]
    t_compress = time.time() - t0
    total_enc = sum(len(e) for e in encoded)
    print(f"Compress: {t_compress:.1f}s  ratio: {total_enc/len(blob):.4f}  "
          f"speed: {len(blob)/1024/t_compress:.1f} KB/s")

    # Decompress at varying B (env override is read fresh by pick_decompress_batch)
    print("\nDecompress throughput at varying cross-chunk batch B:")
    for B in target_bs:
        os.environ["KRUNCH_DECOMPRESS_BATCH"] = str(B)
        # Warmup once
        _ = eng.decompress_chunks_batched(encoded)
        # Timed
        t0 = time.time()
        out = eng.decompress_chunks_batched(encoded)
        t = time.time() - t0
        out_concat = b"".join(out)
        if out_concat[:len(blob)] != blob[:len(out_concat)]:
            print(f"  B={B}: FAIL — output mismatch")
            continue
        print(f"  B={B}: {t:.2f}s   {len(blob)/1024/t:.1f} KB/s")


if __name__ == "__main__":
    main()

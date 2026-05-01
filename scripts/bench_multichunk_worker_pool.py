"""Realistic multi-chunk throughput via DecompressWorkerPool.

Real workloads aren't single-chunk. A 1 MB file becomes 16 × 64 KB
chunks (or N × 32 KB). The worker pool decompresses chunks
concurrently across N processes (each with its own CUDA context).
This benchmark reports the aggregate KB/s a multi-chunk file
would see.
"""
import os, sys, time
os.environ.setdefault("RWKV_CUDA_ON", "1")
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("KRUNCH_CPP_PATH", "1")

from krunch.inference import InferenceEngine
from krunch.worker_pool import DecompressWorkerPool


def make_chunk(n_bytes: int) -> bytes:
    base = (
        b"Krunch is a neural compression codec for text built around "
        b"a small RWKV-4 language model and a custom GPU arithmetic coder. "
        b"It targets large datasets where ratio matters more than latency. "
    )
    out = bytearray()
    while len(out) < n_bytes:
        out.extend(base)
    return bytes(out[:n_bytes])


def main():
    chunk_kb = int(os.environ.get("CHUNK_KB", 32))
    n_chunks = int(os.environ.get("N_CHUNKS", 8))
    n_workers = int(os.environ.get("N_WORKERS", 4))
    chunk_size = chunk_kb * 1024

    # Build encoded chunks via the parent engine
    print(f"prepping {n_chunks} × {chunk_kb} KB chunks via cpp_path encode...",
          flush=True)
    eng = InferenceEngine()
    eng.load()
    chunks = [make_chunk(chunk_size) for _ in range(n_chunks)]
    encoded = []
    t0 = time.time()
    for ch in chunks:
        encoded.append(eng.compress_chunk(ch))
    t_enc = time.time() - t0
    total_in = sum(len(c) for c in chunks)
    total_enc = sum(len(e) for e in encoded)
    print(f"encoded {total_in} B → {total_enc} B "
          f"(ratio={total_enc/total_in:.3f}) in {t_enc:.2f}s "
          f"= {total_in/1024/t_enc:.1f} KB/s",
          flush=True)

    # Now spawn worker pool and time aggregate decompress
    print(f"spawning {n_workers} workers...", flush=True)
    with DecompressWorkerPool(n_workers) as pool:
        # Warm: do a tiny chunk through each worker to JIT-compile
        warm = eng.compress_chunk(b"warmup data " * 4)
        _ = pool.decompress_chunks([warm] * n_workers)

        t0 = time.time()
        decoded = pool.decompress_chunks(encoded)
        t_dec = time.time() - t0

    ok = all(d == c for d, c in zip(decoded, chunks))
    print(f"decoded {total_in} B in {t_dec:.2f}s "
          f"= {total_in/1024/t_dec:.1f} KB/s "
          f"({'PASS' if ok else 'FAIL'} bit-exact)",
          flush=True)
    print(f"asymmetry dec/enc: {t_dec/t_enc:.2f}×", flush=True)


if __name__ == "__main__":
    main()

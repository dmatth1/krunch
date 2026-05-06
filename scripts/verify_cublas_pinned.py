"""
Verify whether KRUNCH_CUBLAS_PINNED=1 is bit-exact and whether it beats
the default cp.async path on A10G compress.

Three configs tested — all must produce identical compressed bytes:
  default:     KRUNCH_HEAD_ASYNC=1 (cp.async + WMMA) — current production
  cublas:      KRUNCH_HEAD_ASYNC=0 + KRUNCH_CUBLAS_PINNED=1 — test
  slow:        KRUNCH_HEAD_ASYNC=0 — det_matmul_tc fallback (T4-like)

Run in fresh process per config (the C++ static const reads env once at
load). Compress same chunks each time, compare bytes, measure speed.

Run on GPU instance via Docker:
  docker run --rm --gpus all \\
    -e KRUNCH_CPP_PATH=1 -e KRUNCH_DETERMINISTIC_MATMUL=1 -e KRUNCH_OWN_WKV=1 \\
    -e KRUNCH_HEAD_ASYNC=$ASYNC -e KRUNCH_CUBLAS_PINNED=$PINNED \\
    -v /tmp:/tmp -v /tmp/inference.py:/app/krunch/inference.py:ro \\
    -v /tmp/cpp_path.py:/app/krunch/cpp_path.py:ro \\
    --entrypoint /opt/conda/bin/python \\
    ghcr.io/dmatth1/krunch:latest \\
    /tmp/verify_cublas_pinned.py --sample /tmp/sample.bin
"""
import argparse
import hashlib
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
    ap.add_argument("--sample", required=True)
    ap.add_argument("--mb", type=float, default=1.0)
    args = ap.parse_args()

    n_bytes = int(args.mb * 1024 * 1024)
    with open(args.sample, "rb") as f:
        raw = f.read(n_bytes)

    label = (
        f"async={os.environ.get('KRUNCH_HEAD_ASYNC','1')} "
        f"3way={os.environ.get('KRUNCH_3WAY_ASYNC','1')} "
        f"cublas_pin={os.environ.get('KRUNCH_CUBLAS_PINNED','0')}"
    )
    print(f"Config: {label}", flush=True)
    print(f"Sample: {len(raw)} bytes", flush=True)

    import torch
    from krunch.inference import engine
    from krunch.chunking import compute_chunk_size, _split_utf8_safe

    print("Loading model...", flush=True)
    engine.load()

    chunks = _split_utf8_safe(raw, compute_chunk_size(len(raw)))
    print(f"n_chunks={len(chunks)}", flush=True)

    # Warmup
    _ = engine.compress_chunk(chunks[0])

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    compressed = [engine.compress_chunk(c) for c in chunks]
    torch.cuda.synchronize()
    t = time.perf_counter() - t0
    rate = len(raw) / 1024 / t
    blob = b"".join(compressed)
    digest = hashlib.sha256(blob).hexdigest()

    # Round-trip check
    decoded = engine.decompress_chunks_batched(compressed)
    out = b"".join(decoded)
    byte_exact = (out == raw)

    print(f"\n=== RESULT ===")
    print(f"  compress: {t:.2f}s ({rate:.1f} KB/s)")
    print(f"  blob_sha256: {digest}")
    print(f"  blob_size: {len(blob)} bytes")
    print(f"  ratio: {len(blob)/len(raw):.4f}")
    print(f"  byte_exact_roundtrip: {byte_exact}")

    result = {
        "config": label,
        "compress_kb_s": round(rate, 1),
        "blob_sha256": digest,
        "blob_size": len(blob),
        "ratio": round(len(blob)/len(raw), 5),
        "byte_exact_roundtrip": byte_exact,
    }
    out_path = os.environ.get("RESULT_PATH", "/tmp/cublas_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Written to {out_path}", flush=True)


if __name__ == "__main__":
    main()

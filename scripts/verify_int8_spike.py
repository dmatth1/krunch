"""
int8 weight quantization SPIKE — quality test.

Question: does per-input-channel uint8 quantization of all matmul +
emb + head weights degrade ratio enough to disqualify int8 for v1.x?

Bellard's ts_zip uses 8-bit weights with the same RWKV-4-Pile-169M
model and reports the same compression ratio. If ours holds, int8
is the next sprint's lever (real kernel: int8 weight × fp16 act →
fp16 out with inline dequant; halves memory bandwidth on
decompress's bandwidth-bound regime).

Test runs in TWO fresh processes:
  fp16 process: KRUNCH_INT8_WEIGHTS=0 (default) — baseline
  int8 process: KRUNCH_INT8_WEIGHTS=1            — spike

Each compresses + decompresses the same 1 MB sample, prints:
  - compress KB/s, decompress KB/s
  - blob sha256 (will differ between fp16 and int8 — different bitstreams)
  - ratio
  - byte_exact roundtrip (within each process — must be True for both)

Caller compares the two RESULT_PATH JSON files to assess the
ratio delta.

Run on GPU instance via Docker:
  docker run --rm --gpus all \\
    -e KRUNCH_CPP_PATH=1 -e KRUNCH_DETERMINISTIC_MATMUL=1 -e KRUNCH_OWN_WKV=1 \\
    -e KRUNCH_INT8_WEIGHTS=$INT8 \\
    -e RESULT_PATH=/tmp/int8_result_$INT8.json \\
    -v /tmp:/tmp -v /tmp/inference.py:/app/krunch/inference.py:ro \\
    -v /tmp/cpp_path.py:/app/krunch/cpp_path.py:ro \\
    --entrypoint /opt/conda/bin/python \\
    ghcr.io/dmatth1/krunch:latest \\
    /tmp/verify_int8_spike.py --sample /tmp/sample.bin --mb 1
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

    int8_on = os.environ.get("KRUNCH_INT8_WEIGHTS", "0") == "1"
    label = "int8" if int8_on else "fp16"
    print(f"=== Config: {label} (KRUNCH_INT8_WEIGHTS={int8_on}) ===", flush=True)
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

    # Compress
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    compressed = [engine.compress_chunk(c) for c in chunks]
    torch.cuda.synchronize()
    t_c = time.perf_counter() - t0
    blob = b"".join(compressed)

    # Decompress
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    decoded = engine.decompress_chunks_batched(compressed)
    torch.cuda.synchronize()
    t_d = time.perf_counter() - t1
    out = b"".join(decoded)
    byte_exact = (out == raw)

    digest = hashlib.sha256(blob).hexdigest()
    rate_c = len(raw) / 1024 / t_c
    rate_d = len(raw) / 1024 / t_d
    ratio = len(blob) / len(raw)

    print(f"\n=== RESULT ({label}) ===")
    print(f"  compress:    {t_c:6.2f}s  ({rate_c:.1f} KB/s)")
    print(f"  decompress:  {t_d:6.2f}s  ({rate_d:.1f} KB/s)")
    print(f"  blob_size:   {len(blob)} bytes")
    print(f"  ratio:       {ratio:.5f}")
    print(f"  blob_sha256: {digest}")
    print(f"  byte_exact_roundtrip: {byte_exact}")

    result = {
        "config": label,
        "int8_weights": int8_on,
        "compress_kb_s": round(rate_c, 1),
        "decompress_kb_s": round(rate_d, 1),
        "blob_size": len(blob),
        "blob_sha256": digest,
        "ratio": round(ratio, 5),
        "byte_exact_roundtrip": byte_exact,
    }
    out_path = os.environ.get("RESULT_PATH", f"/tmp/int8_result_{label}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Written to {out_path}", flush=True)

    if not byte_exact:
        sys.exit(f"FAIL: roundtrip not byte-exact in {label} process")


if __name__ == "__main__":
    main()

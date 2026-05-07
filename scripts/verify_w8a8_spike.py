"""
W8A8 weight quantization quality SPIKE.

Validates that per-output-channel SYMMETRIC int8 quantization
preserves compression ratio on WildChat — the gating question
before investing in the int8 WMMA kernel (Phase 2A).

Per-output-channel because the W8A8 int8-WMMA kernel needs the
weight scale to be applied AFTER WMMA accumulation along K, so
the scale must be constant across K (i.e., per-N output channel).
This is different from the Phase 1 spike which used per-INPUT-channel
asymmetric uint8 (worked for W8A16 weight-only dequant but doesn't
fit the int8 WMMA shape).

Tests three configs in fresh processes (statics read env once):
  fp16        : KRUNCH_INT8_W8A8=0 KRUNCH_INT8_WEIGHTS=0 — baseline
  int8_w8a16  : KRUNCH_INT8_W8A8=0 KRUNCH_INT8_WEIGHTS=1 — Phase 1 spike
                (per-input-channel uint8 + offset, asymmetric)
  int8_w8a8   : KRUNCH_INT8_W8A8=1 KRUNCH_INT8_WEIGHTS=0 — Phase 2A spike
                (per-output-channel int8, symmetric)

Each compresses the same 1 MB sample, prints compress KB/s, ratio,
blob sha256, byte_exact roundtrip. Compare ratio across the three.

If int8_w8a8 ratio is within ~0.3 % of fp16 baseline → green light
to build the int8 WMMA kernel. If it tanks (e.g., > 1 %) → revisit
quant scheme (per-N may be too coarse on certain layers; try mixed
per-tensor + per-N).

Run on GPU instance via Docker (build extension first if not done):
  docker run --rm --gpus all \\
    -e KRUNCH_CPP_PATH=1 -e KRUNCH_DETERMINISTIC_MATMUL=1 -e KRUNCH_OWN_WKV=1 \\
    -e KRUNCH_INT8_W8A8=$W8A8 -e KRUNCH_INT8_WEIGHTS=$W8A16 \\
    -e RESULT_PATH=/tmp/w8a8_$LABEL.json \\
    -v /tmp:/tmp -v /tmp/inference.py:/app/krunch/inference.py:ro \\
    -v /tmp/cpp_path.py:/app/krunch/cpp_path.py:ro \\
    --entrypoint /opt/conda/bin/python \\
    ghcr.io/dmatth1/krunch:latest \\
    /tmp/verify_w8a8_spike.py --sample /tmp/sample.bin --mb 1
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

    w8a16 = os.environ.get("KRUNCH_INT8_WEIGHTS", "0") == "1"
    w8a8 = os.environ.get("KRUNCH_INT8_W8A8", "0") == "1"
    if w8a16:
        label = "int8_w8a16"
    elif w8a8:
        label = "int8_w8a8"
    else:
        label = "fp16"
    print(f"=== Config: {label} ===", flush=True)
    print(f"Sample: {len(raw)} bytes", flush=True)

    import torch
    from krunch.inference import engine
    from krunch.chunking import compute_chunk_size, _split_utf8_safe

    print("Loading model...", flush=True)
    engine.load()

    chunks = _split_utf8_safe(raw, compute_chunk_size(len(raw)))
    print(f"n_chunks={len(chunks)}", flush=True)

    # Warmup using the new compress_chunks API (Phase 0 D3)
    _ = engine.compress_chunks(chunks[:1])

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    compressed = engine.compress_chunks(chunks)
    torch.cuda.synchronize()
    t_c = time.perf_counter() - t0
    blob = b"".join(compressed)

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
        "compress_kb_s": round(rate_c, 1),
        "decompress_kb_s": round(rate_d, 1),
        "blob_size": len(blob),
        "blob_sha256": digest,
        "ratio": round(ratio, 5),
        "byte_exact_roundtrip": byte_exact,
    }
    out_path = os.environ.get("RESULT_PATH", f"/tmp/w8a8_{label}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Written to {out_path}", flush=True)

    if not byte_exact:
        sys.exit(f"FAIL: roundtrip not byte-exact in {label} process")


if __name__ == "__main__":
    main()

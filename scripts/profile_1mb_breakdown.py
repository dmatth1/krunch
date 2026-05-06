"""
Time-breakdown profile of compress + decompress on 1 MB WildChat.

Enables:
  - KRUNCH_CPP_PROFILE=1 for chunk-level breakdown
    (weights / state_init / forward / cdf / ac / copy per compress_chunk)
  - KRUNCH_PHASE_PROFILE=1 for layer-step phase breakdown
    (LN1 / premix3 / KVR / sigmoid / WKV / Ow+residual / LN2 /
     premix2 / FFN_R / FFN_K / FFN_V per rwkv4_layer_step_cpp call)

Output is total-wall + accumulated phase times across all layer-step calls.

Run on the GPU instance via Docker:
  docker run --rm --gpus all \\
    -e KRUNCH_CPP_PATH=1 -e KRUNCH_DETERMINISTIC_MATMUL=1 -e KRUNCH_OWN_WKV=1 \\
    -e KRUNCH_HEAD_ASYNC=0 -e KRUNCH_3WAY_ASYNC=0 \\
    -e KRUNCH_CPP_PROFILE=1 -e KRUNCH_PHASE_PROFILE=1 \\
    -v /tmp:/tmp --entrypoint /opt/conda/bin/python \\
    ghcr.io/dmatth1/krunch:latest \\
    /tmp/profile_1mb_breakdown.py --sample /tmp/sample.bin
"""
import argparse
import logging
import os
import sys
import time

os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("KRUNCH_OWN_WKV", "1")
os.environ.setdefault("KRUNCH_CPP_PATH", "1")
os.environ.setdefault("KRUNCH_CPP_PROFILE", "1")
os.environ.setdefault("KRUNCH_PHASE_PROFILE", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")
os.environ.setdefault("RWKV_JIT_ON", "1")

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--mb", type=float, default=1.0)
    args = ap.parse_args()

    n_bytes = int(args.mb * 1024 * 1024)
    with open(args.sample, "rb") as f:
        raw = f.read(n_bytes)
    print(f"\n=== INPUT: {len(raw)} bytes ===\n", flush=True)

    import torch  # MUST come before krunch_ac_cuda (sets up libc10.so)
    from krunch.inference import engine
    from krunch.chunking import compute_chunk_size, _split_utf8_safe
    from krunch import cpp_path
    import krunch_ac_cuda

    print("Loading model...", flush=True)
    engine.load()

    chunk_size = compute_chunk_size(len(raw))
    chunks = _split_utf8_safe(raw, chunk_size)
    print(f"chunk_size={chunk_size//1024} KB, n_chunks={len(chunks)}\n",
          flush=True)

    # Warmup (gets weights cache + JIT compiles primed). Don't profile this.
    krunch_ac_cuda.reset_phase_times()
    print("=== WARMUP (1 chunk, not profiled) ===", flush=True)
    _ = engine.compress_chunk(chunks[0])
    print("warmup done\n", flush=True)

    # ---- COMPRESS ----
    krunch_ac_cuda.reset_phase_times()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    compressed = []
    for c in chunks:
        compressed.append(engine.compress_chunk(c))
    torch.cuda.synchronize()
    t_compress = time.perf_counter() - t0
    compress_phases = dict(krunch_ac_cuda.read_phase_times())

    print(f"\n=== COMPRESS WALL: {t_compress*1000:.1f} ms "
          f"({len(raw)/1024/t_compress:.1f} KB/s) ===")
    print("\nLayer-step phase breakdown (cumulative across all layer calls):")
    cc = compress_phases.pop("call_count", 0)
    print(f"  total layer-step calls: {cc}")
    total_phase_ms = sum(compress_phases.values())
    for name, ms in sorted(compress_phases.items(), key=lambda x: -x[1]):
        pct = 100 * ms / max(total_phase_ms, 1e-9)
        print(f"    {name:25s} {ms:8.1f} ms  ({pct:5.1f}% of layer-step time)")
    print(f"  layer-step total:        {total_phase_ms:8.1f} ms  "
          f"({100*total_phase_ms/(t_compress*1000):.1f}% of wall)")
    print(f"  outside layer-step:      "
          f"{t_compress*1000 - total_phase_ms:8.1f} ms  "
          f"(softmax/cdf/encode/.cpu()/empty_cache/tokenize/per-chunk overhead)")

    # ---- DECOMPRESS ----
    krunch_ac_cuda.reset_phase_times()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    decoded = engine.decompress_chunks_batched(compressed)
    torch.cuda.synchronize()
    t_decompress = time.perf_counter() - t1
    decompress_phases = dict(krunch_ac_cuda.read_phase_times())

    print(f"\n=== DECOMPRESS WALL: {t_decompress*1000:.1f} ms "
          f"({len(raw)/1024/t_decompress:.1f} KB/s) ===")
    print("\nLayer-step phase breakdown (cumulative across all layer calls):")
    cc2 = decompress_phases.pop("call_count", 0)
    print(f"  total layer-step calls: {cc2}")
    total_phase_ms2 = sum(decompress_phases.values())
    for name, ms in sorted(decompress_phases.items(), key=lambda x: -x[1]):
        pct = 100 * ms / max(total_phase_ms2, 1e-9)
        print(f"    {name:25s} {ms:8.1f} ms  ({pct:5.1f}% of layer-step time)")
    print(f"  layer-step total:        {total_phase_ms2:8.1f} ms  "
          f"({100*total_phase_ms2/(t_decompress*1000):.1f}% of wall)")
    print(f"  outside layer-step:      "
          f"{t_decompress*1000 - total_phase_ms2:8.1f} ms  "
          f"(softmax/cdf/decode/.item()/Python loop/launch overhead)")

    # Roundtrip sanity
    decompressed = b"".join(decoded)
    assert decompressed == raw, "roundtrip MISMATCH"
    print("\nbyte_exact: True")


if __name__ == "__main__":
    main()

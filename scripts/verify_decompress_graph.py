"""
Verify the three decompress modes are bit-exact + benchmark them.

Modes (KRUNCH_DECOMPRESS_GRAPH):
  "eager"     — no graph capture
  "per_layer" — 12 graphs (one per layer); softmax+CDF+decode outside graph
                [legacy, ~1.03× on T4/A10G B=16]
  "full"      — emb → 12 layers → ln_out → head → softmax+CDF → AC decode
                in ONE captured graph; 1 replay per step
                [Week 1 sprint lever, target 3-5× decompress on A10G B=128+]

Bit-exactness is the gate: all three modes MUST produce identical bytes
or AC roundtrip is broken (encoder-decoder symmetry violated).

Run on GPU instance via Docker:
  docker run --rm --gpus all \\
    -e KRUNCH_CPP_PATH=1 -e KRUNCH_DETERMINISTIC_MATMUL=1 -e KRUNCH_OWN_WKV=1 \\
    -v /tmp:/tmp -v /tmp/inference.py:/app/krunch/inference.py:ro \\
    -v /tmp/cpp_path.py:/app/krunch/cpp_path.py:ro \\
    --entrypoint /opt/conda/bin/python \\
    ghcr.io/dmatth1/krunch:latest \\
    /tmp/verify_decompress_graph.py --sample /tmp/sample.bin [--mb 1]
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
    ap.add_argument("--sample", required=True)
    ap.add_argument("--mb", type=float, default=1.0)
    args = ap.parse_args()

    n_bytes = int(args.mb * 1024 * 1024)
    with open(args.sample, "rb") as f:
        raw = f.read(n_bytes)
    print(f"Sample: {len(raw)} bytes ({len(raw)/1024:.0f} KB)", flush=True)

    import torch  # before krunch_ac_cuda
    from krunch.inference import engine
    from krunch.chunking import compute_chunk_size, _split_utf8_safe

    print("Loading model...", flush=True)
    engine.load()

    chunk_size = compute_chunk_size(len(raw))
    chunks = _split_utf8_safe(raw, chunk_size)
    print(f"chunk_size={chunk_size//1024} KB, n_chunks={len(chunks)}", flush=True)

    print("\nCompressing chunks (production per-chunk packed path)...",
          flush=True)
    t0 = time.perf_counter()
    # D3: batch-tokenize via compress_chunks (Phase 0b)
    compressed = engine.compress_chunks(chunks)
    t_compress = time.perf_counter() - t0
    print(f"  compress: {t_compress:.2f}s ({len(raw)/1024/t_compress:.1f} KB/s)",
          flush=True)

    def run(mode: str, label: str):
        os.environ["KRUNCH_DECOMPRESS_GRAPH"] = mode
        torch.cuda.synchronize()
        t = time.perf_counter()
        decoded = engine.decompress_chunks_batched(compressed)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t
        rate = len(raw) / 1024 / elapsed
        out = b"".join(decoded)
        ok = (out == raw)
        print(f"  [{label:14s}] {elapsed:7.2f}s  ({rate:6.1f} KB/s)  "
              f"byte_exact={ok}", flush=True)
        return out, elapsed, ok

    print("\n=== STEP 1: bit-exactness across modes ===", flush=True)
    print("Eager (warmup):", flush=True)
    eager_w, _, eager_w_ok = run("eager", "eager-warmup")
    print("Eager (measured):", flush=True)
    eager, t_eager, eager_ok = run("eager", "eager")

    print("Per-layer graph (warmup, captures 12 graphs):", flush=True)
    per_layer_w, _, pl_w_ok = run("per_layer", "per_layer-w")
    print("Per-layer graph (measured):", flush=True)
    per_layer, t_per_layer, pl_ok = run("per_layer", "per_layer")

    print("Full-step graph (warmup, captures 1 graph):", flush=True)
    full_w, _, full_w_ok = run("full", "full-warmup")
    print("Full-step graph (measured):", flush=True)
    full, t_full, full_ok = run("full", "full")

    # All four runs must produce raw exactly:
    if not (eager_w_ok and eager_ok and pl_w_ok and pl_ok and full_w_ok and full_ok):
        sys.exit("FAIL: at least one decompress wasn't byte-exact vs source")

    # Determinism: warmup == measured for every mode
    if eager_w != eager:
        sys.exit("FAIL: eager not deterministic across runs")
    if per_layer_w != per_layer:
        sys.exit("FAIL: per_layer not deterministic across runs")
    if full_w != full:
        sys.exit("FAIL: full-step not deterministic across runs")

    # Cross-mode bit-exactness — the load-bearing check
    if eager != per_layer:
        sys.exit("FAIL: eager != per_layer (legacy graph capture diverged)")
    if eager != full:
        sys.exit("FAIL: eager != full (whole-step graph capture diverged — "
                 "AC roundtrip would break)")

    print("\n  eager == per_layer: True")
    print("  eager == full:      True (whole-step capture is bit-exact)")

    print("\n=== STEP 2: throughput comparison ===")
    rate_e = len(raw) / 1024 / t_eager
    rate_p = len(raw) / 1024 / t_per_layer
    rate_f = len(raw) / 1024 / t_full
    print(f"  eager     : {t_eager:7.2f}s  ({rate_e:6.1f} KB/s)  1.00× (baseline)")
    print(f"  per_layer : {t_per_layer:7.2f}s  ({rate_p:6.1f} KB/s)  "
          f"{rate_p/rate_e:.2f}×")
    print(f"  full-step : {t_full:7.2f}s  ({rate_f:6.1f} KB/s)  "
          f"{rate_f/rate_e:.2f}×  ← Week 1 lever")

    result = {
        "input_bytes": len(raw),
        "n_chunks": len(chunks),
        "compress_kb_s": round(len(raw)/1024/t_compress, 1),
        "eager_decompress_kb_s": round(rate_e, 1),
        "per_layer_decompress_kb_s": round(rate_p, 1),
        "full_step_decompress_kb_s": round(rate_f, 1),
        "per_layer_speedup": round(rate_p / rate_e, 2),
        "full_step_speedup": round(rate_f / rate_e, 2),
        "byte_exact_all": True,
        "all_modes_match": True,
    }
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    out_path = os.environ.get("RESULT_PATH", "/tmp/verify_graph_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Written to {out_path}", flush=True)


if __name__ == "__main__":
    main()

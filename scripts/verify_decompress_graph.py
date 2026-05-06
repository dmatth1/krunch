"""
Verify CUDA-graph-captured decompress is bit-exact vs the eager path,
then benchmark both on 1 MB WildChat.

Run on GPU instance via Docker (T4 currently up, A10G works the same):
  docker run --rm --gpus all \\
    -e KRUNCH_CPP_PATH=1 -e KRUNCH_DETERMINISTIC_MATMUL=1 -e KRUNCH_OWN_WKV=1 \\
    -e KRUNCH_HEAD_ASYNC=0 -e KRUNCH_3WAY_ASYNC=0 \\
    -v /tmp:/tmp --entrypoint /opt/conda/bin/python \\
    ghcr.io/dmatth1/krunch:latest \\
    /tmp/verify_decompress_graph.py --sample /tmp/sample.bin
"""
import argparse
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
    print(f"Sample: {len(raw)} bytes", flush=True)

    import torch  # before krunch_ac_cuda
    from krunch.inference import engine
    from krunch.chunking import compute_chunk_size, _split_utf8_safe

    print("Loading model...", flush=True)
    engine.load()

    chunk_size = compute_chunk_size(len(raw))
    chunks = _split_utf8_safe(raw, chunk_size)
    print(f"chunk_size={chunk_size//1024} KB, n_chunks={len(chunks)}", flush=True)

    # Compress once — this is the input both paths decompress.
    print("\nCompressing chunks (production per-chunk packed path)...",
          flush=True)
    t0 = time.perf_counter()
    compressed = [engine.compress_chunk(c) for c in chunks]
    t_compress = time.perf_counter() - t0
    print(f"  compress: {t_compress:.2f}s ({len(raw)/1024/t_compress:.1f} KB/s)",
          flush=True)

    def run_decompress(use_graph: bool, label: str):
        # Per-call env override for the decompress dispatch.
        os.environ["KRUNCH_DECOMPRESS_GRAPH"] = "1" if use_graph else "0"
        # The graph cache is keyed by (id(weights), B); identical across runs
        # in this process, so first call captures, subsequent replay.
        torch.cuda.synchronize()
        t = time.perf_counter()
        decoded = engine.decompress_chunks_batched(compressed)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t
        rate = len(raw) / 1024 / elapsed
        out = b"".join(decoded)
        ok = (out == raw)
        print(f"  [{label:8s}] {elapsed:7.2f}s  ({rate:6.1f} KB/s)  "
              f"byte_exact={ok}", flush=True)
        return out, elapsed, ok

    # ---- 1. Bit-exactness: run eager twice, then graph twice; compare. ----
    print("\n=== STEP 1: bit-exactness ===", flush=True)
    print("Eager path (no graph), warmup:", flush=True)
    eager1, _, eager1_ok = run_decompress(use_graph=False, label="eager-1")
    print("Eager path (no graph), measured:", flush=True)
    eager2, t_eager, eager2_ok = run_decompress(use_graph=False, label="eager-2")
    print("Graph path, warmup (captures graphs):", flush=True)
    graph1, _, graph1_ok = run_decompress(use_graph=True, label="graph-1")
    print("Graph path, measured (replays):", flush=True)
    graph2, t_graph, graph2_ok = run_decompress(use_graph=True, label="graph-2")

    if not (eager1_ok and eager2_ok and graph1_ok and graph2_ok):
        sys.exit("FAIL: at least one decompress wasn't byte-exact vs source")

    if eager1 != eager2:
        sys.exit("FAIL: eager path is not deterministic across runs")
    if graph1 != graph2:
        sys.exit("FAIL: graph path is not deterministic across runs")
    if eager2 != graph2:
        sys.exit("FAIL: eager and graph paths produce DIFFERENT bytes "
                 "(graph capture is not bit-exact)")

    print("\n  eager == eager: True (deterministic)")
    print("  graph == graph: True (deterministic)")
    print("  eager == graph: True (graph capture is bit-exact)")

    # ---- 2. Throughput summary. ----
    print("\n=== STEP 2: throughput ===")
    speedup = t_eager / t_graph
    print(f"  eager : {t_eager:.2f}s  ({len(raw)/1024/t_eager:.1f} KB/s)")
    print(f"  graph : {t_graph:.2f}s  ({len(raw)/1024/t_graph:.1f} KB/s)")
    print(f"  speedup: {speedup:.2f}×")

    # ---- 3. Result JSON. ----
    import json
    result = {
        "input_bytes": len(raw),
        "n_chunks": len(chunks),
        "compress_kb_s": round(len(raw)/1024/t_compress, 1),
        "eager_decompress_kb_s": round(len(raw)/1024/t_eager, 1),
        "graph_decompress_kb_s": round(len(raw)/1024/t_graph, 1),
        "graph_speedup": round(speedup, 2),
        "byte_exact_eager": True,
        "byte_exact_graph": True,
        "eager_eq_graph_bytes": True,
    }
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    out_path = os.environ.get("RESULT_PATH", "/tmp/verify_graph_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Written to {out_path}", flush=True)


if __name__ == "__main__":
    main()

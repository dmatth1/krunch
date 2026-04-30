"""Realistic chunk throughput on the C++ path.

Generates a ~64 KB English text chunk (project README × N) and times
compress + decompress on the C++ path. Reports KB/s on each side and
the asymmetry ratio.

Compares with KRUNCH_CPP_PATH=0 (BlinkDL forward) for sanity.
"""
import os, sys, time
os.environ.setdefault("RWKV_CUDA_ON", "1")
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")

# Force per-call cpp_path.cpp_path_enabled() check to read live env.
from krunch.inference import InferenceEngine


def make_chunk(n_bytes: int) -> bytes:
    base = (
        b"Krunch is a neural compression codec for text built around "
        b"a small RWKV-4 language model and a custom GPU arithmetic coder. "
        b"It targets large datasets where ratio matters more than latency. "
        b"The reference implementation runs on a single NVIDIA GPU and "
        b"parallelizes across machines using whatever batch system the "
        b"caller already runs. Compression is deterministic; the decoder "
        b"reproduces the encoder bitstream byte-for-byte across hardware. "
    )
    out = bytearray()
    while len(out) < n_bytes:
        out.extend(base)
    return bytes(out[:n_bytes])


def time_path(eng, data, label):
    t0 = time.time()
    encoded = eng.compress_chunk(data)
    t_enc = time.time() - t0
    t0 = time.time()
    decoded = eng.decompress_chunk(encoded)
    t_dec = time.time() - t0
    ok = decoded == data
    n_in = len(data) / 1024
    enc_kbs = n_in / t_enc
    dec_kbs = n_in / t_dec
    print(f"[{label}] in={len(data):>6} B  out={len(encoded):>5} B  "
          f"ratio={len(encoded)/len(data):.3f}  "
          f"enc={t_enc:.2f}s ({enc_kbs:.1f} KB/s)  "
          f"dec={t_dec:.2f}s ({dec_kbs:.1f} KB/s)  "
          f"ratio_dec/enc={t_dec/t_enc:.2f}×  "
          f"{'PASS' if ok else 'FAIL'}",
          flush=True)
    return ok


def main():
    sizes_kb = [int(s) for s in os.environ.get("BENCH_SIZES_KB", "8,32,64").split(",")]
    eng = InferenceEngine()
    eng.load()

    for kb in sizes_kb:
        data = make_chunk(kb * 1024)
        # Warmup once per size (lazy compiles, weight caching, allocator)
        os.environ["KRUNCH_CPP_PATH"] = "1"
        time_path(eng, data[:512], f"warmup-cpp")

        # C++ path
        os.environ["KRUNCH_CPP_PATH"] = "1"
        time_path(eng, data, f"cpp  size={kb}KB")

        # BlinkDL path (sanity / baseline) — note this won't roundtrip
        # bit-exact in general, so just measure speed; FAIL is expected.
        os.environ["KRUNCH_CPP_PATH"] = "0"
        time_path(eng, data, f"stck size={kb}KB")


if __name__ == "__main__":
    main()

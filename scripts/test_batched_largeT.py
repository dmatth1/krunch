"""Bit-exact unit test for batched decompress at large T per chunk.

Earlier `test_batched_decompress.py` PASSED 8/8 chunks at 8 KB each
(~1500 tokens per chunk). Production T3 hits CRC mismatch with
64 KB chunks (~13K tokens per chunk). Isolates whether the batched
decompress correctly handles long autoregressive streams.

Compresses N independent 64KB chunks of WildChat-like text, then:
  A) decompresses each chunk individually via _decompress_chunk_cpp
  B) decompresses all N together via _decompress_chunks_batched_cpp

If (A) matches input bytes and (B) doesn't → bug in batched
decompress glue (probably input_buf/offsets at large bitstream).

If (A) also fails → bug is upstream (streaming compress wrote a
bitstream the per-chunk decode also can't reverse).
"""
import os
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("KRUNCH_CPP_PATH", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import time
from krunch.inference import InferenceEngine


def make_chunks(n_chunks: int, chunk_kb: int = 64):
    """Synthetic: repeated base text with tiny per-chunk XOR variation."""
    base = (
        b"Krunch is a neural compression codec for text built around "
        b"a small RWKV-4 language model and a custom GPU arithmetic coder. "
        b"It targets large datasets where ratio matters more than latency. "
        b"The reference implementation runs on a single NVIDIA GPU and "
        b"parallelizes across machines using whatever batch system the "
        b"caller already runs. Compression is deterministic; the decoder "
        b"reproduces the encoder bitstream byte-for-byte across hardware. "
    )
    sz = chunk_kb * 1024
    out = bytearray()
    while len(out) < sz:
        out.extend(base)
    one = bytes(out[:sz])
    return [bytes([(b ^ (i & 0x0F)) & 0xFF for b in one]) for i in range(n_chunks)]


def make_wildchat_chunks(n_chunks: int, chunk_kb: int = 64,
                          src: str = "/tmp/sample.bin"):
    """Real WildChat slices — the content that triggers the production bug."""
    sz = chunk_kb * 1024
    with open(src, "rb") as f:
        raw = f.read()
    if len(raw) < n_chunks * sz:
        raise RuntimeError(f"sample {src} too small for N={n_chunks} × {chunk_kb}KB")
    return [raw[i * sz : (i + 1) * sz] for i in range(n_chunks)]


def main():
    n_chunks = int(os.environ.get("N_CHUNKS", 4))
    chunk_kb = int(os.environ.get("CHUNK_KB", 64))
    use_wildchat = os.environ.get("USE_WILDCHAT") == "1"

    eng = InferenceEngine()
    eng.load()

    if use_wildchat:
        chunks = make_wildchat_chunks(n_chunks, chunk_kb)
        print(f"using WildChat content from /tmp/sample.bin")
    else:
        chunks = make_chunks(n_chunks, chunk_kb)
        print(f"using synthetic content")
    print(f"compressing N={n_chunks} chunks of {chunk_kb} KB each...", flush=True)
    t0 = time.time()
    encoded = [eng.compress_chunk(c) for c in chunks]
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    # ============== A: per-chunk via _decompress_chunk_cpp ==============
    skip_seq = os.environ.get("SKIP_SEQ") == "1"
    if not skip_seq:
        print("\n[A] per-chunk decompress (engine.decompress_chunk):")
        t0 = time.time()
        seq_decoded = [eng.decompress_chunk(e) for e in encoded]
        print(f"  done in {time.time() - t0:.1f}s")
        seq_pass = sum(1 for d, c in zip(seq_decoded, chunks) if d == c)
        print(f"  byte-exact: {seq_pass}/{n_chunks}")
    else:
        print("\n[A] per-chunk decompress: SKIPPED (SKIP_SEQ=1)")
        seq_decoded = chunks  # placeholder so comparison below doesn't crash
        seq_pass = n_chunks

    # ============== B: batched via _decompress_chunks_batched_cpp ==============
    print("\n[B] batched decompress (engine.decompress_chunks_batched):")
    t0 = time.time()
    bat_decoded = eng.decompress_chunks_batched(encoded)
    print(f"  done in {time.time() - t0:.1f}s")
    bat_pass = sum(1 for d, c in zip(bat_decoded, chunks) if d == c)
    print(f"  byte-exact: {bat_pass}/{n_chunks}")

    # Compare A and B per chunk
    print("\nA vs B per-chunk:")
    for i in range(n_chunks):
        match_input = bat_decoded[i] == chunks[i]
        match_seq = bat_decoded[i] == seq_decoded[i]
        print(f"  chunk {i}: bat==input={match_input} bat==seq={match_seq}")
        if not match_input and i == 0:
            # First diff offset
            n = min(len(bat_decoded[0]), len(chunks[0]))
            for j in range(n):
                if bat_decoded[0][j] != chunks[0][j]:
                    print(f"    first byte diff at offset {j}: "
                          f"got 0x{bat_decoded[0][j]:02x} "
                          f"expected 0x{chunks[0][j]:02x}")
                    break

    if seq_pass == n_chunks and bat_pass == n_chunks:
        print("\nALL PASS — both sequential and batched decompress are byte-exact.")
        raise SystemExit(0)
    elif seq_pass == n_chunks and bat_pass < n_chunks:
        print("\nFAIL ISOLATED TO BATCHED PATH — sequential is correct.")
        raise SystemExit(1)
    else:
        print(f"\nFAIL UPSTREAM — sequential also fails ({seq_pass}/{n_chunks}).")
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Local smoke test (no GPU, no RWKV-LM required).
Validates: blob header, chunking dispatch, AC encode/decode roundtrip.
Uses a mock model that returns random-but-deterministic logits.

Run: python scripts/smoke_test_local.py
Requires: pip install constriction tokenizers zstandard
"""

import sys
import os
import struct
import zlib
import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# 1. Blob header roundtrip
# ---------------------------------------------------------------------------

def test_blob_header():
    from krunch.inference import encode_header, decode_header, HEADER_SIZE, BLOB_MAGIC

    hdr = encode_header(original_len=123456, n_chunks=7, crc32=0xDEADBEEF)
    assert len(hdr) == HEADER_SIZE, f"header size mismatch: {len(hdr)} != {HEADER_SIZE}"

    parsed = decode_header(hdr)
    assert parsed["original_len"] == 123456
    assert parsed["n_chunks"] == 7
    assert parsed["crc32"] == 0xDEADBEEF
    assert parsed["blob_version"] == 1
    assert parsed["model_id"] == 1
    print("  PASS blob header encode/decode")


# ---------------------------------------------------------------------------
# 2. AC encode/decode roundtrip with deterministic fake logits
# ---------------------------------------------------------------------------

def test_ac_roundtrip():
    from krunch.inference import ac_encode, ac_decode, _softmax_np

    rng = random.Random(42)
    vocab = 256
    tokens = [rng.randint(0, vocab - 1) for _ in range(50)]

    # Fake logits: random but deterministic
    logits_seq = np.random.default_rng(42).standard_normal((len(tokens), vocab)).astype(np.float32)

    encoded = ac_encode(tokens, logits_seq)

    # Decode: replay the same logits in order — logits_fn ignores its inputs
    # because the test logits aren't actually conditional on prior tokens.
    logits_iter = iter(logits_seq)

    def logits_fn(state, token):
        logits = next(logits_iter)
        return logits, state

    decoded = ac_decode(encoded, len(tokens), logits_fn)
    assert decoded == tokens, f"AC decode mismatch:\n  got {decoded[:10]}\n  want {tokens[:10]}"
    print(f"  PASS AC encode/decode ({len(tokens)} tokens, {len(encoded)} bytes)")


# ---------------------------------------------------------------------------
# 3. Chunking roundtrip with mock neural codec
# ---------------------------------------------------------------------------

def test_chunking_roundtrip():
    from krunch.chunking import compress_all, decompress_all, CHUNK_SIZE

    raw = b"Hello, Krunch! " * (CHUNK_SIZE // 15 + 500)
    print(f"  Input: {len(raw):,} bytes, chunk size: {CHUNK_SIZE:,}")

    # Mock neural codec: identity. Lets us verify the chunking machinery
    # (entry packing/unpacking) without needing the real model.
    def passthrough(chunk: bytes) -> bytes:
        return chunk

    entries, n_chunks = compress_all(raw, passthrough)
    entries_bytes = b"".join(entries)
    print(f"  Chunks: {n_chunks}, entries total: {len(entries_bytes):,} bytes")

    recovered = decompress_all(entries_bytes, n_chunks, passthrough)
    assert recovered == raw, f"chunking roundtrip mismatch (len {len(recovered)} vs {len(raw)})"
    print(f"  PASS chunking roundtrip ({n_chunks} chunks)")


# ---------------------------------------------------------------------------
# 3b. Threaded decompress: byte-identical to sequential
# ---------------------------------------------------------------------------

def test_threaded_decompress_byte_identical():
    """Verify the threaded decompress path produces byte-identical output
    to the sequential one. Uses a fake stateful neural_fn (xor with a
    counter-assigned chunk index) to detect any thread state leakage —
    if state crossed threads the output would scramble."""
    import threading
    import struct as _struct
    import importlib
    import krunch.chunking
    from krunch.chunking import compress_all, CHUNK_SIZE

    # 5+ chunks so the threading path actually engages
    raw = (b"abcdefghijklmnopqrstuvwxyz" * 250_000)[:5 * CHUNK_SIZE + 12345]
    counter = [0]
    lock = threading.Lock()

    def encode(chunk: bytes) -> bytes:
        with lock:
            idx = counter[0]
            counter[0] += 1
        return _struct.pack(">I", idx) + bytes(b ^ (idx & 0xFF) for b in chunk)

    def decode(encoded: bytes) -> bytes:
        idx = _struct.unpack(">I", encoded[:4])[0]
        return bytes(b ^ (idx & 0xFF) for b in encoded[4:])

    entries, n = compress_all(raw, encode)
    entries_bytes = b"".join(entries)

    os.environ["KRUNCH_DECOMPRESS_BATCH"] = "1"
    importlib.reload(krunch.chunking)
    seq = krunch.chunking.decompress_all(entries_bytes, n, decode)

    os.environ["KRUNCH_DECOMPRESS_BATCH"] = "8"
    importlib.reload(krunch.chunking)
    par = krunch.chunking.decompress_all(entries_bytes, n, decode)

    assert seq == raw, f"sequential broken (len {len(seq)} vs {len(raw)})"
    assert par == seq, "threaded path produced different bytes than sequential"
    print(f"  PASS threaded decompress byte-identical ({n} chunks, "
          f"sequential vs 8-thread agree)")


# ---------------------------------------------------------------------------
# 4. CRC32 integrity check
# ---------------------------------------------------------------------------

def test_crc():
    data = b"test data for crc" * 100
    crc = zlib.crc32(data) & 0xFFFFFFFF
    assert zlib.crc32(data) & 0xFFFFFFFF == crc
    # Mutate one byte — should differ
    mutated = bytearray(data)
    mutated[50] ^= 0xFF
    assert zlib.crc32(bytes(mutated)) & 0xFFFFFFFF != crc
    print("  PASS CRC32 integrity check")


# ---------------------------------------------------------------------------
# 5. Tokenizer loads correctly
# ---------------------------------------------------------------------------

def test_tokenizer():
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    tok_path = os.path.join(model_dir, "20B_tokenizer.json")
    if not os.path.exists(tok_path):
        print(f"  SKIP tokenizer (not found at {tok_path})")
        return
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    enc = tok.encode("Hello, world!")
    dec = tok.decode(enc.ids)
    assert "Hello" in dec
    print(f"  PASS tokenizer load + encode/decode (vocab={tok.get_vocab_size()})")


# ---------------------------------------------------------------------------

def test_compute_chunk_size():
    """Dynamic chunk sizing: 64 KB floor for small files, ~total/(4·B)
    for large ones; deterministic across workers seeing same total."""
    import os
    from krunch.chunking import compute_chunk_size, _CHUNK_SIZE_FLOOR

    # Snapshot env so the explicit-pin test doesn't leak.
    saved_chunk = os.environ.pop("KRUNCH_CHUNK_SIZE", None)
    saved_target = os.environ.pop("KRUNCH_TARGET_B", None)
    try:
        # Default target_B=128 → target_chunks=512.

        # Tiny file: floors at 64 KB.
        assert compute_chunk_size(10_000) == _CHUNK_SIZE_FLOOR
        assert compute_chunk_size(0) == _CHUNK_SIZE_FLOOR

        # 10 MB: 10MB/512 = ~20 KB → floor wins → 64 KB.
        assert compute_chunk_size(10 * 1024 * 1024) == _CHUNK_SIZE_FLOOR

        # 100 MB: 100MB/512 = 200 KB → above floor.
        assert compute_chunk_size(100 * 1024 * 1024) == 200 * 1024

        # 1 GB: 1GB/512 = 2 MB → 512 chunks of 2 MB.
        assert compute_chunk_size(1024 * 1024 * 1024) == 2 * 1024 * 1024

        # Determinism across two calls with same total.
        assert compute_chunk_size(50_000_000) == compute_chunk_size(50_000_000)

        # KRUNCH_TARGET_B override changes the math.
        os.environ["KRUNCH_TARGET_B"] = "512"  # H100 class, 4×=2048 chunks
        assert compute_chunk_size(1024 * 1024 * 1024) == 512 * 1024
        del os.environ["KRUNCH_TARGET_B"]

        # Explicit KRUNCH_CHUNK_SIZE pin overrides everything.
        os.environ["KRUNCH_CHUNK_SIZE"] = "131072"
        assert compute_chunk_size(1024 * 1024 * 1024) == 131072
        assert compute_chunk_size(1000) == 131072
    finally:
        os.environ.pop("KRUNCH_CHUNK_SIZE", None)
        os.environ.pop("KRUNCH_TARGET_B", None)
        if saved_chunk is not None:
            os.environ["KRUNCH_CHUNK_SIZE"] = saved_chunk
        if saved_target is not None:
            os.environ["KRUNCH_TARGET_B"] = saved_target

    print(f"  PASS compute_chunk_size (floor / scale / pin / override)")


def test_dynamic_chunking_roundtrip():
    """compress_all without explicit chunk_size auto-derives from len(raw),
    and decompress_all roundtrips byte-exactly. Same data, different total
    sizes pick different chunk counts."""
    from krunch.chunking import compress_all, decompress_all

    def passthrough(b: bytes) -> bytes:
        return b

    # Small input: should fit in one chunk (64 KB floor).
    raw_small = b"abc" * 10_000  # 30 KB
    entries, n = compress_all(raw_small, passthrough)
    assert n == 1, f"30 KB should be 1 chunk, got {n}"
    assert decompress_all(b"".join(entries), n, passthrough) == raw_small

    # Big input: splits into multiple chunks; same chunk_size on both sides.
    raw_big = b"abc" * 200_000  # 600 KB → ceil(600K/64K) = 10 chunks
    entries, n = compress_all(raw_big, passthrough)
    assert n >= 9, f"600 KB should give ~10 chunks at 64K, got {n}"
    assert decompress_all(b"".join(entries), n, passthrough) == raw_big

    # Distributed semantics: passing total_size > len(raw) gives the
    # global chunk_size that the byte-range alignment used.
    entries_d, n_d = compress_all(raw_big, passthrough,
                                   total_size=10 * 1024 * 1024)
    # 10 MB total → 64 KB chunks; same as auto-derive for raw_big.
    assert n_d == n
    assert decompress_all(b"".join(entries_d), n_d, passthrough) == raw_big

    print("  PASS dynamic chunking roundtrip (auto-derive + total_size override)")


def main():
    print("Krunch local smoke test")
    print("=" * 40)
    tests = [
        test_blob_header,
        test_ac_roundtrip,
        test_chunking_roundtrip,
        test_compute_chunk_size,
        test_dynamic_chunking_roundtrip,
        test_threaded_decompress_byte_identical,
        test_crc,
        test_tokenizer,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            import traceback; traceback.print_exc()
            failures += 1
    print("=" * 40)
    if failures:
        print(f"FAILED: {failures}/{len(tests)} tests")
        sys.exit(1)
    else:
        print(f"ALL PASS ({len(tests)}/{len(tests)})")


if __name__ == "__main__":
    main()

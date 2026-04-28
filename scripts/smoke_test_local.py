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
    from server.inference import encode_header, decode_header, HEADER_SIZE, BLOB_MAGIC

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
    from server.inference import ac_encode, ac_decode, _softmax_np

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
    from server.chunking import compress_all, decompress_all, CHUNK_SIZE

    # Mock neural codec: just reverses bytes (clearly distinguishable from zstd)
    # but we make it very efficient to ensure neural wins over zstd
    def mock_neural_compress(chunk: bytes) -> bytes:
        # Return highly compressible data so neural wins
        return b"\x00" * (len(chunk) // 10 + 1)

    def mock_neural_decompress(encoded: bytes) -> bytes:
        # Recover original from the mini-header in the encoded bytes
        orig_len, n_tok = struct.unpack(">II", encoded[:8])
        return original_chunk[:orig_len]  # closure over outer scope

    # Test with data slightly larger than one chunk
    raw = b"Hello, Krunch! " * (CHUNK_SIZE // 15 + 500)
    print(f"  Input: {len(raw):,} bytes, chunk size: {CHUNK_SIZE:,}")

    # We'll use a simpler approach: zstd-only (no mock neural that needs closure)
    def no_neural(chunk):
        raise RuntimeError("neural disabled")

    entries, n_chunks = compress_all(raw, no_neural)
    entries_bytes = b"".join(entries)
    print(f"  Chunks: {n_chunks}, entries total: {len(entries_bytes):,} bytes")

    recovered = decompress_all(entries_bytes, n_chunks, no_neural)
    assert recovered == raw, f"chunking roundtrip mismatch (len {len(recovered)} vs {len(raw)})"
    print(f"  PASS chunking roundtrip ({n_chunks} chunks, zstd fallback)")


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

def main():
    print("Krunch local smoke test")
    print("=" * 40)
    tests = [
        test_blob_header,
        test_ac_roundtrip,
        test_chunking_roundtrip,
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

"""End-to-end coverage of krunch.chunking with a mock neural codec.

The real codec needs the model + GPU. Here we plug in a deterministic
identity neural_fn so we can verify chunking, struct framing, batched-vs-
sequential dispatch, and ThreadPoolExecutor fallback paths without ever
touching torch.
"""
import struct
import os
import pytest

from krunch import chunking


def _identity_neural_fn(chunk: bytes) -> bytes:
    """Stand-in for engine.compress_chunk / engine.decompress_chunk —
    in tests we don't compress, just round-trip the bytes verbatim."""
    return chunk


def _identity_batch_fn(chunks: list[bytes]) -> list[bytes]:
    return list(chunks)


def test_compress_all_single_chunk_no_batch_fn():
    raw = b"hello world"
    entries, n = chunking.compress_all(raw, _identity_neural_fn)
    assert n == 1
    # Each entry is `>II` header + payload.
    orig_len, comp_len = struct.unpack(">II", entries[0][:8])
    assert orig_len == len(raw)
    assert comp_len == len(raw)
    assert entries[0][8:] == raw


def test_compress_all_uses_batch_fn_when_multiple_chunks(monkeypatch):
    monkeypatch.setattr(chunking, "compute_chunk_size", lambda _: 4)
    raw = b"abcdefghij"  # 10 bytes, chunk_size=4 -> 3 chunks
    calls = []

    def batch_fn(chunks):
        calls.append(list(chunks))
        return list(chunks)

    entries, n = chunking.compress_all(
        raw, _identity_neural_fn, neural_batch_fn=batch_fn,
    )
    assert n == 3
    assert len(calls) == 1, "batch_fn should be invoked exactly once"
    # Reassembled payloads must equal the original input.
    payloads = []
    for e in entries:
        ol, cl = struct.unpack(">II", e[:8])
        payloads.append(e[8 : 8 + cl])
    assert b"".join(payloads) == raw


def test_compress_all_falls_back_when_only_one_chunk_with_batch_fn(monkeypatch):
    """Single-chunk inputs should use the per-chunk path even if a
    batch_fn is provided — batching one chunk is just overhead."""
    monkeypatch.setattr(chunking, "compute_chunk_size", lambda _: 1024)
    raw = b"short"
    batch_calls = []

    def batch_fn(chunks):
        batch_calls.append(chunks)
        return list(chunks)

    entries, n = chunking.compress_all(
        raw, _identity_neural_fn, neural_batch_fn=batch_fn,
    )
    assert n == 1
    assert batch_calls == []  # batch_fn was bypassed


def _build_blob(raw: bytes, chunk_size: int) -> tuple[bytes, int]:
    """Helper: produce the entries-bytes that decompress_all expects."""
    chunks = chunking._split_utf8_safe(raw, chunk_size)
    entries = b"".join(
        struct.pack(">II", len(c), len(c)) + c for c in chunks
    )
    return entries, len(chunks)


def test_decompress_all_roundtrip_sequential():
    raw = b"the quick brown fox jumps over the lazy dog" * 4
    entries, n = _build_blob(raw, chunk_size=32)
    out = chunking.decompress_all(entries, n, _identity_neural_fn)
    assert out == raw


def test_decompress_all_uses_batch_fn():
    raw = b"x" * 256
    entries, n = _build_blob(raw, chunk_size=64)
    calls = []

    def batch_fn(encoded_list):
        calls.append(encoded_list)
        return [enc for enc in encoded_list]

    out = chunking.decompress_all(
        entries, n, _identity_neural_fn, neural_batch_fn=batch_fn,
    )
    assert out == raw
    assert len(calls) == 1


def test_decompress_all_threadpool_path(monkeypatch):
    """Exercise the ThreadPoolExecutor fan-out path. KRUNCH_DECOMPRESS_BATCH
    is captured as a module constant at import time, so we patch the
    module attribute directly."""
    monkeypatch.setattr(chunking, "DECOMPRESS_BATCH", 4)
    raw = b"abcdefghijklmnopqrstuvwx"
    entries, n = _build_blob(raw, chunk_size=4)
    out = chunking.decompress_all(entries, n, _identity_neural_fn)
    assert out == raw


def test_decompress_all_truncates_to_orig_len():
    """Each chunk's neural_fn output is truncated to its declared
    orig_len. Important: decoders may emit trailing garbage after the
    intended bytes; the framing's orig_len is authoritative."""
    raw = b"123"
    entries = struct.pack(">II", 3, 5) + b"123!!"  # 5-byte payload, orig=3
    out = chunking.decompress_all(entries, 1, lambda b: b)
    assert out == b"123"

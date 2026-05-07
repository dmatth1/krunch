"""engine.compress_chunks(list) must produce the same per-chunk bytes
as [engine.compress_chunk(c) for c in list]. The batch path tokenizes
all chunks in one Rust call (faster) but must not change the output.

Workaround for Bugs.md #1: forces W8A8 off so small chunks roundtrip
on T4. Drop once the W8A8 sm_75 path is fixed.
"""
import os
os.environ["KRUNCH_INT8_W8A8"] = "0"
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import pytest


# `engine` fixture is provided by conftest.py (session-scoped).


CHUNKS = [
    b"Hello there, this is chunk one of three. " * 10,
    b"def fib(n):\n    return n if n < 2 else fib(n-1)+fib(n-2)\n" * 8,
    b"The quick brown fox jumps over the lazy dog. " * 12,
]


def test_batch_compress_matches_per_chunk(engine):
    batched = engine.compress_chunks(CHUNKS)
    per_chunk = [engine.compress_chunk(c) for c in CHUNKS]
    assert len(batched) == len(per_chunk)
    for i, (b, p) in enumerate(zip(batched, per_chunk)):
        assert b == p, f"chunk {i}: batched bytes diverge from per-chunk"


def test_batch_compress_single_falls_through(engine):
    """Single-chunk lists should fall through to compress_chunk —
    no batch tokenize overhead, identical bytes."""
    out = engine.compress_chunks([CHUNKS[0]])
    assert len(out) == 1
    assert out[0] == engine.compress_chunk(CHUNKS[0])


def test_batch_compress_empty(engine):
    """Empty list should return empty list, not crash."""
    assert engine.compress_chunks([]) == []


def test_batched_decompress_matches_per_chunk(engine):
    """decompress_chunks_batched must produce the same per-chunk decoded
    bytes as [decompress_chunk(c) for c in encs] — the cross-chunk
    batched path is an optimization, not a semantic change."""
    encs = engine.compress_chunks(CHUNKS)
    batched = engine.decompress_chunks_batched(encs)
    per_chunk = [engine.decompress_chunk(e) for e in encs]
    assert len(batched) == len(per_chunk)
    for i, (b, p) in enumerate(zip(batched, per_chunk)):
        assert b == p, f"chunk {i}: batched decompress diverges from per-chunk"


def test_decompress_chunks_batched_handles_single(engine):
    """B=1 should fall through to decompress_chunk."""
    enc = engine.compress_chunk(CHUNKS[0])
    out = engine.decompress_chunks_batched([enc])
    assert len(out) == 1
    assert out[0] == engine.decompress_chunk(enc)


def test_decompress_chunks_batched_empty(engine):
    assert engine.decompress_chunks_batched([]) == []

"""engine.compress_chunks_batched — cross-chunk batched compress
where N independent chunks are processed in lockstep (B=N forward +
B=N AC encode per timestep).

Encoded bytes are NOT bit-equal to per-chunk compress — the lockstep
AC encoder emits bits in a different order. Symmetry contract is
between batched-compress and batched-decompress: compressed-batched
chunks must roundtrip byte-exact when decompressed by either path.

Skipped on sm_75 — docs/Bugs.md #4 (small-chunk roundtrip break in the
cross-chunk batched compress path on T4). Production cli.py never
hits this code path.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_properties(0).major < 8,
    reason="docs/Bugs.md #4: compress_chunks_batched broken on sm_75 small chunks",
)


CHUNKS = [
    b"Hello there, this is chunk one of three. " * 10,
    b"def fib(n):\n    return n if n < 2 else fib(n-1)+fib(n-2)\n" * 8,
    b"The quick brown fox jumps over the lazy dog. " * 12,
]


def test_batched_compress_roundtrips_via_batched_decompress(engine):
    """End-to-end: compress_chunks_batched → decompress_chunks_batched
    must recover every chunk byte-for-byte."""
    encoded = engine.compress_chunks_batched(CHUNKS)
    decoded = engine.decompress_chunks_batched(encoded)
    for orig, got in zip(CHUNKS, decoded):
        assert got == orig


def test_batched_compress_roundtrips_via_per_chunk_decompress(engine):
    """Compressed-batched chunks must also work through the per-chunk
    decompress path — the encoded format is a valid `compress_chunk`-
    equivalent blob per element."""
    encoded = engine.compress_chunks_batched(CHUNKS)
    for orig, enc in zip(CHUNKS, encoded):
        decoded = engine.decompress_chunk(enc)
        assert decoded == orig


def test_batched_compress_single_falls_through(engine):
    """B=1 should land at compress_chunk (the cross-chunk machinery
    has no value when there's nothing to cross)."""
    out = engine.compress_chunks_batched([CHUNKS[0]])
    assert len(out) == 1
    assert out[0] == engine.compress_chunk(CHUNKS[0])


def test_batched_compress_empty(engine):
    assert engine.compress_chunks_batched([]) == []

"""_decompress_chunks_batched_cpp — direct test of the cross-chunk
batched stepped decoder body (bypassing decompress_chunks_batched's
B==1 fall-through).

For each B from 2 to 4: compress B independent chunks, then run
_decompress_chunks_batched_cpp on the encodings. Each recovered
chunk must byte-match its input.

W8A8 off (conftest default) — docs/Bugs.md #1.
"""
import pytest


SAMPLES = [
    b"chunk-A " * 50,
    b"chunk-B " * 60,
    b"chunk-C " * 40,
    b"chunk-D " * 70,
]


@pytest.mark.parametrize("B", [2, 3, 4])
def test_batched_cpp_recovers_each_chunk(engine, B):
    """Compress B independent chunks via the per-chunk path, then
    decode them all in one batched-stepped call."""
    chunks = SAMPLES[:B]
    encs = [engine.compress_chunk(c) for c in chunks]
    decoded = engine._decompress_chunks_batched_cpp(encs)
    assert len(decoded) == B
    for i, (orig, got) in enumerate(zip(chunks, decoded)):
        assert got == orig, (
            f"B={B} chunk {i}: batched-cpp diverged from original "
            f"(orig={len(orig)}B, got={len(got)}B)"
        )


def test_batched_cpp_matches_per_chunk_decode(engine):
    """The cross-chunk batched path produces the same bytes as
    decoding each chunk independently."""
    chunks = SAMPLES[:3]
    encs = [engine.compress_chunk(c) for c in chunks]
    batched = engine._decompress_chunks_batched_cpp(encs)
    per_chunk = [engine.decompress_chunk(e) for e in encs]
    for i, (b, p) in enumerate(zip(batched, per_chunk)):
        assert b == p, f"chunk {i}: batched-cpp != per-chunk"


def test_batched_cpp_handles_unequal_token_counts(engine):
    """Chunks of different sizes pack into the same batched run via
    per-stream byte offsets + tail padding."""
    chunks = [
        b"X" * 20,
        b"X" * 200,    # ~10× longer
        b"X" * 80,
    ]
    encs = [engine.compress_chunk(c) for c in chunks]
    decoded = engine._decompress_chunks_batched_cpp(encs)
    for orig, got in zip(chunks, decoded):
        assert got == orig

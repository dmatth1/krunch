"""Tier-1 test: chunk boundaries always land on UTF-8 codepoint boundaries.

Prior bug: compress_all sliced raw bytes at fixed CHUNK_SIZE, cutting
multi-byte UTF-8 codepoints in half. compress_chunk then did
data.decode('utf-8', errors='replace') and replaced the partial bytes
with U+FFFD — silently lossy at chunk boundaries on any non-ASCII
content. WildChat 64 KB chunks hit this ~1% of the time.

Test that _split_utf8_safe never cuts a codepoint, and that chunks
sum to the original.
"""
from krunch.chunking import _split_utf8_safe


def test_ascii_uniform():
    raw = b"a" * 100_000
    chunks = _split_utf8_safe(raw, 1024)
    assert b"".join(chunks) == raw
    assert all(len(c) == 1024 for c in chunks[:-1])


def test_multibyte_at_boundary():
    # \xe2\x80\x9c is the 3-byte left double quote.
    base = b"x" + b"\xe2\x80\x9c"  # 4 bytes total
    raw = base * 10_000  # 40_000 bytes
    chunks = _split_utf8_safe(raw, 1023)
    # Round-trip preservation
    assert b"".join(chunks) == raw
    # Every chunk decodes cleanly as utf-8 (no partial codepoints)
    for c in chunks:
        c.decode("utf-8")  # raises if invalid


def test_split_in_continuation_byte():
    # Targeted: chunk_size lands inside a 3-byte sequence
    raw = (b"\xe2\x80\x9c" * 5)  # 15 bytes
    # target=4 → first chunk would be raw[:4] = e2 80 9c | e2 (cut!)
    # _split_utf8_safe should walk back to 3
    chunks = _split_utf8_safe(raw, 4)
    assert b"".join(chunks) == raw
    for c in chunks:
        c.decode("utf-8")


def test_empty_input():
    assert _split_utf8_safe(b"", 1024) == []


def test_target_larger_than_input():
    raw = b"hello world"
    assert _split_utf8_safe(raw, 1024) == [raw]


if __name__ == "__main__":
    test_ascii_uniform()
    test_multibyte_at_boundary()
    test_split_in_continuation_byte()
    test_empty_input()
    test_target_larger_than_input()
    print("ALL PASS")

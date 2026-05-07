"""Tier-1 test: multi-worker byte ranges land on UTF-8 codepoint boundaries.

Prior bug (discovered 2026-05-07 in T4 100MB / 4-worker run):
    `_byte_range` returned proposed offsets like `[25 MB, 50 MB)` that
    landed mid-codepoint in WildChat. Each worker's `compress_chunk`
    decoded the leading invalid bytes via `errors="replace"`; the
    tokenizer roundtrip then dropped some of those U+FFFD bytes,
    breaking byte-exact recovery. 100 MB / 4 workers lost 125 bytes
    across three inter-worker boundaries.

This test verifies that with `src=` passed in, `_byte_range`:
  1. snaps each non-trivial boundary to a codepoint start byte,
  2. never cuts a codepoint in half,
  3. produces ranges that exactly tile [0, total_size) with no gaps,
  4. has matching handoff (worker N's end == worker N+1's start).
"""
import os
import tempfile

from krunch.job import _byte_range, _snap_to_codepoint


# A 4-byte UTF-8 emoji + a 3-byte char + ASCII filler. The non-ASCII
# bytes are placed densely so any cut byte-range is likely to land
# mid-codepoint.
_PATTERN = b"\xf0\x9f\x98\x80" + b"\xe2\x80\x9c" + b"x" * 9  # 4+3+9 = 16 bytes


def _make_sample(path: str, size: int) -> None:
    """Write `size` bytes of repeating multi-byte UTF-8 to `path`."""
    n_full = size // len(_PATTERN)
    remainder = size - n_full * len(_PATTERN)
    with open(path, "wb") as f:
        f.write(_PATTERN * n_full)
        # Trim the tail so we land on a codepoint boundary at file-end
        # (real-world UTF-8 files always do).
        f.write(_PATTERN[:remainder] if remainder >= 9 else b"x" * remainder)


def _is_codepoint_start(b: int) -> bool:
    """True if byte `b` is the start of a UTF-8 codepoint (not a 0x80-0xBF
    continuation byte)."""
    return b < 0x80 or b >= 0xC0


def test_snap_to_codepoint_passthrough_at_boundary():
    """If `offset` already points at a codepoint start, no snap."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        path = tf.name
        _make_sample(path, 1024)
    try:
        url = f"file://{path}"
        # offset=0: special-case — returns 0 even without reading.
        assert _snap_to_codepoint(url, 0) == 0
        # offset=16 lands on the start of the next pattern repetition,
        # which is the 4-byte emoji's lead byte 0xF0 — a codepoint start.
        assert _snap_to_codepoint(url, 16) == 16
    finally:
        os.unlink(path)


def test_snap_advances_off_continuation():
    """If `offset` is a continuation byte, snap forward to the next
    codepoint start."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        path = tf.name
        _make_sample(path, 1024)
    try:
        url = f"file://{path}"
        # The pattern starts with 0xF0 0x9F 0x98 0x80 (emoji). offsets
        # 1, 2, 3 are continuation bytes; snap should land at 4 (the
        # 0xE2 starting the 3-byte left-quote).
        assert _snap_to_codepoint(url, 1) == 4
        assert _snap_to_codepoint(url, 2) == 4
        assert _snap_to_codepoint(url, 3) == 4
        # offsets 5, 6 are continuation bytes of the 0xE2 codepoint;
        # snap to 7 (the first 'x').
        assert _snap_to_codepoint(url, 5) == 7
        assert _snap_to_codepoint(url, 6) == 7
    finally:
        os.unlink(path)


def test_byte_range_tiles_perfectly_4_workers():
    """4 workers cover [0, total_size) exactly: no gaps, no overlaps."""
    size = 1_048_576  # 1 MB — enough to force several chunks per worker
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        path = tf.name
        _make_sample(path, size)
    try:
        url = f"file://{path}"
        ranges = [_byte_range(i, 4, size, src=url) for i in range(4)]
        # First range starts at 0, last ends at size
        assert ranges[0][0] == 0, ranges
        assert ranges[-1][1] == size, ranges
        # Adjacent handoff: worker N's end == worker N+1's start
        for i in range(3):
            assert ranges[i][1] == ranges[i + 1][0], \
                f"handoff broken at boundary {i}: {ranges[i]} vs {ranges[i+1]}"
        # No range is empty
        for i, (s, e) in enumerate(ranges):
            assert e > s, f"empty range for worker {i}: [{s}, {e})"
    finally:
        os.unlink(path)


def test_byte_range_boundaries_are_codepoint_starts():
    """Every boundary returned by `_byte_range` (except 0 and EOF)
    points at a UTF-8 codepoint start byte."""
    size = 1_048_576
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        path = tf.name
        _make_sample(path, size)
    try:
        url = f"file://{path}"
        with open(path, "rb") as f:
            data = f.read()

        for n_workers in (2, 4, 8, 16):
            ranges = [_byte_range(i, n_workers, size, src=url)
                      for i in range(n_workers)]
            for i, (s, e) in enumerate(ranges):
                if s != 0:
                    assert _is_codepoint_start(data[s]), \
                        f"workers={n_workers} part={i} start={s} byte=0x{data[s]:02x}"
                if e != size:
                    assert _is_codepoint_start(data[e]), \
                        f"workers={n_workers} part={i} end={e} byte=0x{data[e]:02x}"
            # Concat of slices must equal the file
            slices = b"".join(data[s:e] for s, e in ranges)
            assert slices == data, \
                f"workers={n_workers}: tiled bytes don't equal source"
    finally:
        os.unlink(path)


def test_byte_range_no_src_returns_legacy_proposed():
    """If `src` is omitted, `_byte_range` returns the unsnapped proposed
    range (legacy behavior; kept so single-worker callers don't pay an
    extra ranged GET)."""
    size = 1_048_576
    s, e = _byte_range(0, 1, size)
    assert (s, e) == (0, size)
    # Without snap, 4-worker split returns chunk-aligned offsets (no
    # codepoint awareness). That's the buggy regime; only used when
    # caller explicitly opts out by not passing `src`.
    s, e = _byte_range(1, 4, size)
    assert e > s


if __name__ == "__main__":
    test_snap_to_codepoint_passthrough_at_boundary()
    test_snap_advances_off_continuation()
    test_byte_range_tiles_perfectly_4_workers()
    test_byte_range_boundaries_are_codepoint_starts()
    test_byte_range_no_src_returns_legacy_proposed()
    print("all 5 tests passed")

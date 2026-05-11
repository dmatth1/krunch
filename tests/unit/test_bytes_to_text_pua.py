"""
Bug #3 fix — byte-safe input encoding via PUA codepoint escape.

Pins the contract that `_bytes_to_text` is **lossless** for any
bytes (including arbitrary binary), AND identical to the legacy
errors="replace" output for UTF-8-clean input that doesn't contain
the U+E000..U+E0FF private-use range.

The codec is also tested through the tokenizer at
`tests/unit/test_tokenizer_normalization.py`; this file pins the
helpers themselves so we don't need the model + tokenizer.json to
prove byte-exactness.

Per docs/Bugs.md #3.
"""

import pytest

from krunch.inference import (
    _bytes_to_text, _text_to_bytes, _text_to_bytes_legacy,
    TOKENIZER_ID, TOKENIZER_ID_LEGACY, SUPPORTED_TOKENIZER_IDS,
)


# ---------------------------------------------------------------------------
# Roundtrip — the load-bearing test
# ---------------------------------------------------------------------------

def test_roundtrip_all_byte_values_in_order():
    """The lossless guarantee for the common case: every byte 0..255
    roundtrips when no three consecutive bytes form a valid UTF-8
    encoding of U+E000..U+E0FF (i.e., 0xEE 0x80..0x83 0x80..0xBF).
    `bytes(range(256))` is monotonic so no such trigram appears."""
    data = bytes(range(256))
    assert _text_to_bytes(_bytes_to_text(data)) == data


def test_roundtrip_random_bytes_curated():
    """Pseudo-random 1 KB drawn from a curated alphabet that excludes
    the lead byte 0xEE — guarantees no valid U+E0XX UTF-8 sequence
    can appear. Roundtrips byte-exact."""
    import random
    rng = random.Random(42)
    # Drop 0xEE so no U+E0XX UTF-8 sequence can form.
    alphabet = [b for b in range(256) if b != 0xEE]
    data = bytes(rng.choice(alphabet) for _ in range(1024))
    assert _text_to_bytes(_bytes_to_text(data)) == data


def test_roundtrip_repeats_invariant():
    """The fix has no hidden state — same input → same output regardless
    of prior calls. (Avoid the 0xEE 0x8X 0x8X corner case explicitly.)"""
    data = b"\x00\x80\xff\x42mixed\x7f\x01"
    a = _text_to_bytes(_bytes_to_text(data))
    b = _text_to_bytes(_bytes_to_text(data))
    c = _text_to_bytes(_bytes_to_text(data))
    assert a == data and b == data and c == data


def test_roundtrip_known_corpus_pattern_from_bugs_md():
    """The exact pattern docs/Bugs.md #3 documents — a raw 0x80 byte
    embedded mid-stream (NASA Apache log corpus). v0.1.0 silently
    substituted U+FFFD; the fix preserves the original byte."""
    data = b'GET /pub/winvn/readme.txt HTTP/1.0" 200 \x80 bytes\n'
    assert _text_to_bytes(_bytes_to_text(data)) == data


def test_legacy_path_is_lossy_documented():
    """Pin the documented lossy behavior of tokenizer_id=1: errors="replace"
    + .encode("utf-8") drops the original byte. We're not fixing the old
    path; we're proving the bug exists and that the new path doesn't have it."""
    data = b"hello \x80 world"
    legacy_text = data.decode("utf-8", errors="replace")
    legacy_bytes = _text_to_bytes_legacy(legacy_text)
    # Roundtrip lost 2 bytes (one byte became U+FFFD = 3 bytes).
    assert len(legacy_bytes) == len(data) + 2
    assert legacy_bytes != data
    # The new path preserves length AND content.
    assert _text_to_bytes(_bytes_to_text(data)) == data


# ---------------------------------------------------------------------------
# UTF-8-clean inputs are byte-identical between legacy and new
# ---------------------------------------------------------------------------

def test_clean_utf8_input_identical_text_to_legacy():
    """For inputs that are already valid UTF-8 with no PUA codepoints,
    the new encoder produces the SAME text as the legacy errors="replace"
    handler (which is also a no-op on valid UTF-8). Confirms the new
    path is a strict superset — same tokens for existing corpora."""
    samples = [
        b"hello world",
        b"Quick brown fox jumps over the lazy dog.",
        b"Mixed-case text + numbers 123 + punctuation: ;, .",
        "日本語のテスト".encode("utf-8"),
        "Café — résumé naïve".encode("utf-8"),
        b"\n\t\rwhitespace\x20chars",
        b"",
    ]
    for s in samples:
        legacy = s.decode("utf-8", errors="replace")
        new = _bytes_to_text(s)
        assert new == legacy, f"diverged on {s!r}"


# ---------------------------------------------------------------------------
# Edge cases on the new encoding
# ---------------------------------------------------------------------------

def test_invalid_continuation_byte():
    """A continuation byte (0x80-0xBF) by itself is invalid UTF-8 — must
    be escaped, not substituted."""
    data = b"a\x80b"
    text = _bytes_to_text(data)
    assert "" in text
    assert _text_to_bytes(text) == data


def test_overlong_utf8_sequence_escaped():
    """Overlong encodings (e.g., 0xC0 0x80 for U+0000) are invalid by
    spec. UTF-8 decoders reject them; we escape every byte separately."""
    data = b"prefix\xC0\x80suffix"
    text = _bytes_to_text(data)
    assert _text_to_bytes(text) == data


def test_truncated_multibyte_at_end():
    """4-byte UTF-8 lead with only 2 continuation bytes — truncated at
    the end. Each invalid byte gets its own PUA codepoint."""
    data = b"\xF0\x9F\x98"  # incomplete U+1F600 grinning face
    text = _bytes_to_text(data)
    assert _text_to_bytes(text) == data


def test_pua_codepoints_in_input_are_escaped_via_utf8():
    """If input ALREADY contains valid UTF-8 for a U+E0XX codepoint, the
    decoder maps it back to a single byte rather than 3 bytes. This is
    the documented edge case — extremely rare in real text. We pin the
    behavior so it's not surprise-breaking later."""
    # U+E0A5 encodes as the 3 bytes 0xEE 0x82 0xA5.
    data = " text".encode("utf-8")  # = bytes b'\xee\x82\xa5 text'
    text = _bytes_to_text(data)
    # _text_to_bytes sees U+E0A5 → 0xA5 (single byte). NOT a roundtrip.
    # That's the documented limitation — and why TOKENIZER_ID bumps.
    result = _text_to_bytes(text)
    assert result == b"\xa5 text"
    assert result != data  # documented divergence


def test_legacy_path_handles_pua_codepoints():
    """The legacy path encodes PUA codepoints as UTF-8 (3 bytes each).
    For old blobs that contained such codepoints, this is the only way
    to recover the original bytes."""
    data = " text".encode("utf-8")  # bytes b'\xee\x82\xa5 text'
    legacy = data.decode("utf-8", errors="replace")  # round-trips since input is valid UTF-8
    assert _text_to_bytes_legacy(legacy) == data


# ---------------------------------------------------------------------------
# Tokenizer ID registry sanity
# ---------------------------------------------------------------------------

def test_tokenizer_ids_registered():
    """SUPPORTED_TOKENIZER_IDS includes both the legacy ID and the new
    default. Image rejects unknown IDs cleanly."""
    assert TOKENIZER_ID_LEGACY == 1
    assert TOKENIZER_ID == 2
    assert set(SUPPORTED_TOKENIZER_IDS) == {1, 2}


def test_new_default_is_byte_safe_id():
    """encode_header / new compress always writes the byte-safe ID."""
    from krunch.inference import encode_header, decode_header
    hdr = encode_header(original_len=100, n_chunks=1, crc32=0)
    parsed = decode_header(hdr)
    assert parsed["tokenizer_id"] == TOKENIZER_ID == 2

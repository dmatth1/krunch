"""
Blob format versioning contract.

A krunch image must reject blobs produced with an incompatible
model_id, tokenizer_id, or blob_version. Silently producing garbage
when the model changed is a worse failure mode than a clean error.

This test file pins the contract: any future change to the blob
format that breaks readability of older blobs MUST bump model_id
(or blob_version, depending on what changed). See V1_PLAN.md and
docs/TIER_3_CLEANUP.md §5.4 for versioning policy.
"""

import struct
import pytest

from krunch.inference import (
    encode_header, decode_header, HEADER_FMT, HEADER_SIZE,
    BLOB_MAGIC, BLOB_VERSION, MODEL_ID, MODEL_ID_ADAPTIVE,
    SUPPORTED_MODEL_IDS, TOKENIZER_ID,
    IncompatibleBlobError,
    MAX_CHUNKS, MAX_ORIGINAL_LEN, MAX_CHUNK_BYTES,
)


def _hand_pack(blob_version=BLOB_VERSION, model_id=MODEL_ID,
               tokenizer_id=TOKENIZER_ID, adapter_id=0, adapter_version=0,
               flags=0, original_len=1234, n_chunks=1, crc32=0,
               magic=BLOB_MAGIC) -> bytes:
    """Build a header by hand so we can inject incompatible field values
    that encode_header() would never produce."""
    return struct.pack(
        HEADER_FMT,
        magic, blob_version,
        model_id, tokenizer_id,
        adapter_id, adapter_version, flags,
        original_len, n_chunks, crc32,
    )


def test_round_trip_current_image():
    """Blobs produced by encode_header are accepted by decode_header."""
    hdr = encode_header(original_len=999, n_chunks=2, crc32=0xCAFEBABE)
    parsed = decode_header(hdr)
    assert parsed["blob_version"] == BLOB_VERSION
    assert parsed["model_id"] == MODEL_ID
    assert parsed["tokenizer_id"] == TOKENIZER_ID
    assert parsed["original_len"] == 999
    assert parsed["n_chunks"] == 2
    assert parsed["crc32"] == 0xCAFEBABE


def test_reject_bad_magic():
    bad = _hand_pack(magic=b"XXXX")
    with pytest.raises(ValueError, match="bad magic"):
        decode_header(bad)


def test_reject_short_header():
    with pytest.raises(ValueError, match="too short"):
        decode_header(b"abc")


def test_reject_unknown_model_id():
    """A future image using a different model must not silently produce
    garbage when fed an old blob — and vice versa. model_id mismatch is
    a hard error."""
    unknown_id = max(SUPPORTED_MODEL_IDS) + 1
    bad = _hand_pack(model_id=unknown_id)
    with pytest.raises(IncompatibleBlobError, match="model_id"):
        decode_header(bad)


def test_accepts_adaptive_head_model_id():
    """MODEL_ID_ADAPTIVE (NEXT-3) is also supported by this image. A
    blob produced with KRUNCH_ADAPTIVE_HEAD=1 must round-trip through
    decode_header without IncompatibleBlobError."""
    hdr = _hand_pack(model_id=MODEL_ID_ADAPTIVE)
    parsed = decode_header(hdr)
    assert parsed["model_id"] == MODEL_ID_ADAPTIVE


def test_reject_unknown_tokenizer_id():
    bad = _hand_pack(tokenizer_id=TOKENIZER_ID + 1)
    with pytest.raises(IncompatibleBlobError, match="tokenizer_id"):
        decode_header(bad)


def test_reject_unknown_blob_version():
    """blob_version is the framing version (header + entry layout).
    A future blob_version may be unreadable even if model_id matches."""
    bad = _hand_pack(blob_version=BLOB_VERSION + 1)
    with pytest.raises(IncompatibleBlobError, match="blob_version"):
        decode_header(bad)


def test_strict_false_returns_fields_without_check():
    """Inspection / testing use case — caller wants to see the header
    fields even if they're incompatible, e.g. to print a useful error."""
    bad = _hand_pack(model_id=MODEL_ID + 99)
    parsed = decode_header(bad, strict=False)
    assert parsed["model_id"] == MODEL_ID + 99


def test_header_size_constant_matches_struct():
    """HEADER_SIZE must match HEADER_FMT — an off-by-one here would
    silently chop the first byte of the entries section."""
    assert HEADER_SIZE == struct.calcsize(HEADER_FMT)


def test_adapter_fields_round_trip():
    hdr = encode_header(original_len=10, n_chunks=1, crc32=0,
                        adapter_id=42, adapter_version=7, flags=0x05)
    parsed = decode_header(hdr)
    assert parsed["adapter_id"] == 42
    assert parsed["adapter_version"] == 7
    assert parsed["flags"] == 0x05


def test_reject_oversized_n_chunks():
    """A malformed/malicious blob declaring n_chunks > MAX_CHUNKS must
    be rejected up-front. Without this cap a 2^32-1 n_chunks would
    spin the decoder loop and OOM long before CRC32 ever runs."""
    bad = _hand_pack(n_chunks=MAX_CHUNKS + 1)
    with pytest.raises(IncompatibleBlobError, match="MAX_CHUNKS"):
        decode_header(bad)


def test_reject_oversized_original_len():
    """Same for original_len — declared u64 in the header. Cap at
    MAX_ORIGINAL_LEN to bound the post-decode buffer."""
    bad = _hand_pack(original_len=MAX_ORIGINAL_LEN + 1)
    with pytest.raises(IncompatibleBlobError, match="MAX_ORIGINAL_LEN"):
        decode_header(bad)


def test_max_caps_orders_of_magnitude_above_v1_workload():
    """Sanity: the caps must be far above any realistic v1 input. 10 GB
    at the 64 KB chunk floor is ~163K chunks; the cap should be way
    above that to give headroom but well below u32-max."""
    assert MAX_CHUNKS >= 10_000_000  # 10M minimum headroom
    assert MAX_CHUNKS < 2**32           # below u32-max so the header field can't overflow
    assert MAX_ORIGINAL_LEN >= 10**11   # ≥ 100 GB realistic
    assert MAX_ORIGINAL_LEN < 2**64
    assert MAX_CHUNK_BYTES >= 1 << 20   # ≥ 1 MB (well above 64 KB chunk floor)


def test_per_chunk_size_cap_rejects_oversized_entry():
    """chunking.decompress_all must reject per-chunk orig_len/comp_len
    exceeding MAX_CHUNK_BYTES. Catches malformed mini-headers inside
    the entries section."""
    from krunch.chunking import decompress_all
    # Build an entries blob whose first chunk declares orig_len =
    # MAX_CHUNK_BYTES + 1.
    bogus_orig_len = MAX_CHUNK_BYTES + 1
    bogus = struct.pack(">II", bogus_orig_len, 0)
    with pytest.raises(ValueError, match="MAX_CHUNK_BYTES"):
        decompress_all(bogus, n_chunks=1, neural_fn=lambda b: b)


def test_header_format_string_is_frozen():
    """If this assertion fires, the on-disk format changed: bump
    BLOB_VERSION or MODEL_ID and update CHANGELOG before merging.
    Old blobs with the old format will not be readable by this image."""
    assert HEADER_FMT == ">4sBIIIHHQII", (
        f"HEADER_FMT changed to {HEADER_FMT!r}; bump BLOB_VERSION and "
        f"document in CHANGELOG"
    )
    assert HEADER_SIZE == 37

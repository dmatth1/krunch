"""Pure-Python pieces of krunch.inference (no GPU, no model load).

Covers:
  - encode_header / decode_header round-trip + magic / version checks
  - _softmax_clip_normalize numerical contract (encode + decode sides
    must produce identical probs or the AC bitstream isn't byte-exact)
"""
import struct
import numpy as np
import pytest

from krunch.inference import (
    encode_header, decode_header, HEADER_SIZE, HEADER_FMT,
    BLOB_MAGIC, BLOB_VERSION, _softmax_clip_normalize,
)


# ---- Header round-trip -----------------------------------------------------

def test_header_roundtrip_default_args():
    raw = encode_header(original_len=1_048_576, n_chunks=8, crc32=0xCAFEBABE)
    assert len(raw) == HEADER_SIZE
    out = decode_header(raw)
    assert out["original_len"] == 1_048_576
    assert out["n_chunks"] == 8
    assert out["crc32"] == 0xCAFEBABE
    assert out["blob_version"] == BLOB_VERSION
    assert out["adapter_id"] == 0
    assert out["adapter_version"] == 0
    assert out["flags"] == 0


def test_header_roundtrip_with_adapter():
    raw = encode_header(
        original_len=42, n_chunks=1, crc32=0,
        adapter_id=99, adapter_version=2, flags=0x0001,
    )
    out = decode_header(raw)
    assert out["adapter_id"] == 99
    assert out["adapter_version"] == 2
    assert out["flags"] == 0x0001


def test_header_starts_with_magic():
    raw = encode_header(0, 0, 0)
    assert raw[:4] == BLOB_MAGIC


def test_decode_rejects_bad_magic():
    raw = encode_header(0, 0, 0)
    bad = b"NOPE" + raw[4:]
    with pytest.raises(ValueError, match="bad magic"):
        decode_header(bad)


def test_decode_rejects_short_buffer():
    with pytest.raises(ValueError, match="blob too short"):
        decode_header(b"abc")


def test_header_size_matches_format():
    # If HEADER_FMT changes, HEADER_SIZE must update — guards the
    # blob format spec from drifting silently.
    assert HEADER_SIZE == struct.calcsize(HEADER_FMT)


def test_decode_ignores_trailing_bytes():
    raw = encode_header(100, 1, 0)
    # Real callers slice header off the front of a longer blob.
    out = decode_header(raw + b"\x00" * 10_000)
    assert out["original_len"] == 100


# ---- _softmax_clip_normalize ----------------------------------------------

def test_softmax_1d_sums_to_one():
    logits = np.array([1.0, 2.0, 3.0, 0.5], dtype=np.float32)
    probs = _softmax_clip_normalize(logits)
    assert probs.dtype == np.float32
    assert probs.shape == (4,)
    assert pytest.approx(probs.sum(), abs=1e-5) == 1.0


def test_softmax_2d_each_row_sums_to_one():
    logits = np.random.randn(7, 50).astype(np.float32)
    probs = _softmax_clip_normalize(logits)
    assert probs.shape == (7, 50)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_softmax_clips_zero_probabilities_to_floor():
    """The clip step exists because constriction's Categorical rejects
    zero-probability symbols — they're un-decodable in arithmetic coding.
    Ensure the floor is enforced even for a near-degenerate distribution."""
    logits = np.array([1000.0, 0.0, 0.0], dtype=np.float32)
    probs = _softmax_clip_normalize(logits)
    assert (probs > 0).all(), "every token must have positive probability"
    assert probs.min() >= 1e-9 / 2  # rough lower bound after renorm


def test_softmax_encode_decode_sides_match_bitexact():
    """Critical for byte-exact AC roundtrip: encode and decode must compute
    identical probs from identical logits, otherwise the bitstream they
    produce/consume diverges."""
    rng = np.random.default_rng(42)
    logits = rng.standard_normal(1024).astype(np.float32)
    p_encode = _softmax_clip_normalize(logits)
    p_decode = _softmax_clip_normalize(logits.copy())
    np.testing.assert_array_equal(p_encode, p_decode)


def test_softmax_stable_for_extreme_logits():
    """Float32 max ~3.4e38; logits past ~88 overflow exp() if the
    max-subtraction step is missing. Production sees these on confident
    predictions — guard the numerical-stability invariant."""
    logits = np.array([200.0, 100.0, 0.0], dtype=np.float32)
    probs = _softmax_clip_normalize(logits)
    assert np.isfinite(probs).all()
    assert pytest.approx(probs.sum(), abs=1e-5) == 1.0


# ---- ac_encode / ac_decode round-trip --------------------------------------

from krunch.inference import ac_encode, ac_decode


def test_ac_roundtrip_recovers_tokens():
    """The arithmetic coder must be lossless: feed in tokens + their
    distributions, encode, then decode with the same per-step
    distributions, recover the same tokens. This is the core invariant
    the whole codec depends on."""
    rng = np.random.default_rng(0)
    n_tokens, vocab = 64, 256
    logits_seq = rng.standard_normal((n_tokens, vocab)).astype(np.float32)
    # Sample tokens from the distributions (simulating what a real model
    # would have predicted) so the bitstream stays compact.
    tokens = []
    for i in range(n_tokens):
        p = _softmax_clip_normalize(logits_seq[i])
        tokens.append(int(rng.choice(vocab, p=p)))

    bitstream = ac_encode(tokens, logits_seq)
    assert isinstance(bitstream, bytes)

    # Decode using the same per-step logits.
    cursor = {"i": 0}

    def logits_fn(_state, _last_input):
        i = cursor["i"]
        cursor["i"] += 1
        return logits_seq[i], None

    decoded = ac_decode(bitstream, n_tokens, logits_fn)
    assert decoded == tokens


def test_ac_encode_rejects_length_mismatch():
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((10, 256)).astype(np.float32)
    with pytest.raises(AssertionError, match="length mismatch"):
        ac_encode([1, 2, 3], logits)  # 3 tokens vs 10 logits


def test_ac_encode_handles_empty_input():
    """Zero-token edge case — produces an empty (or near-empty) bitstream
    that decode round-trips back to []."""
    logits = np.empty((0, 256), dtype=np.float32)
    bitstream = ac_encode([], logits)
    decoded = ac_decode(bitstream, 0, lambda *_: (None, None))
    assert decoded == []

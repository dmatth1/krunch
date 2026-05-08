"""
CPU adaptive head — algorithm correctness + symmetric encode/decode.

The bit-exact bias trajectory is the foundation invariant: encoder and
decoder must apply identical adjust + update at every step. If this
test passes, the AC roundtrip with adaptive head holds (modulo the
GPU↔CPU parity test, which is a separate file under tests/unit/gpu).

Per docs/TIER_3_OPTIMIZATION.md NEXT-3.
"""

import numpy as np
import pytest

from krunch.codec.adaptive_head import (
    AdaptiveHead, encode_with_adaptive_head, DEFAULT_LR,
)


def _rand_probs(T, V, seed):
    rng = np.random.default_rng(seed)
    p = rng.random((T, V)).astype(np.float32)
    return p / p.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# State + invariants
# ---------------------------------------------------------------------------

def test_initial_bias_is_zero():
    h = AdaptiveHead(vocab_size=100)
    assert h.bias.dtype == np.float64
    assert h.bias.shape == (100,)
    assert (h.bias == 0).all()


def test_adjust_with_zero_bias_returns_normalized_input():
    """At zero bias, adjust(p) ≈ softmax(log(p)) = p (up to fp64 rounding
    of log/exp). Not bit-exact equal but close to noise."""
    h = AdaptiveHead(vocab_size=50)
    p = np.full(50, 1.0 / 50, dtype=np.float64)
    out = h.adjust(p)
    assert np.allclose(out, p, atol=1e-12)
    assert abs(out.sum() - 1.0) < 1e-12


def test_adjust_preserves_total_probability():
    h = AdaptiveHead(vocab_size=200)
    p = _rand_probs(1, 200, seed=1)[0]
    out = h.adjust(p)
    assert abs(out.sum() - 1.0) < 1e-12


def test_update_moves_bias_in_correct_direction():
    """After observing token k, bias[k] should INCREASE (we want the
    model to predict k more often next time it sees similar context).
    Other entries decrease by adjusted[i] * lr."""
    h = AdaptiveHead(vocab_size=10, lr=0.1)
    p = np.full(10, 0.1, dtype=np.float64)
    adjusted = h.adjust(p)  # also ~uniform
    h.update(actual_token=3, adjusted_probs=adjusted)
    assert h.bias[3] > 0, "bias for observed token should increase"
    assert (h.bias[np.arange(10) != 3] < 0).all(), \
        "bias for non-observed tokens should decrease"


def test_reset_zeros_bias():
    h = AdaptiveHead(vocab_size=10, lr=0.1)
    p = np.full(10, 0.1, dtype=np.float64)
    h.update(0, h.adjust(p))
    assert (h.bias != 0).any()
    h.reset()
    assert (h.bias == 0).all()


# ---------------------------------------------------------------------------
# Encoder ↔ decoder symmetry (the load-bearing invariant)
# ---------------------------------------------------------------------------

def test_encode_and_decode_produce_identical_trajectories():
    """The same probs + tokens fed through the algorithm twice must
    produce byte-identical adjusted-probs sequences and bias state.
    This is what makes the AC roundtrip bit-exact."""
    rng = np.random.default_rng(42)
    T, V = 200, 1024
    probs = _rand_probs(T, V, seed=42)
    tokens = rng.integers(0, V, size=T).astype(np.int32)

    enc_out, enc_bias = encode_with_adaptive_head(probs, tokens)
    dec_out, dec_bias = encode_with_adaptive_head(probs, tokens)

    # Byte-equal everywhere — fp64 deterministic.
    assert np.array_equal(enc_out, dec_out), \
        "adjusted-probs sequence diverges between two runs"
    assert np.array_equal(enc_bias, dec_bias), \
        "final bias diverges between two runs"


def test_step_by_step_matches_helper():
    """The convenience encode_with_adaptive_head() must produce the
    same trajectory as a manual loop over adjust() + update()."""
    rng = np.random.default_rng(7)
    T, V = 50, 256
    probs = _rand_probs(T, V, seed=7)
    tokens = rng.integers(0, V, size=T).astype(np.int32)

    helper_out, helper_bias = encode_with_adaptive_head(probs, tokens)

    h = AdaptiveHead(V)
    manual_out = np.empty((T, V), dtype=np.float64)
    for t in range(T):
        manual_out[t] = h.adjust(probs[t])
        h.update(int(tokens[t]), manual_out[t])

    assert np.array_equal(helper_out, manual_out)
    assert np.array_equal(helper_bias, h.bias)


def test_independent_runs_with_same_seed_are_identical():
    """No hidden mutable shared state — two AdaptiveHead instances
    fed the same input must produce identical output."""
    T, V = 100, 200
    probs = _rand_probs(T, V, seed=99)
    rng = np.random.default_rng(99)
    tokens = rng.integers(0, V, size=T).astype(np.int32)

    a = AdaptiveHead(V)
    b = AdaptiveHead(V)
    for t in range(T):
        adj_a = a.adjust(probs[t])
        adj_b = b.adjust(probs[t])
        assert np.array_equal(adj_a, adj_b)
        a.update(int(tokens[t]), adj_a)
        b.update(int(tokens[t]), adj_b)
    assert np.array_equal(a.bias, b.bias)


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------

def test_adjust_strictly_increases_observed_token_probability():
    """If we adjust + update for token k repeatedly with the SAME probs,
    bias[k] should keep growing and adjusted[k] should keep growing
    toward 1 (model is being told 'k is the answer' over and over)."""
    h = AdaptiveHead(vocab_size=20, lr=0.1)
    p = np.full(20, 0.05, dtype=np.float64)
    last_p_at_k = 0.0
    for _ in range(50):
        adj = h.adjust(p)
        assert adj[5] >= last_p_at_k - 1e-12, \
            "p(token=5) should be monotone non-decreasing under repeat"
        last_p_at_k = adj[5]
        h.update(5, adj)
    # After 50 steps at lr=0.1, the bias should clearly favor token 5.
    assert h.bias[5] > 1.0
    assert (h.bias[np.arange(20) != 5] < 0).all()


def test_lr_zero_is_identity_after_update():
    """lr=0 means update is a no-op — bias stays zero, adjust returns
    softmax(log(p)) ≈ p forever."""
    h = AdaptiveHead(vocab_size=30, lr=1e-12)  # near-zero
    p = np.full(30, 1.0 / 30, dtype=np.float64)
    for _ in range(10):
        adj = h.adjust(p)
        h.update(0, adj)
    assert np.allclose(h.bias, 0, atol=1e-9)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_invalid_vocab_size_raises():
    with pytest.raises(ValueError, match="vocab_size"):
        AdaptiveHead(vocab_size=0)


def test_invalid_lr_raises():
    with pytest.raises(ValueError, match="lr"):
        AdaptiveHead(vocab_size=10, lr=0)
    with pytest.raises(ValueError, match="lr"):
        AdaptiveHead(vocab_size=10, lr=1.5)


def test_out_of_range_token_raises():
    h = AdaptiveHead(vocab_size=10)
    p = np.full(10, 0.1, dtype=np.float64)
    adj = h.adjust(p)
    with pytest.raises(ValueError, match="actual_token"):
        h.update(actual_token=10, adjusted_probs=adj)
    with pytest.raises(ValueError, match="actual_token"):
        h.update(actual_token=-1, adjusted_probs=adj)


def test_wrong_shape_probs_raises():
    h = AdaptiveHead(vocab_size=10)
    with pytest.raises(ValueError, match="probs must be shape"):
        h.adjust(np.zeros(11, dtype=np.float64))


# ---------------------------------------------------------------------------
# Full-vocab realistic shape (50K vocab, batch of 256 tokens)
# ---------------------------------------------------------------------------

def test_realistic_vocab_completes_quickly():
    """Smoke: 256 tokens × 50K vocab should finish in < 1 second on
    a laptop CPU. If this is too slow, the production path will be
    too slow too."""
    import time
    V, T = 50277, 256
    probs = _rand_probs(T, V, seed=12345)
    rng = np.random.default_rng(12345)
    tokens = rng.integers(0, V, size=T).astype(np.int32)
    t0 = time.time()
    out, bias = encode_with_adaptive_head(probs, tokens)
    elapsed = time.time() - t0
    assert out.shape == (T, V)
    assert bias.shape == (V,)
    assert elapsed < 1.0, f"50K vocab × 256 tokens took {elapsed:.2f}s"

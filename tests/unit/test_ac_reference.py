"""
CPU reference range coder tests.

This is the spec the CUDA kernel must match byte-for-byte.
"""

import numpy as np
import pytest

from krunch_ac.cdf import probs_to_cdf, T as CDF_T
from krunch_ac.cpu_reference import encode, decode


def _rand_probs(N, V, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.random((N, V)).astype(np.float32)
    p /= p.sum(axis=1, keepdims=True)
    return p


def _roundtrip(probs, symbols):
    cdf = probs_to_cdf(probs)
    bs = encode(cdf, symbols)
    decoded = decode(bs, cdf)
    return bs, decoded


def test_cdf_invariants_random():
    p = _rand_probs(64, 256, seed=1)
    cdf = probs_to_cdf(p)
    assert cdf.shape == (64, 257)
    assert (cdf[:, 0] == 0).all()
    assert (cdf[:, -1] == CDF_T).all()
    assert (np.diff(cdf, axis=1) >= 1).all(), "every bin needs width >= 1"


def test_cdf_invariants_extreme():
    # Near-deterministic (one symbol takes all mass)
    p = np.full((1, 100), 1e-9, dtype=np.float32)
    p[0, 50] = 1.0
    p /= p.sum()
    cdf = probs_to_cdf(p)
    assert cdf[0, -1] == CDF_T
    assert (np.diff(cdf[0]) >= 1).all()


def test_roundtrip_single():
    p = _rand_probs(1, 64, seed=2)
    sym = np.array([17], dtype=np.int32)
    _, decoded = _roundtrip(p, sym)
    assert decoded[0] == 17


def test_roundtrip_short_sequence():
    p = _rand_probs(8, 32, seed=3)
    rng = np.random.default_rng(7)
    sym = rng.integers(0, 32, size=8).astype(np.int32)
    bs, decoded = _roundtrip(p, sym)
    assert (decoded == sym).all(), f"expected {sym}, got {decoded}"
    assert len(bs) > 0


@pytest.mark.parametrize("N,V,seed", [
    (1, 4, 10), (16, 64, 11), (256, 256, 12),
    (1024, 1024, 13), (4096, 50277, 14),
])
def test_roundtrip_varied_shapes(N, V, seed):
    p = _rand_probs(N, V, seed=seed)
    rng = np.random.default_rng(seed + 100)
    # Sample symbols according to the distributions to stay realistic.
    cum = np.cumsum(p, axis=1)
    u = rng.random((N, 1)).astype(np.float32)
    sym = (u < cum).argmax(axis=1).astype(np.int32)
    _, decoded = _roundtrip(p, sym)
    assert (decoded == sym).all()


def test_roundtrip_all_same_symbol():
    # Edge: highly compressible. All tokens are token 0.
    p = _rand_probs(100, 50, seed=20)
    sym = np.zeros(100, dtype=np.int32)
    bs, decoded = _roundtrip(p, sym)
    assert (decoded == sym).all()


def test_roundtrip_low_prob_symbol():
    # Edge: encode a symbol with probability ~ 1/(T-V), the smallest
    # representable. Should still round-trip.
    V = 200
    N = 50
    p = np.full((N, V), 1e-6, dtype=np.float32)
    p[:, 0] = 1.0 - 1e-6 * (V - 1)
    p /= p.sum(axis=1, keepdims=True)
    sym = np.full(N, 199, dtype=np.int32)  # the rare symbol every step
    _, decoded = _roundtrip(p, sym)
    assert (decoded == sym).all()


def test_determinism():
    # Same inputs -> same bitstream byte-for-byte.
    p = _rand_probs(128, 256, seed=42)
    rng = np.random.default_rng(42)
    sym = rng.integers(0, 256, size=128).astype(np.int32)
    cdf = probs_to_cdf(p)
    bs1 = encode(cdf, sym)
    bs2 = encode(cdf, sym)
    assert bs1 == bs2


def test_compression_makes_progress():
    # On a peaky distribution, ratio should be << 1 bit/symbol -> uniform.
    N, V = 2048, 1024
    p = np.full((N, V), 1.0, dtype=np.float32)
    p[:, 0] = 1000.0  # very peaky
    p /= p.sum(axis=1, keepdims=True)
    sym = np.zeros(N, dtype=np.int32)  # always the high-prob symbol
    cdf = probs_to_cdf(p)
    bs = encode(cdf, sym)
    # Uniform encoding: log2(1024) = 10 bits/sym -> 2560 bytes.
    # We should be way under that.
    assert len(bs) < N // 4, f"peaky stream encoded to {len(bs)} bytes, expected << {N // 4}"


def test_50k_vocab_realistic():
    # Realistic LM shape: 50K vocab, 256 tokens/batch.
    N, V = 256, 50277
    rng = np.random.default_rng(99)
    # Long-tailed distribution: top-100 tokens carry most mass.
    base = rng.exponential(scale=0.001, size=(N, V)).astype(np.float32)
    top_idx = rng.integers(0, V, size=(N, 100))
    for i in range(N):
        base[i, top_idx[i]] += rng.exponential(scale=1.0, size=100).astype(np.float32)
    p = base / base.sum(axis=1, keepdims=True)
    # Sample symbols
    cum = np.cumsum(p, axis=1)
    u = rng.random((N, 1)).astype(np.float32)
    sym = (u < cum).argmax(axis=1).astype(np.int32)
    _, decoded = _roundtrip(p, sym)
    assert (decoded == sym).all()

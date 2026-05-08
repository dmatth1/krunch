"""
CPU-only arithmetic tests for compute_decompress_batch.

The function imports torch but the env-var override path doesn't
need CUDA, so we can exercise it without a GPU. The hardware-
detection branch is exercised by the GPU test
`tests/unit/gpu/test_compute_decompress_batch.py`.

Per docs/TIER_3_CLEANUP.md §4.1.
"""

import pytest

from krunch import cpp_path


def test_env_override_explicit_value(monkeypatch):
    """KRUNCH_DECOMPRESS_BATCH should pin B exactly (subject to
    n_chunks clamp)."""
    monkeypatch.setenv("KRUNCH_DECOMPRESS_BATCH", "32")
    assert cpp_path.compute_decompress_batch(n_chunks=100) == 32
    assert cpp_path.compute_decompress_batch(n_chunks=10) == 10  # clamp
    assert cpp_path.compute_decompress_batch(n_chunks=-1) == 32


def test_env_override_clamps_to_at_least_one(monkeypatch):
    """B=0 doesn't make sense; clamp up to 1."""
    monkeypatch.setenv("KRUNCH_DECOMPRESS_BATCH", "0")
    assert cpp_path.compute_decompress_batch(n_chunks=100) == 1


def test_env_override_invalid_falls_through(monkeypatch):
    """Non-int env value falls through to the default (no crash)."""
    monkeypatch.setenv("KRUNCH_DECOMPRESS_BATCH", "not-a-number")
    # Default branch: on no-cuda machines, default=64 clamped by n_chunks.
    out = cpp_path.compute_decompress_batch(n_chunks=10, default=64)
    assert 1 <= out <= 10


def test_n_chunks_clamps_default(monkeypatch):
    """Without env override, on a no-CUDA machine the default branch
    applies. Result must clamp to n_chunks."""
    monkeypatch.delenv("KRUNCH_DECOMPRESS_BATCH", raising=False)
    out = cpp_path.compute_decompress_batch(n_chunks=4, default=64)
    assert out == 4


def test_back_compat_alias(monkeypatch):
    """`pick_decompress_batch` is a thin wrapper used by older
    callers — must produce the same result for the no-n_chunks case."""
    monkeypatch.setenv("KRUNCH_DECOMPRESS_BATCH", "16")
    assert cpp_path.pick_decompress_batch() == cpp_path.compute_decompress_batch(n_chunks=-1)


def test_negative_n_chunks_treated_as_unbounded(monkeypatch):
    """n_chunks <= 0 means 'caller doesn't know yet, give me your
    best guess'. Must not produce zero or negative."""
    monkeypatch.setenv("KRUNCH_DECOMPRESS_BATCH", "8")
    assert cpp_path.compute_decompress_batch(n_chunks=-1) == 8
    assert cpp_path.compute_decompress_batch(n_chunks=0) == 8

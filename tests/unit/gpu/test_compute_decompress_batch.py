"""compute_decompress_batch picks cross-chunk batch B from runtime GPU
specs + n_chunks. Pure dispatch logic — no model needed."""
import os
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import pytest
import torch

from krunch.cpp_path import compute_decompress_batch


def test_env_override_takes_precedence(monkeypatch):
    monkeypatch.setenv("KRUNCH_DECOMPRESS_BATCH", "33")
    assert compute_decompress_batch(n_chunks=100) == 33


def test_env_override_clamped_by_n_chunks(monkeypatch):
    """Env says 64 but only 4 chunks exist — return 4."""
    monkeypatch.setenv("KRUNCH_DECOMPRESS_BATCH", "64")
    assert compute_decompress_batch(n_chunks=4) == 4


def test_env_override_invalid_falls_through(monkeypatch):
    """Garbage env var should not crash — fall through to formula."""
    monkeypatch.setenv("KRUNCH_DECOMPRESS_BATCH", "not-an-int")
    out = compute_decompress_batch(n_chunks=128)
    assert out >= 1


def test_n_chunks_one_returns_one(monkeypatch):
    monkeypatch.delenv("KRUNCH_DECOMPRESS_BATCH", raising=False)
    assert compute_decompress_batch(n_chunks=1) == 1


def test_returns_at_least_one(monkeypatch):
    """Even with no n_chunks hint, must return >= 1."""
    monkeypatch.delenv("KRUNCH_DECOMPRESS_BATCH", raising=False)
    assert compute_decompress_batch() >= 1


def test_formula_scales_with_sm_count(monkeypatch):
    """Formula: cp.async (sm_80+) → n_sms*8; sm_75 → n_sms*2.
    For sm_80+ A10G/L40S/etc., result should comfortably exceed 64
    on a real datacenter GPU."""
    monkeypatch.delenv("KRUNCH_DECOMPRESS_BATCH", raising=False)
    p = torch.cuda.get_device_properties(0)
    out = compute_decompress_batch(n_chunks=10_000)  # n_chunks not the limiter
    if p.major >= 8:
        # n_sms * 8 minus memory cap. Even a tiny GPU should hit > 32.
        assert out >= 32, f"sm_{p.major}{p.minor}: B={out} too low"
    else:
        # sm_75 (T4 = 40 SMs) → ~80 expected.
        assert 1 <= out <= p.multi_processor_count * 4

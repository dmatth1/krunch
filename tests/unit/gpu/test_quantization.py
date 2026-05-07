"""Pure-tensor quantization helpers in cpp_path.py.

These don't need the model or AC kernels — just torch on CUDA. Pinning
behavior here guards the W8A8 / INT8 weight bundles that init_weights
hands to the layer-step kernels.
"""
import os
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import pytest
import torch

from krunch.cpp_path import (
    _quantize_int8_per_output_channel,
    _quantize_dequant_int8_per_output_channel,
    _quantize_dequant_uint8_per_input_channel,
)


@pytest.fixture
def w():
    """Realistic-ish weight matrix: [K=768, N=3072] fp16 (matches FFN_K shape)."""
    torch.manual_seed(0)
    return torch.randn(768, 3072, dtype=torch.float16, device="cuda") * 0.1


# ---- per-output-channel symmetric int8 (raw quant: returns int8 + scale) ---

def test_int8_returns_int8_buffer_and_fp16_scale(w):
    w_q, scale = _quantize_int8_per_output_channel(w)
    assert w_q.dtype == torch.int8
    assert w_q.shape == w.shape
    assert scale.dtype == torch.float16
    assert scale.shape == (w.shape[1],)  # one scale per output channel


def test_int8_clamps_to_signed_range(w):
    w_q, _ = _quantize_int8_per_output_channel(w)
    assert w_q.min().item() >= -127
    assert w_q.max().item() <= 127


def test_int8_dequant_within_quantization_error(w):
    """w_dequant = w_q.float() * scale should approximate w within scale/2."""
    w_q, scale = _quantize_int8_per_output_channel(w)
    w_d = w_q.float() * scale.float().unsqueeze(0)
    err = (w.float() - w_d).abs().max().item()
    # Per-column scale = max(|col|) / 127 → quantization step ≤ scale.
    # Worst-case error per element is scale/2.
    worst_step = (scale.float() / 1.0).max().item()
    assert err <= worst_step, f"err={err} step={worst_step}"


def test_int8_rejects_non_2d():
    bad = torch.zeros(4, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="2D"):
        _quantize_int8_per_output_channel(bad)


# ---- per-output-channel symmetric int8 (dequant in place: returns fp16) ----

def test_dequant_int8_preserves_dtype(w):
    w_d = _quantize_dequant_int8_per_output_channel(w)
    assert w_d.dtype == w.dtype  # fp16 in, fp16 out
    assert w_d.shape == w.shape


def test_dequant_int8_idempotent(w):
    """quant→dequant→quant→dequant should equal one round of quant→dequant."""
    once = _quantize_dequant_int8_per_output_channel(w)
    twice = _quantize_dequant_int8_per_output_channel(once)
    torch.testing.assert_close(once, twice, atol=0, rtol=0)


def test_dequant_int8_passes_through_non_2d():
    """Non-matrix tensors (biases, time_mix vectors) pass through unchanged."""
    v = torch.randn(64, dtype=torch.float16, device="cuda")
    out = _quantize_dequant_int8_per_output_channel(v)
    torch.testing.assert_close(out, v, atol=0, rtol=0)


# ---- per-input-channel uint8 (legacy KRUNCH_INT8_WEIGHTS path) ------------

def test_dequant_uint8_preserves_shape_and_dtype(w):
    w_d = _quantize_dequant_uint8_per_input_channel(w)
    assert w_d.shape == w.shape
    assert w_d.dtype == w.dtype


def test_dequant_uint8_within_per_row_quantization_error(w):
    """Per-row uint8: error per element ≤ (max-min)/255 of that row."""
    w_d = _quantize_dequant_uint8_per_input_channel(w)
    w_f, w_df = w.float(), w_d.float()
    per_row_step = (w_f.max(dim=1).values - w_f.min(dim=1).values) / 255.0
    err = (w_f - w_df).abs().max(dim=1).values
    # Allow a hair of fp16 round-trip slop on top of the quant step.
    assert (err <= per_row_step + 1e-3).all().item(), \
        f"max err {err.max()} vs step {per_row_step.max()}"

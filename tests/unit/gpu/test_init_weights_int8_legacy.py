"""init_weights with KRUNCH_INT8_WEIGHTS=1 — legacy per-input-channel
uint8 quantization path (predates W8A8). Quantizes layer matmul
weights AND emb + head, then dequantizes back to fp16 in place.

This path hasn't been exercised in production for a while; the test
just pins the dispatch shape so accidental refactors can't silently
remove it. Mutually exclusive with KRUNCH_INT8_W8A8.
"""
import os
import pytest
import torch

from krunch import cpp_path


def _init_with_env(model, **env):
    cpp_path._WEIGHTS_CACHE.clear()
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return cpp_path.init_weights(model, "cuda")


def test_int8_weights_returns_fp16_dtype(model):
    """All layer matmul weights stay fp16 dtype after dequant — the
    quantization is a quality-degraded pass through, not a dtype change."""
    weights = _init_with_env(model, KRUNCH_INT8_WEIGHTS="1",
                             KRUNCH_INT8_W8A8="0", KRUNCH_BF16="0")
    layer0 = weights["layers"][0]
    # Every entry that's a 2D matrix (matmul weight) should be fp16.
    matmul_count = 0
    for t in layer0:
        if t.dim() == 2:
            assert t.dtype == torch.float16, \
                f"matmul weight got dtype {t.dtype}, expected fp16"
            matmul_count += 1
    assert matmul_count == 7, f"expected 7 matmul tensors, got {matmul_count}"


def test_int8_weights_actually_quantized(model):
    """Quantize-dequant must lose precision. Compare to a fresh
    no-quant bundle: the matmul weights should differ."""
    cpp_path._WEIGHTS_CACHE.clear()
    plain = _init_with_env(model, KRUNCH_INT8_WEIGHTS="0",
                           KRUNCH_INT8_W8A8="0", KRUNCH_BF16="0")
    cpp_path._WEIGHTS_CACHE.clear()
    quant = _init_with_env(model, KRUNCH_INT8_WEIGHTS="1",
                           KRUNCH_INT8_W8A8="0", KRUNCH_BF16="0")
    plain_kw = plain["layers"][0][7]
    quant_kw = quant["layers"][0][7]
    diff = (plain_kw.float() - quant_kw.float()).abs().max().item()
    assert diff > 0, "INT8 quant did not change the weights"


def test_int8_w8a8_mutually_exclusive(model):
    cpp_path._WEIGHTS_CACHE.clear()
    os.environ["KRUNCH_INT8_WEIGHTS"] = "1"
    os.environ["KRUNCH_INT8_W8A8"] = "1"
    try:
        with pytest.raises(ValueError, match="at most one"):
            cpp_path.init_weights(model, "cuda")
    finally:
        os.environ.pop("KRUNCH_INT8_WEIGHTS", None)
        os.environ.pop("KRUNCH_INT8_W8A8", None)

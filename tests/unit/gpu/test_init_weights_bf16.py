"""init_weights with KRUNCH_BF16=1 — alternative weight-precision
path. Routes the 7 layer matmul weights (Kw/Vw/Rw/Ow + ffn_Kw/Vw/Rw)
through the bf16 cp.async WMMA kernel instead of fp16.

Bytes diverge from the fp16 codec → encoder + decoder must both set
KRUNCH_BF16=1 for AC roundtrip to hold (handled at v2 model_id
territory). This test only covers the dispatch shape — the actual
roundtrip-with-bf16 test is end-to-end and lives in the integration
suite.
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


def _bf16_count(layer):
    return sum(1 for t in layer if t.dtype == torch.bfloat16)


def test_bf16_routes_layer_matmul_weights_to_bfloat16(model):
    """Layer matmul weights (att Kw/Vw/Rw/Ow + ffn Kw/Vw/Rw = 7) should
    convert to bf16; non-matmul weights (ln, time_mix, time_decay,
    time_first) keep their original dtype."""
    weights = _init_with_env(model, KRUNCH_BF16="1",
                             KRUNCH_INT8_W8A8="0", KRUNCH_INT8_WEIGHTS="0")
    layer0 = weights["layers"][0]
    assert _bf16_count(layer0) == 7, \
        f"expected 7 bf16 matmul weights, got {_bf16_count(layer0)}"


def test_bf16_off_keeps_fp16(model):
    """Default (no env): no bf16 anywhere in the layer bundle."""
    weights = _init_with_env(model, KRUNCH_BF16="0",
                             KRUNCH_INT8_W8A8="0", KRUNCH_INT8_WEIGHTS="0")
    layer0 = weights["layers"][0]
    assert _bf16_count(layer0) == 0


def test_bf16_cache_key_separate_from_fp16(model):
    """Toggling KRUNCH_BF16 must miss the cache so fresh weights are
    built — otherwise a prior fp16 bundle would be returned."""
    cpp_path._WEIGHTS_CACHE.clear()
    fp16_w = _init_with_env(model, KRUNCH_BF16="0",
                            KRUNCH_INT8_W8A8="0", KRUNCH_INT8_WEIGHTS="0")
    bf16_w = _init_with_env(model, KRUNCH_BF16="1",
                            KRUNCH_INT8_W8A8="0", KRUNCH_INT8_WEIGHTS="0")
    # Different cache entries → different bundle objects.
    assert fp16_w is not bf16_w
    # Spot-check dtypes really differ in the matmul band.
    assert _bf16_count(fp16_w["layers"][0]) == 0
    assert _bf16_count(bf16_w["layers"][0]) == 7

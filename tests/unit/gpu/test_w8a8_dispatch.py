"""W8A8 weight bundle dispatch — pin the keys init_weights returns
based on the KRUNCH_INT8_W8A8 env knob.

Pure dispatch logic test: doesn't run a forward pass. The point is
that init_weights:
  - populates `w8a8_int8` and `w8a8_scale` arrays when KRUNCH_INT8_W8A8=1
  - leaves them absent when KRUNCH_INT8_W8A8=0 (fp16 path only)

This catches refactor regressions in the W8A8 dispatch ladder
(`forward_packed_window` → W8A8 path at M >= 256, fp16 fallback otherwise).
"""
import os
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import pytest
import torch
from krunch.inference import _load_rwkv, MODEL_PATH
from krunch import cpp_path


@pytest.fixture(scope="module")
def model():
    RWKV = _load_rwkv()
    return RWKV(model=str(MODEL_PATH).removesuffix(".pth"),
                strategy="cuda fp16", verbose=False)


def _init_with_env(model, **env):
    """init_weights' cache key includes the W8A8 env vars; toggle them
    in the test, clear the cache, then re-init."""
    cpp_path._WEIGHTS_CACHE.clear()
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return cpp_path.init_weights(model, "cuda")


def test_w8a8_on_populates_int8_arrays(model):
    weights = _init_with_env(model, KRUNCH_INT8_W8A8="1",
                             KRUNCH_INT8_WEIGHTS="0", KRUNCH_BF16="0")
    assert "w8a8_int8" in weights, "W8A8 on: int8 weight bundle missing"
    assert "w8a8_scale" in weights, "W8A8 on: scale bundle missing"
    # 12 layers × 7 matmul weights = 84 entries; structure check.
    assert len(weights["w8a8_int8"]) == 12, "expected 12 layer entries"
    layer0 = weights["w8a8_int8"][0]
    assert len(layer0) == 7, "expected 7 W8A8 matmul weights per layer"
    assert all(t.dtype == torch.int8 for t in layer0), \
        "W8A8 weights must be int8"


def test_w8a8_off_omits_int8_arrays(model):
    weights = _init_with_env(model, KRUNCH_INT8_W8A8="0",
                             KRUNCH_INT8_WEIGHTS="0", KRUNCH_BF16="0")
    assert "w8a8_int8" not in weights, (
        "W8A8 off: int8 weight bundle should be absent (fp16 path only)"
    )
    assert "w8a8_scale" not in weights, "W8A8 off: scale bundle should be absent"


def test_w8a8_int8_weights_mutually_exclusive(model):
    """Setting both KRUNCH_INT8_WEIGHTS=1 and KRUNCH_INT8_W8A8=1 must
    raise — the schemes are incompatible (different quant recipes)."""
    cpp_path._WEIGHTS_CACHE.clear()
    os.environ["KRUNCH_INT8_WEIGHTS"] = "1"
    os.environ["KRUNCH_INT8_W8A8"] = "1"
    try:
        with pytest.raises(ValueError, match="at most one"):
            cpp_path.init_weights(model, "cuda")
    finally:
        os.environ.pop("KRUNCH_INT8_WEIGHTS", None)
        os.environ.pop("KRUNCH_INT8_W8A8", None)

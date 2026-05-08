"""
Encoder/decoder symmetry across every codec flavor — the W8A8 bug
prevention test.

The Bugs.md "W8A8 multi-chunk CLI roundtrip break" failure was an
asymmetry between which kernel the encoder dispatched
(`forward_packed_window`) and which the decoder dispatched
(`forward_stepped_batched`) on identical inputs. The fix was to
remove the M>=256 gate so both sides unconditionally route through
the same W8A8 kernel.

This test pins that invariant directly: for every codec flavor we
ship (fp16 baseline, W8A8 default, future bf16), the logits emitted
by `forward_packed_window` over a sequence MUST byte-equal the
logits emitted by `forward_stepped_batched` (B=1) walked one token
at a time over the same sequence.

If a future codec change re-introduces an asymmetric guard, this
test fires before the bug ships, instead of weeks later on a 10 MB
A10G run with `CRC32 mismatch`.

Per docs/TIER_3_CLEANUP.md §4.2.
"""
import os
import pytest
import torch

from krunch import cpp_path
from krunch.inference import BOS_TOKEN


# 16 tokens — long enough to exercise multi-step state evolution and
# to span both >0 and <SEQ_BATCH window paths in
# forward_packed_window. Short enough to be cheap.
TOKENS = [BOS_TOKEN, 510, 3158, 8516, 30013, 27287, 689, 253,
          12, 1820, 99, 4096, 7777, 32100, 8, 211]


def _packed_logits(weights, tokens):
    """Run forward_packed_window over the full sequence in one call.
    Mirrors what the encoder does on the production compress path
    inside `inference._compress_chunk_cpp`."""
    state = cpp_path.fresh_state(weights)
    input_t = torch.as_tensor(tokens, dtype=torch.long, device="cuda")
    return cpp_path.forward_packed_window(weights, input_t, state,
                                          off=0, n=len(tokens))


def _stepped_batched_logits(weights, tokens):
    """Run forward_stepped_batched (B=1) one token per step. Mirrors
    what the decoder does inside `_decompress_chunks_batched_cpp`
    (the production multi-chunk decompress path).

    Returns a [T, V] tensor of fp16 logits to match packed's shape.
    """
    B = 1
    state = cpp_path.fresh_state_batched(weights, B)
    out_per_step = []
    for tok in tokens:
        cur_in = torch.as_tensor([tok], dtype=torch.long, device="cuda")
        logits = cpp_path.forward_stepped_batched(weights, cur_in, state)
        out_per_step.append(logits.squeeze(0).clone())
    return torch.stack(out_per_step, dim=0)


def _flush_caches():
    """Drop module-level caches so a fresh init_weights is built per
    parametrized run (each codec flavor gets its own bundle)."""
    for cache_name in (
        "_WEIGHTS_CACHE", "_STEPPED_BUFS_CACHE", "_GRAPH_CACHE",
        "_STEPPED_BATCHED_BUFS_CACHE", "_GRAPH_BATCHED_CACHE",
        "_FULL_STEP_BUFS_CACHE", "_FULL_STEP_GRAPH_CACHE",
        "_BATCHED_STATE_CACHE",
    ):
        cache = getattr(cpp_path, cache_name, None)
        if cache is not None:
            cache.clear()


# fp16 + W8A8 are the production-path flavors. Bf16 is opt-in (NEXT-2);
# encoder + decoder must both use it. Per `init_weights`, bf16 is
# mutually exclusive with W8A8 (so explicitly set W8A8=0 here).
_FLAVORS = [
    pytest.param(
        "fp16",
        {"KRUNCH_INT8_W8A8": "0", "KRUNCH_INT8_WEIGHTS": "0", "KRUNCH_BF16": "0"},
        id="fp16",
    ),
    pytest.param(
        "w8a8",
        {"KRUNCH_INT8_W8A8": "1", "KRUNCH_INT8_WEIGHTS": "0", "KRUNCH_BF16": "0"},
        id="w8a8",
    ),
    pytest.param(
        "bf16",
        {"KRUNCH_INT8_W8A8": "0", "KRUNCH_INT8_WEIGHTS": "0", "KRUNCH_BF16": "1"},
        id="bf16",
    ),
]


@pytest.mark.parametrize("flavor,env_overrides", _FLAVORS)
def test_packed_window_matches_stepped_batched(engine, flavor, env_overrides,
                                               monkeypatch):
    """Every codec flavor: forward_packed_window logits MUST byte-equal
    forward_stepped_batched logits over the same input. This is the
    invariant the W8A8 bug violated."""
    for k, v in env_overrides.items():
        monkeypatch.setenv(k, v)
    _flush_caches()

    weights = cpp_path.init_weights(engine._model, "cuda")

    # If the codec flavor isn't actually present in the bundle (e.g.
    # W8A8 weights weren't pre-quantized), skip rather than silently
    # validating a different path than the test name claims.
    if flavor == "w8a8" and "w8a8_int8" not in weights:
        pytest.skip("W8A8 weights not in bundle; init_weights produced fp16")
    if flavor == "bf16":
        import torch
        # Pick any layer matmul weight (Kw at layers[i][7]) and check dtype.
        layer_kw_dtype = weights["layers"][0][7].dtype
        if layer_kw_dtype != torch.bfloat16:
            pytest.skip(
                f"Bf16 weights not in bundle (got {layer_kw_dtype}); "
                f"init_weights ignored KRUNCH_BF16=1"
            )

    packed = _packed_logits(weights, TOKENS).to(torch.float32)
    stepped = _stepped_batched_logits(weights, TOKENS).to(torch.float32)

    assert packed.shape == stepped.shape, (
        f"{flavor}: shape mismatch packed={tuple(packed.shape)} "
        f"vs stepped_batched={tuple(stepped.shape)}"
    )

    # Bit-equal is the strict invariant. Allow no tolerance — the AC
    # path's CDF construction will round 1-LSB differences to
    # different integers and break roundtrip.
    if not torch.equal(packed, stepped):
        diff = (packed - stepped).abs()
        max_abs = float(diff.max())
        n_diff = int((diff > 0).any(dim=-1).sum())
        max_token = int(diff.amax(dim=-1).argmax())
        pytest.fail(
            f"{flavor}: packed != stepped_batched. "
            f"max_abs={max_abs:.3e} mismatched_steps={n_diff}/{len(TOKENS)} "
            f"first_bad_step={max_token}. "
            f"This is the failure mode that caused the W8A8 multi-chunk "
            f"roundtrip break — see docs/Bugs.md fixed entries."
        )


def test_packed_partial_window_matches_stepped(engine, monkeypatch):
    """Tail-of-chunk regression: a partial window (n < SEQ_BATCH) must
    take the same kernel path as a full one. The Bugs.md W8A8 bug
    manifested specifically because the gate fell back to fp16 on
    partial windows while the decoder stayed on W8A8."""
    monkeypatch.setenv("KRUNCH_INT8_W8A8", "1")
    _flush_caches()

    weights = cpp_path.init_weights(engine._model, "cuda")
    if "w8a8_int8" not in weights:
        pytest.skip("W8A8 weights not in bundle")

    # 3 tokens — well below any plausible threshold gate.
    tail_tokens = TOKENS[:3]
    packed = _packed_logits(weights, tail_tokens).to(torch.float32)
    stepped = _stepped_batched_logits(weights, tail_tokens).to(torch.float32)
    assert torch.equal(packed, stepped), (
        "W8A8 partial window (n=3) diverged from stepped_batched — "
        "kernel-asymmetry regression. Check forward_packed_window "
        "for any newly-added gate that doesn't have a matching "
        "guard on the decoder side. (docs/Bugs.md fixed entries)"
    )

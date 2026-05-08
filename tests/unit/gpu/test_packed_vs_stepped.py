"""Encoder packed vs decoder stepped — the foundation invariant
that AC bit-exact roundtrip depends on.

The encoder runs `forward_packed_window` over an input slice and
emits logits[off..off+n]. The decoder runs `forward_stepped` one
token at a time and emits logits[t] per call. For AC roundtrip to
hold, logits[t] from packed must equal logits[t] from stepped, byte-
for-byte, for every t.

If this invariant breaks, the AC encoder writes bits against one CDF
and the decoder reads against a different CDF — output bytes still
look plausible but don't recover the input. That's exactly the
failure mode in Bugs.md #1 and #4.

Force W8A8 OFF here: forward_packed_window with W8A8 calls
rwkv4_layer_step_cpp_w8a8, while forward_stepped calls
rwkv4_layer_step_cpp_t1 (fp16 T=1 specialized) — different kernels by
design, not bit-equal. Production AC roundtrip holds because cli.py's
single-chunk decode uses forward_stepped (fp16) and the chunks-batched
decode uses forward_stepped_batched (W8A8) — each path is internally
symmetric. This test pins the fp16 invariant of forward_packed_window
itself, not cross-kernel equality.
"""
import os
os.environ["KRUNCH_INT8_W8A8"] = "0"

import pytest
import torch

from krunch import cpp_path
from krunch.inference import BOS_TOKEN


# Short token sequence — enough to exercise multi-step state evolution
# without paying full T-cost of a real chunk.
TOKENS = [BOS_TOKEN, 510, 3158, 8516, 30013, 27287, 689, 253]


def test_packed_first_token_matches_stepped(weights):
    """Packed forward at off=0, n=1 must produce the same logits as
    a single forward_stepped call from fresh state."""
    state_p = cpp_path.fresh_state(weights)
    state_s = cpp_path.fresh_state(weights)
    input_t = torch.as_tensor(TOKENS, dtype=torch.long, device="cuda")
    packed = cpp_path.forward_packed_window(weights, input_t, state_p, off=0, n=1)
    stepped = cpp_path.forward_stepped(weights, TOKENS[0], state_s)
    assert torch.equal(packed[0], stepped), \
        "packed[0] must equal stepped[BOS] (encoder/decoder asymmetry)"


def test_packed_full_window_matches_stepped_sequence(weights):
    """For T tokens fed one-at-a-time stepped, the per-step logits
    must equal the same row of a single packed forward over [0, T).
    Pin this for T up to 8."""
    T = len(TOKENS)
    state_p = cpp_path.fresh_state(weights)
    state_s = cpp_path.fresh_state(weights)
    input_t = torch.as_tensor(TOKENS, dtype=torch.long, device="cuda")

    packed = cpp_path.forward_packed_window(weights, input_t, state_p, off=0, n=T)
    for t in range(T):
        stepped = cpp_path.forward_stepped(weights, TOKENS[t], state_s)
        assert torch.equal(packed[t], stepped), \
            f"packed[{t}] != stepped[{TOKENS[t]}] — AC roundtrip would break here"


def test_forward_packed_wrapper_matches_window(weights):
    """`forward_packed(weights, ids, state)` is a thin wrapper around
    `forward_packed_window(weights, ids, state, off=0, n=T)`. Confirm
    the wrapper produces bit-identical output to the underlying call."""
    state_a = cpp_path.fresh_state(weights)
    state_b = cpp_path.fresh_state(weights)
    input_t = torch.as_tensor(TOKENS, dtype=torch.long, device="cuda")
    a = cpp_path.forward_packed(weights, input_t, state_a)
    b = cpp_path.forward_packed_window(weights, input_t, state_b, off=0,
                                        n=len(TOKENS))
    assert torch.equal(a, b), "forward_packed diverged from forward_packed_window"


def test_packed_window_offset_matches_stepped_resume(weights):
    """forward_packed_window(off=k) on existing state must equal
    forward_stepped through tokens[k:] from the same state. This is
    what compress_chunk relies on when it iterates SEQ_BATCH-sized
    windows: state carries forward, each window must produce the
    same logits as if the whole chunk had been packed in one shot."""
    T = len(TOKENS)
    split = T // 2

    # Stepped: feed all T tokens one at a time.
    state_s = cpp_path.fresh_state(weights)
    stepped_logits = []
    for tok in TOKENS:
        stepped_logits.append(
            cpp_path.forward_stepped(weights, tok, state_s).clone()
        )

    # Packed: window 1 covers [0, split), state carries to window 2
    # which covers [split, T). The two windows together must reproduce
    # the same logits as the stepped sequence.
    state_p = cpp_path.fresh_state(weights)
    input_t = torch.as_tensor(TOKENS, dtype=torch.long, device="cuda")
    w1 = cpp_path.forward_packed_window(weights, input_t, state_p, 0, split)
    w2 = cpp_path.forward_packed_window(weights, input_t, state_p, split, T - split)
    packed_logits = list(w1) + list(w2)

    for t in range(T):
        assert torch.equal(packed_logits[t], stepped_logits[t]), \
            f"window-resumed packed[{t}] != stepped[{t}]"

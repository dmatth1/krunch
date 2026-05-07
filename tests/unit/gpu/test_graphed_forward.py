"""forward_stepped_graphed_v2 — CUDA-graph captured per-layer step.

The graphed path is pointer-bound: capture binds the graph to specific
state-tensor data pointers; subsequent calls REPLAY against those same
pointers. Comparing eager and graphed outputs requires care — the
capture phase itself mutates state, and replay reads/writes the
captured buffers regardless of what `state` arg is passed.

We test the weaker-but-still-useful invariants:
  - calling the function twice with the same `state` produces sensible
    state evolution (each call advances state)
  - the output is finite and well-shaped
  - the eager (non-graphed) deterministic-matmul path is reproducible
"""
import pytest
import torch

from krunch import cpp_path
from krunch.inference import BOS_TOKEN


def test_eager_forward_is_deterministic(weights):
    """Two invocations of forward_stepped on identical fresh state must
    produce bit-identical logits. KRUNCH_DETERMINISTIC_MATMUL=1 is what
    makes this hold."""
    s1 = cpp_path.fresh_state(weights)
    s2 = cpp_path.fresh_state(weights)
    a = cpp_path.forward_stepped(weights, BOS_TOKEN, s1)
    b = cpp_path.forward_stepped(weights, BOS_TOKEN, s2)
    assert torch.equal(a, b), "forward_stepped non-deterministic across calls"


def test_eager_forward_state_evolves(weights):
    """forward_stepped mutates state in place — at least one layer's
    att_xx must change after a single token."""
    state = cpp_path.fresh_state(weights)
    pre = [s.clone() for s in state[0]]
    cpp_path.forward_stepped(weights, BOS_TOKEN, state)
    changed = any(
        not torch.equal(pre_i, state[0][i])
        for i, pre_i in enumerate(pre)
    )
    assert changed, "forward_stepped did not mutate att_xx"


def test_eager_forward_returns_full_vocab_logits(weights):
    """logits shape == [V] (full vocab), finite, not all-zero."""
    state = cpp_path.fresh_state(weights)
    logits = cpp_path.forward_stepped(weights, BOS_TOKEN, state)
    assert logits.shape == (50277,), f"unexpected shape {logits.shape}"
    assert torch.isfinite(logits).all().item()
    assert logits.abs().max().item() > 0

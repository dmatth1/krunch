"""fresh_state, fresh_state_batched, fresh_state_batched_cached — RWKV
state initialization shapes/dtypes/identity invariants."""
import os
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import pytest
import torch

from krunch import cpp_path


N_LAYER = 12

# `weights` fixture (session-scoped) is provided by conftest.py.


# ---- fresh_state (B=1) ----------------------------------------------------

def test_fresh_state_returns_5_tuple_per_layer(weights):
    state = cpp_path.fresh_state(weights)
    assert len(state) == 5  # (att_xx, aa, bb, pp, ffn_xx)
    for arr in state:
        assert len(arr) == N_LAYER


def test_fresh_state_shapes_and_dtypes(weights):
    """Single-stream state tensors carry a leading 1-dim so the layer
    step kernels can use the same broadcast logic as batched callers."""
    state = cpp_path.fresh_state(weights)
    n_embd = weights["n_embd"]
    n_att = weights["n_att"]
    for i in range(N_LAYER):
        assert state[0][i].shape == (1, n_embd) and state[0][i].dtype == torch.float16
        assert state[1][i].shape == (1, n_att)  and state[1][i].dtype == torch.float32
        assert state[2][i].shape == (1, n_att)  and state[2][i].dtype == torch.float32
        assert state[3][i].shape == (1, n_att)  and state[3][i].dtype == torch.float32
        assert state[4][i].shape == (1, n_embd) and state[4][i].dtype == torch.float16


def test_fresh_state_pp_initialized_to_neg_inf(weights):
    """RWKV-4 attention initial pp must be -1e30 (functions as -inf for
    the running-max-softmax in WKV)."""
    state = cpp_path.fresh_state(weights)
    for i in range(N_LAYER):
        assert (state[3][i] == -1e30).all().item()


def test_fresh_state_other_buffers_zero(weights):
    """All non-pp state buffers must start zeroed."""
    state = cpp_path.fresh_state(weights)
    for slot_idx in (0, 1, 2, 4):
        for i in range(N_LAYER):
            assert (state[slot_idx][i] == 0).all().item()


# ---- fresh_state_batched (B>=1) -------------------------------------------

def test_fresh_state_batched_shapes(weights):
    B = 7
    n_embd = weights["n_embd"]
    n_att = weights["n_att"]
    state = cpp_path.fresh_state_batched(weights, B)
    for i in range(N_LAYER):
        assert state[0][i].shape == (B, n_embd)
        assert state[1][i].shape == (B, n_att)
        assert state[3][i].shape == (B, n_att)
        assert (state[3][i] == -1e30).all().item()


# ---- fresh_state_batched_cached (graph capture) ---------------------------

def test_cached_state_returns_same_tensor_identity(weights):
    """Graph capture binds tensor data pointers; the cached fetch must
    return the SAME tensor objects across calls (not just equal-valued)."""
    B = 4
    s1 = cpp_path.fresh_state_batched_cached(weights, B)
    s2 = cpp_path.fresh_state_batched_cached(weights, B)
    for i in range(N_LAYER):
        assert s1[0][i].data_ptr() == s2[0][i].data_ptr(), \
            "cached state changed identity — would dangle captured graphs"


def test_cached_state_independent_per_B(weights):
    """Different B values get independent cache entries."""
    s2 = cpp_path.fresh_state_batched_cached(weights, 2)
    s4 = cpp_path.fresh_state_batched_cached(weights, 4)
    assert s2[0][0].data_ptr() != s4[0][0].data_ptr()
    assert s2[0][0].shape[0] == 2
    assert s4[0][0].shape[0] == 4

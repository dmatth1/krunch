"""MLX port of RWKV-v4 + HiRA (L3TC-200K architecture).

Task 25. Paired with Phase 14 specialist training plan. MLX auto-fuses
compute graphs on Apple Silicon so the T-loop in WKV runs as a single
Metal dispatch instead of T=2048 separate ops (the task-9 bottleneck).

**What's mirrored from `vendor/L3TC/models/RWKV_V4/rwkv_tc_hira_train.py`:**

- RWKV_TimeMix: time_decay, time_first, time_mix_{k,v,r,g}, WKV
  recurrence, HiRA low-rank additions on key/value/receptance, output
  projection.
- RWKV_ChannelMix: time_mix_{k,r}, square-ReLU FFN with HiRA on
  key/value/receptance.
- Block: ln0 on first block, short residual, ln1+att+ln2+ffn residuals.
- Top level: emb, blocks, ln_out, head. No weight tying.

**Differences from PyTorch reference:**

- `time_shift` is a raw prepend-zeros-shift-right-by-1 instead of
  nn.ZeroPad2d (MLX handles this with mx.pad directly).
- `time_mix_g` parameter is defined but unused in Phase 11 training
  (only xg is computed, never consumed). Kept for weight-loading
  compatibility.
- WKV is expressed as a pure-MLX recurrence using the same numerically-
  stable formulation as `_wkv_cpu_forward` in train_l3tc_phase11.py.
  MLX's graph capture + mx.compile fuses this into a single dispatch.

**Parameter name mapping (PyTorch state_dict -> MLX):** direct 1:1 by
name. `convert_pt_to_mlx` handles the transposition of Linear weights
(PyTorch stores (out, in); MLX's nn.Linear uses (out, in) too — no
transpose needed). Verified in numerical_parity_check.py.

Usage:
    from scripts.mlx_rwkv import RwkvTcHira
    model = RwkvTcHira(vocab_size=16384, hidden_size=96,
                      num_hidden_layers=2, intermediate_size=96,
                      rwkv_rank=4, ctx_len=2048)
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

# Custom-backward WKV. Task 25: replaces the auto-diff WKV whose backward
# tape unrolled per-timestep (~9K tok/s full training step). The
# `custom_function` primitive prevents MLX from auto-differentiating
# through our T-loop and uses the hand-derived backward in `mlx_wkv.py`
# instead. Math is identical (both ported from wkv_cuda.cu).
try:
    from mlx_wkv import wkv as _wkv_custom
except ImportError:
    _wkv_custom = None


def time_shift(x: mx.array) -> mx.array:
    """Shift [B, T, C] by one timestep along T, prepending zeros.

    Equivalent to nn.ZeroPad2d((0,0,1,-1)) in the PyTorch reference.
    """
    zeros = mx.zeros_like(x[:, :1, :])
    return mx.concatenate([zeros, x[:, :-1, :]], axis=1)


def wkv_forward(time_decay: mx.array, time_first: mx.array,
                k: mx.array, v: mx.array) -> mx.array:
    """WKV recurrence. Numerically-stable form matching the CUDA kernel.

    time_decay: (C,), trainable log-decay (real negative decay is exp(time_decay))
    time_first: (C,), trainable
    k, v:       (B, T, C)
    returns:    (B, T, C)

    The math matches `_wkv_cpu_forward` from train_l3tc_phase11.py and
    the RWKV-v4 CUDA kernel. Invariant: at each timestep t, we maintain
    running (aa, bb, pp) state and emit out[t] = aa_plus_bonus / bb_plus_bonus
    where the bonus is time_first-weighted current (k,v) contribution.
    """
    # MLX is lazy; concretize dtypes/shape before entering the loop.
    w = -mx.exp(time_decay)  # (C,) — always negative (decay strength)
    u = time_first           # (C,)
    B, T, C = k.shape

    # Running state starts at (0, 0, -inf). Using -1e38 for numerical
    # reasons: real -inf would NaN the first subtraction in e1.
    aa = mx.zeros((B, C), dtype=k.dtype)
    bb = mx.zeros((B, C), dtype=k.dtype)
    pp = mx.full((B, C), -1e38, dtype=k.dtype)

    outs = []
    for t in range(T):
        kk = k[:, t]           # (B, C)
        vv = v[:, t]           # (B, C)
        # "Current-step bonus" — uses time_first as the current weight.
        ww = u + kk
        p = mx.maximum(pp, ww)
        e1 = mx.exp(pp - p)
        e2 = mx.exp(ww - p)
        a = e1 * aa + e2 * vv
        b = e1 * bb + e2
        outs.append(a / b)
        # Update running state for next timestep — uses time_decay.
        ww2 = pp + w
        p2 = mx.maximum(ww2, kk)
        e1b = mx.exp(ww2 - p2)
        e2b = mx.exp(kk - p2)
        aa = e1b * aa + e2b * vv
        bb = e1b * bb + e2b
        pp = p2
    # mx.stack over the T dimension. MLX's graph fusion turns this
    # pattern into a single Metal dispatch after mx.compile.
    return mx.stack(outs, axis=1)


class RwkvTimeMix(nn.Module):
    def __init__(self, n_embed: int, n_layer: int, layer_id: int,
                 rwkv_rank: float, dropout_prob: float):
        super().__init__()
        self.n_embed = n_embed
        self.layer_id = layer_id
        attn_sz = n_embed

        # Fancy init matching the PyTorch reference. These are
        # overwritten by load_weights when we're converting from a
        # checkpoint, but we want correct init for from-scratch runs.
        ratio_0_to_1 = (layer_id / (n_layer - 1)) if n_layer > 1 else 0.0
        ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)

        # time_decay: one value per channel, seed pattern that grades
        # from slow decay (early channels) to fast decay (late channels).
        decay_seed = mx.array([
            -5.0 + 8.0 * (h / (attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            for h in range(attn_sz)
        ])
        self.time_decay = decay_seed

        # time_first: base log(0.3) plus a ±0.25 zigzag by channel index.
        zigzag = mx.array([((i + 1) % 3 - 1) * 0.5 for i in range(attn_sz)])
        self.time_first = mx.full((attn_sz,), math.log(0.3)) + zigzag

        # time_mix: per-channel blend ratios from 0 to 1. Stored (1, 1, C).
        x = mx.arange(n_embed, dtype=mx.float32) / n_embed
        x = x.reshape(1, 1, n_embed)
        self.time_mix_k = mx.power(x, ratio_1_to_almost0)
        self.time_mix_v = mx.power(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1
        self.time_mix_r = mx.power(x, 0.5 * ratio_1_to_almost0)
        # time_mix_g kept for state_dict compatibility (computed but
        # unused in the Phase 11 reference training path).
        self.time_mix_g = mx.power(x, 0.5 * ratio_1_to_almost0)

        self.key = nn.Linear(n_embed, n_embed, bias=False)
        self.value = nn.Linear(n_embed, n_embed, bias=False)
        self.receptance = nn.Linear(n_embed, n_embed, bias=False)

        high_rank = int(n_embed * rwkv_rank)
        self.key_A = nn.Linear(n_embed, high_rank, bias=False)
        self.key_B = nn.Linear(high_rank, attn_sz, bias=False)
        self.value_A = nn.Linear(n_embed, high_rank, bias=False)
        self.value_B = nn.Linear(high_rank, attn_sz, bias=False)
        self.receptance_A = nn.Linear(n_embed, high_rank, bias=False)
        self.receptance_B = nn.Linear(high_rank, attn_sz, bias=False)

        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None
        self.output = nn.Linear(attn_sz, n_embed, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        xx = time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk) + self.key_B(self.key_A(xk))
        v = self.value(xv) + self.value_B(self.value_A(xv))
        r = self.receptance(xr) + self.receptance_B(self.receptance_A(xr))
        sr = mx.sigmoid(r)

        # time_decay/time_first are (C,); WKV wants them as (C,).
        # Prefer the custom-backward `wkv` from mlx_wkv; fall back to the
        # auto-diff wkv_forward if mlx_wkv didn't import (for tests).
        wkv_fn = _wkv_custom if _wkv_custom is not None else wkv_forward
        wkv_out = wkv_fn(self.time_decay, self.time_first, k, v)
        rwkv = sr * wkv_out
        if self.dropout is not None:
            rwkv = self.dropout(rwkv)
        return self.output(rwkv)


class RwkvChannelMix(nn.Module):
    def __init__(self, n_embed: int, ffn_dim: int, n_layer: int,
                 layer_id: int, rwkv_rank: float, dropout_prob: float):
        super().__init__()
        self.n_embed = n_embed
        self.ffn_dim = ffn_dim

        ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)
        x = mx.arange(n_embed, dtype=mx.float32) / n_embed
        x = x.reshape(1, 1, n_embed)
        self.time_mix_k = mx.power(x, ratio_1_to_almost0)
        self.time_mix_r = mx.power(x, ratio_1_to_almost0)

        self.key = nn.Linear(n_embed, ffn_dim, bias=False)
        self.receptance = nn.Linear(n_embed, n_embed, bias=False)
        self.value = nn.Linear(ffn_dim, n_embed, bias=False)

        high_rank = int(n_embed * rwkv_rank)
        self.key_A = nn.Linear(n_embed, high_rank, bias=False)
        self.key_B = nn.Linear(high_rank, ffn_dim, bias=False)
        self.value_A = nn.Linear(ffn_dim, high_rank, bias=False)
        self.value_B = nn.Linear(high_rank, n_embed, bias=False)
        self.receptance_A = nn.Linear(n_embed, high_rank, bias=False)
        self.receptance_B = nn.Linear(high_rank, n_embed, bias=False)

        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None

    def __call__(self, x: mx.array) -> mx.array:
        xx = time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk) + self.key_B(self.key_A(xk))
        k = mx.square(mx.maximum(k, 0.0))  # square-ReLU
        kv = self.value(k) + self.value_B(self.value_A(k))
        r = self.receptance(xr) + self.receptance_B(self.receptance_A(xr))

        rkv = mx.sigmoid(r) * kv
        if self.dropout is not None:
            rkv = self.dropout(rkv)
        return rkv


class Block(nn.Module):
    def __init__(self, n_embed: int, ffn_dim: int, n_layer: int,
                 layer_id: int, rwkv_rank: float, dropout_prob: float):
        super().__init__()
        self.layer_id = layer_id
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.att = RwkvTimeMix(n_embed, n_layer, layer_id, rwkv_rank, dropout_prob)
        self.ffn = RwkvChannelMix(n_embed, ffn_dim, n_layer, layer_id, rwkv_rank, dropout_prob)
        self.short = nn.Linear(n_embed, n_embed, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        if self.layer_id == 0:
            x = self.ln0(x)
        short = mx.maximum(self.short(x), 0.0)  # ReLU
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x + short


class RwkvTcHira(nn.Module):
    """MLX port of L3TC's RWKV_TC_HIRA (2L × 96H × 16K base config)."""

    def __init__(self, vocab_size: int = 16384, hidden_size: int = 96,
                 num_hidden_layers: int = 2, intermediate_size: int = 96,
                 rwkv_rank: float = 4, ctx_len: int = 2048,
                 dropout_prob: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ctx_len = ctx_len

        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.blocks = [
            Block(hidden_size, intermediate_size, num_hidden_layers,
                  i, rwkv_rank, dropout_prob)
            for i in range(num_hidden_layers)
        ]
        self.ln_out = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, input_token: mx.array) -> mx.array:
        """input_token: (B, T) int32/int64; returns logits (B, T, V)."""
        # Phase 11 script passed (B, 1, T); MLX-side we accept (B, T) only
        # to keep the graph simpler. `scripts/mlx_rwkv_parity.py` adapts.
        x = self.emb(input_token)
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)
        return self.head(x)

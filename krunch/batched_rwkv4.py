"""
Batched RWKV-4 forward pass — runs `B` independent token streams through
one model.forward() call. The BlinkDL WKV CUDA kernel already supports
B>1 via `_b * T * C + _c` indexing; the rwkv package's Python wrapper
hardcodes B=1. This module reimplements the v4 forward on top of a
loaded `rwkv.model.RWKV` instance with a real batch dim.

Used by krunch's compress path to push N parallel chunk-streams through
the model in one launch instead of N sequential single-stream forwards,
breaking the per-stream throughput wall (~95-110K tok/s on A10G).

Limitations:
- v4 only (RWKV-4-Pile-169M is what krunch ships in v1).
- fp16 strategy only (`cuda fp16`); no quantized/i8 paths.
- `full_output=True` semantics: returns logits at every position.

The model weights and per-layer strategies are read from the supplied
`RWKV` instance; we never duplicate weights.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def _matmul(x, w, output_dtype=None):
    """Mirror rwkv.model.matmul_float for fp16 cuBLAS path. Plain `x @ w`
    has different fp16 precision than rwkv's `gemm_fp16_cublas` (the
    upstream uses fp32 accumulation via cuBLAS); using `@` produces
    logits that diverge by ~12 abs from the unbatched reference path.

    `KRUNCH_PLAIN_MATMUL=1` falls back to `@` (torch.compile-friendly:
    `torch.ops.rwkv.gemm_fp16_cublas` is opaque to dynamo and triggers
    FakeTensor errors during graph trace). Use this when wrapping the
    forward in torch.compile / torch.cuda.graph. Drift implications
    are measured separately by bench_forward.py.
    """
    import os
    if output_dtype is None:
        output_dtype = x.dtype
    use_plain = os.environ.get("KRUNCH_PLAIN_MATMUL") == "1"
    if (not use_plain and x.dtype == w.dtype == torch.float16
            and x.device.type == 'cuda'):
        # gemm_fp16_cublas only supports matched dims (2D x 2D or 3D x 3D).
        # Our case is 3D x 2D — flatten to 2D, gemm, reshape.
        assert x.dim() in (2, 3) and w.dim() == 2
        if x.dim() == 3:
            B, T, C = x.shape
            x_flat = x.contiguous().view(B * T, C)
            c_flat = torch.empty((B * T, w.shape[-1]), dtype=output_dtype, device=x.device)
            torch.ops.rwkv.gemm_fp16_cublas(x_flat, w, c_flat)
            return c_flat.view(B, T, w.shape[-1])
        c = torch.empty((x.shape[0], w.shape[-1]), dtype=output_dtype, device=x.device)
        torch.ops.rwkv.gemm_fp16_cublas(x, w, c)
        return c
    return (x @ w).to(output_dtype)


def init_state_batched(m, B: int, device="cuda") -> List[torch.Tensor]:
    """Build a fresh state list for a B-way batched v4 forward.
    Returns 5 * n_layer tensors (matching m.forward's state contract)
    but with a leading batch dim added.

    State layout per layer (v4):
      i*5+0  att_xx   [B, n_embd]   atype (fp16)
      i*5+1  att_aa   [B, n_att]    fp32
      i*5+2  att_bb   [B, n_att]    fp32
      i*5+3  att_pp   [B, n_att]    fp32, init -1e30
      i*5+4  ffn_xx   [B, n_embd]   atype (fp16)
    """
    args = m.args
    state: List[torch.Tensor] = [None] * args.n_layer * 5
    for i in range(args.n_layer):
        dd = m.strategy[i]
        atype = dd.atype
        state[i*5+0] = torch.zeros((B, args.n_embd), dtype=atype, device=device).contiguous()
        state[i*5+1] = torch.zeros((B, args.n_att), dtype=torch.float32, device=device).contiguous()
        state[i*5+2] = torch.zeros((B, args.n_att), dtype=torch.float32, device=device).contiguous()
        state[i*5+3] = torch.full((B, args.n_att), -1e30, dtype=torch.float32, device=device).contiguous()
        state[i*5+4] = torch.zeros((B, args.n_embd), dtype=atype, device=device).contiguous()
    return state


def _ffn_seq_batched(x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw):
    """v4 channel-mixing layer, batched over leading dim B.

    x: [B, T, C] (fp16)
    sx: [B, C] previous-token state (fp16)

    Returns (x_out: [B, T, C], new_sx: [B, C]).
    """
    xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
    # Concat per-batch: previous sx as time-step -1, drop last to keep T length.
    sx_full = torch.cat((sx.unsqueeze(1), xx[:, :-1, :]), dim=1)
    kx = xx * k_mix + sx_full * (1 - k_mix)
    rx = xx * r_mix + sx_full * (1 - r_mix)
    r = torch.sigmoid(_matmul(rx, rw))
    vx = torch.relu(_matmul(kx, kw)) ** 2
    out = r * _matmul(vx, vw)
    return x + out, xx[:, -1, :]


def _att_seq_batched(x, sx, aa, bb, pp,
                     ln_w, ln_b, k_mix, v_mix, r_mix,
                     t_decay, t_first,
                     kw, vw, rw, ow):
    """v4 time-mixing layer, batched over leading dim B.

    x: [B, T, C], sx: [B, C], aa/bb/pp: [B, C] (fp32 for state).
    """
    B, T, C = x.shape
    xx = F.layer_norm(x, (C,), weight=ln_w, bias=ln_b)
    sx_full = torch.cat((sx.unsqueeze(1), xx[:, :-1, :]), dim=1)
    kx = xx * k_mix + sx_full * (1 - k_mix)
    vx = xx * v_mix + sx_full * (1 - v_mix)
    rx = xx * r_mix + sx_full * (1 - r_mix)

    r = torch.sigmoid(_matmul(rx, rw))
    # k, v in fp32 for the WKV kernel.
    k = _matmul(kx, kw, output_dtype=torch.float32)
    v = _matmul(vx, vw, output_dtype=torch.float32)

    import os
    use_pure_wkv = os.environ.get("KRUNCH_PURE_WKV") == "1"

    if use_pure_wkv:
        # Pure-torch WKV recurrence — torch.compile-friendly because no
        # custom op (the rwkv package's wkv_forward is opaque to dynamo).
        # Slower per-step than the kernel for T>1 because the recurrence
        # is a Python loop, but for T=1 it's a single update so cost is
        # similar. Numerical match should be within fp16 noise of the
        # kernel since the math is identical.
        y_steps = []
        aa_c = aa
        bb_c = bb
        pp_c = pp
        for t in range(T):
            kt = k[:, t, :]            # [B, C]
            vt = v[:, t, :]
            ww = pp_c + t_first
            p = torch.maximum(ww, kt)
            e1 = torch.exp(ww - p)
            e2 = torch.exp(kt - p)
            yt = (e1 * aa_c + e2 * vt) / (e1 * bb_c + e2)
            y_steps.append(yt)
            # State update for next timestep
            ww2 = pp_c - torch.exp(t_decay)
            p2 = torch.maximum(ww2, kt)
            e1_2 = torch.exp(ww2 - p2)
            e2_2 = torch.exp(kt - p2)
            aa_c = e1_2 * aa_c + e2_2 * vt
            bb_c = e1_2 * bb_c + e2_2
            pp_c = p2
        y = torch.stack(y_steps, dim=1).to(x.dtype)  # [B, T, C]
    else:
        # Flatten batch + time for the kernel: it expects contiguous [B*T, C]
        # layout indexed as `_b * T * C + _c`.
        k_flat = k.contiguous().view(B * T, C)
        v_flat = v.contiguous().view(B * T, C)
        y_flat = torch.empty_like(k_flat)
        aa_c = aa.contiguous(); bb_c = bb.contiguous(); pp_c = pp.contiguous()
        torch.ops.rwkv.wkv_forward(
            B, T, C,
            t_decay.contiguous(), t_first.contiguous(),
            k_flat, v_flat, y_flat,
            aa_c, bb_c, pp_c,
        )
        y = y_flat.view(B, T, C).to(x.dtype)

    out = _matmul(r * y, ow)
    return x + out, xx[:, -1, :], aa_c, bb_c, pp_c


def forward_batched(m, tokens_batches,
                    state: Optional[List[torch.Tensor]] = None,
                    full_output: bool = True
                    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Run B token streams through the v4 model in one batched forward.

    Args:
        m: rwkv.model.RWKV (loaded with strategy='cuda fp16').
        tokens_batches: either
          - `List[List[int]]` of B equal-length token streams, or
          - `torch.LongTensor` of shape [B, T] already on the model's
            device (sync-free path used by batched decompress).
        state: optional initial state from `init_state_batched`. None ->
            zero-initialized state.
        full_output: if False, return only the last position's logits per
            batch.

    Returns:
        (logits, new_state)
        logits: [B, T, V] (or [B, V] if not full_output)
        new_state: list of per-layer tensors, same shape as `state`.
    """
    assert m.version == 4, f"batched forward implemented for v4 only, got v{m.version}"
    args = m.args
    w = m.w
    device = w['emb.weight'].device

    # Accept tensor inputs to avoid the host-side sync that
    # `torch.as_tensor(python_list)` requires per step in the decode loop.
    if isinstance(tokens_batches, torch.Tensor):
        idx = tokens_batches.to(device=device, dtype=torch.long)
        if idx.dim() == 1:
            idx = idx.unsqueeze(1)
        B, T = idx.shape
    else:
        B = len(tokens_batches)
        T = len(tokens_batches[0])
        assert all(len(b) == T for b in tokens_batches), \
            "all batch streams must have same token-length T (caller pads)"
        idx = torch.as_tensor(tokens_batches, dtype=torch.long, device=device)

    if state is None:
        state = init_state_batched(m, B, device=device)

    # Embedding lookup: [B, T] -> [B, T, n_embd]
    x = w['emb.weight'][idx]  # [B, T, n_embd], dtype = emb.weight dtype

    for i in range(args.n_layer):
        dd = m.strategy[i]
        atype = dd.atype
        x = x.to(dtype=atype, device=dd.device)
        # Move state slices to this layer's device + dtype as well —
        # rwkv strategies can place layers on different devices and the
        # state was initialized on the embedding's device, which may not
        # match for every layer.
        for s_off in range(5):
            tgt = state[i*5+s_off]
            target_dtype = atype if s_off in (0, 4) else torch.float32
            if tgt.device != dd.device or tgt.dtype != target_dtype:
                state[i*5+s_off] = tgt.to(device=dd.device, dtype=target_dtype)
        att = f'blocks.{i}.att.'
        ffn = f'blocks.{i}.ffn.'
        bbb = f'blocks.{i}.'

        # Time-mix (attention)
        x, state[i*5+0], state[i*5+1], state[i*5+2], state[i*5+3] = _att_seq_batched(
            x, state[i*5+0], state[i*5+1], state[i*5+2], state[i*5+3],
            w[bbb+'ln1.weight'], w[bbb+'ln1.bias'],
            w[att+'time_mix_k'], w[att+'time_mix_v'], w[att+'time_mix_r'],
            w[att+'time_decay'], w[att+'time_first'],
            w[att+'key.weight'], w[att+'value.weight'],
            w[att+'receptance.weight'], w[att+'output.weight'],
        )

        # Channel-mix (ffn)
        x, state[i*5+4] = _ffn_seq_batched(
            x, state[i*5+4],
            w[bbb+'ln2.weight'], w[bbb+'ln2.bias'],
            w[ffn+'time_mix_k'], w[ffn+'time_mix_r'],
            w[ffn+'key.weight'], w[ffn+'value.weight'], w[ffn+'receptance.weight'],
        )

    if not full_output:
        x = x[:, -1, :]  # [B, n_embd]
    x = F.layer_norm(x, (args.n_embd,),
                     weight=w['ln_out.weight'], bias=w['ln_out.bias'])
    x = x @ w['head.weight']
    return x, state

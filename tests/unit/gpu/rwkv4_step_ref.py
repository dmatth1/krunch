"""Reference Python implementation of RWKV-4 single-step forward (B=1, T=1).

This is the ground truth that the upcoming CUDA fused kernel
(`krunch_ac/cuda/rwkv_step.cu`) will be tested against. Written to
match `_att_seq_batched + _ffn_seq_batched` from `batched_rwkv4.py`
exactly at B=1, T=1, with state mutated in place.

The function is also fast: at B=1, T=1 every operation is a small
GEMV or elementwise pass. Pure torch ops; works with torch.compile
once the BlinkDL custom-op blockers are bypassed.

Usage:
    from rwkv4_step_ref import init_state_step, step_one
    state = init_state_step(model)
    logits, state = step_one(model, token_id, state)
"""

import os
import torch
import torch.nn.functional as F


def init_state_step(model, device="cuda"):
    """Build state list for B=1 single-step forward. Returns 5*n_layer
    tensors matching `forward_batched`'s state contract but with B=1
    fixed and shape `[C]` (no leading batch dim — saves the squeeze).
    """
    args = model.args
    state = [None] * (args.n_layer * 5)
    for i in range(args.n_layer):
        dd = model.strategy[i]
        atype = dd.atype
        state[i*5+0] = torch.zeros(args.n_embd, dtype=atype, device=device)
        state[i*5+1] = torch.zeros(args.n_att, dtype=torch.float32, device=device)
        state[i*5+2] = torch.zeros(args.n_att, dtype=torch.float32, device=device)
        state[i*5+3] = torch.full((args.n_att,), -1e30, dtype=torch.float32, device=device)
        state[i*5+4] = torch.zeros(args.n_embd, dtype=atype, device=device)
    return state


def _layer_step(x, att_xx, aa, bb, pp, ffn_xx,
                ln1_w, ln1_b, ln2_w, ln2_b,
                tm_k, tm_v, tm_r, t_decay, t_first,
                Kw, Vw, Rw, Ow,
                ffn_tm_k, ffn_tm_r, ffn_Kw, ffn_Vw, ffn_Rw):
    """One RWKV-4 layer at B=1, T=1.

    Args:
        x: [n_embd] input vector (fp16)
        att_xx, ffn_xx: [n_embd] previous-token state (fp16)
        aa, bb, pp: [n_att] WKV state (fp32)
        weights: matrices stored as [in, out] (rwkv convention) — apply
            via `x @ W`, not `x @ W.T`.

    Returns:
        x_new: [n_embd] (fp16)
        att_xx_new: [n_embd]
        aa_new, bb_new, pp_new: [n_att]
        ffn_xx_new: [n_embd]
    """
    C = x.shape[0]

    # ---- Time-mix ----
    xx = F.layer_norm(x, (C,), weight=ln1_w, bias=ln1_b)

    kx = xx * tm_k + att_xx * (1 - tm_k)
    vx = xx * tm_v + att_xx * (1 - tm_v)
    rx = xx * tm_r + att_xx * (1 - tm_r)

    # GEMVs: x[C] @ W[C,C].T = output[C]. We store W as [out, in], so
    # output = x @ W.T. (BlinkDL stores weight transposed for matmul.)
    r = torch.sigmoid(rx @ Rw)
    k = (kx @ Kw).float()  # fp32 for WKV
    v = (vx @ Vw).float()

    # WKV update — pure-torch single-step (matches the kernel's math).
    # Output:    y = (e1 * aa + e2 * v) / (e1 * bb + e2)
    #            with ww = pp + t_first; p = max(ww, k); e1 = exp(ww-p); e2 = exp(k-p)
    # State:     ww2 = pp - exp(t_decay); p2 = max(ww2, k);
    #            aa <- e1' * aa + e2' * v; bb <- e1' * bb + e2'; pp <- p2
    ww = pp + t_first
    p = torch.maximum(ww, k)
    e1 = torch.exp(ww - p)
    e2 = torch.exp(k - p)
    y = (e1 * aa + e2 * v) / (e1 * bb + e2)

    decay = torch.exp(t_decay)  # t_decay is stored as log-decay
    ww2 = pp - decay
    p2 = torch.maximum(ww2, k)
    e1_2 = torch.exp(ww2 - p2)
    e2_2 = torch.exp(k - p2)
    aa_new = e1_2 * aa + e2_2 * v
    bb_new = e1_2 * bb + e2_2
    pp_new = p2

    att_out = (r * y.to(x.dtype)) @ Ow
    x = x + att_out
    att_xx_new = xx

    # ---- Channel-mix ----
    xx2 = F.layer_norm(x, (C,), weight=ln2_w, bias=ln2_b)

    ffn_kx = xx2 * ffn_tm_k + ffn_xx * (1 - ffn_tm_k)
    ffn_rx = xx2 * ffn_tm_r + ffn_xx * (1 - ffn_tm_r)

    r_ffn = torch.sigmoid(ffn_rx @ ffn_Rw)
    k_ffn = torch.relu(ffn_kx @ ffn_Kw) ** 2  # [n_ffn]
    v_ffn = k_ffn @ ffn_Vw                    # [n_embd]
    x = x + r_ffn * v_ffn
    ffn_xx_new = xx2

    return x, att_xx_new, aa_new, bb_new, pp_new, ffn_xx_new


def step_one(model, token_id: int, state):
    """Full single-step forward for B=1, T=1. Mutates `state` in place
    by replacing each layer's tensors with the new values; returns
    the next-token logits [V] in fp32.
    """
    w = model.w
    args = model.args
    device = w['emb.weight'].device

    x = w['emb.weight'][token_id]  # [n_embd]

    for i in range(args.n_layer):
        dd = model.strategy[i]
        atype = dd.atype
        x = x.to(dtype=atype, device=dd.device)
        # State slices may live on different devices (per layer strategy).
        s_xx  = state[i*5+0].to(device=dd.device, dtype=atype)
        s_aa  = state[i*5+1].to(device=dd.device, dtype=torch.float32)
        s_bb  = state[i*5+2].to(device=dd.device, dtype=torch.float32)
        s_pp  = state[i*5+3].to(device=dd.device, dtype=torch.float32)
        s_ffn = state[i*5+4].to(device=dd.device, dtype=atype)

        att = f'blocks.{i}.att.'
        ffn = f'blocks.{i}.ffn.'
        bbb = f'blocks.{i}.'

        x, s_xx_new, s_aa_new, s_bb_new, s_pp_new, s_ffn_new = _layer_step(
            x, s_xx, s_aa, s_bb, s_pp, s_ffn,
            w[bbb+'ln1.weight'], w[bbb+'ln1.bias'],
            w[bbb+'ln2.weight'], w[bbb+'ln2.bias'],
            w[att+'time_mix_k'].squeeze(), w[att+'time_mix_v'].squeeze(),
            w[att+'time_mix_r'].squeeze(),
            w[att+'time_decay'], w[att+'time_first'],
            # rwkv stores weights as [in, out]: x @ W gives output. No .T.
            w[att+'key.weight'], w[att+'value.weight'],
            w[att+'receptance.weight'], w[att+'output.weight'],
            w[ffn+'time_mix_k'].squeeze(), w[ffn+'time_mix_r'].squeeze(),
            w[ffn+'key.weight'], w[ffn+'value.weight'],
            w[ffn+'receptance.weight'],
        )

        state[i*5+0] = s_xx_new
        state[i*5+1] = s_aa_new
        state[i*5+2] = s_bb_new
        state[i*5+3] = s_pp_new
        state[i*5+4] = s_ffn_new

    x = F.layer_norm(x, (args.n_embd,),
                     weight=w['ln_out.weight'], bias=w['ln_out.bias'])
    logits = x @ w['head.weight']  # head stored as [n_embd, V], no .T
    return logits, state


# Self-test against forward_batched(B=1, T=1) at the same input.
if __name__ == "__main__":
    import sys
    if "--test" not in sys.argv:
        print("usage: python rwkv4_step_ref.py --test  (requires CUDA + RWKV model)")
        sys.exit(0)

    os.environ.setdefault("RWKV_JIT_ON", "1")
    os.environ.setdefault("RWKV_CUDA_ON", "1")
    os.environ["KRUNCH_PURE_WKV"] = "1"  # forward_batched should match our pure WKV

    from krunch.inference import _load_rwkv, MODEL_PATH
    from krunch.batched_rwkv4 import forward_batched, init_state_batched

    RWKV = _load_rwkv()
    model = RWKV(model=str(MODEL_PATH).removesuffix(".pth"),
                 strategy="cuda fp16", verbose=False)
    device = "cuda"

    # Reference (forward_batched) state and ours, parallel
    state_fb = init_state_batched(model, B=1, device=device)
    state_ref = init_state_step(model, device=device)
    tok_t = torch.zeros(1, 1, dtype=torch.long, device=device)

    seq = [0, 100, 1000, 5, 42, 9999]
    for tok in seq:
        tok_t.fill_(tok)
        logits_fb, state_fb = forward_batched(model, tok_t, state_fb, full_output=False)
        logits_fb = logits_fb.flatten().float()

        logits_ref, state_ref = step_one(model, tok, state_ref)
        logits_ref = logits_ref.float()

        diff = (logits_fb - logits_ref).abs().max().item()
        print(f"DRIFT_ref_vs_fb tok={tok} max_abs={diff:.4f}", flush=True)

"""Bit-exact C++ orchestration path for compress + decompress.

Wraps `krunch_ac_cuda.rwkv4_layer_step_cpp{,_t1}` + the deterministic
matmul kernel + per-row softmax/CDF so encoder (packed T) and decoder
(stepped T=1) produce byte-identical AC bitstreams.

Set `KRUNCH_DETERMINISTIC_MATMUL=1` and `KRUNCH_CPP_PATH=1` to
activate. Verified bit-exact on the 32-token end-to-end harness in
`scripts/test_e2e_ac_roundtrip.py`.
"""
from __future__ import annotations

import os

N_LAYER = 12

_WEIGHTS_CACHE: dict[int, dict] = {}


def init_weights(model, device: str = "cuda") -> dict:
    """Extract per-layer weights once per (model, device). Cached by
    id(model)."""
    import torch
    key = id(model)
    if key in _WEIGHTS_CACHE:
        return _WEIGHTS_CACHE[key]

    w = model.w

    def fix(t, dt=None):
        t = t.to(device).contiguous()
        if dt is not None and t.dtype != dt:
            t = t.to(dtype=dt).contiguous()
        return t

    layers = []
    for i in range(N_LAYER):
        bbb = f"blocks.{i}."
        att = f"blocks.{i}.att."
        ffn = f"blocks.{i}.ffn."
        layers.append([
            fix(w[bbb+'ln1.weight'], torch.float16),
            fix(w[bbb+'ln1.bias'], torch.float16),
            fix(w[att+'time_mix_k'].squeeze(), torch.float16),
            fix(w[att+'time_mix_v'].squeeze(), torch.float16),
            fix(w[att+'time_mix_r'].squeeze(), torch.float16),
            fix(w[att+'time_decay'], torch.float32),
            fix(w[att+'time_first'], torch.float32),
            fix(w[att+'key.weight'], torch.float16),
            fix(w[att+'value.weight'], torch.float16),
            fix(w[att+'receptance.weight'], torch.float16),
            fix(w[att+'output.weight'], torch.float16),
            fix(w[bbb+'ln2.weight'], torch.float16),
            fix(w[bbb+'ln2.bias'], torch.float16),
            fix(w[ffn+'time_mix_k'].squeeze(), torch.float16),
            fix(w[ffn+'time_mix_r'].squeeze(), torch.float16),
            fix(w[ffn+'key.weight'], torch.float16),
            fix(w[ffn+'value.weight'], torch.float16),
            fix(w[ffn+'receptance.weight'], torch.float16),
        ])

    bundle = {
        "layers": layers,
        "emb_w": fix(w['emb.weight'], torch.float16),
        "ln_out_w": fix(w['ln_out.weight'], torch.float16),
        "ln_out_b": fix(w['ln_out.bias'], torch.float16),
        "head_w": fix(w['head.weight'], torch.float16),
        "n_embd": int(model.args.n_embd),
        "n_att": int(model.args.n_att),
        "device": device,
    }
    _WEIGHTS_CACHE[key] = bundle
    return bundle


def fresh_state(weights: dict):
    """Initial RWKV-4 state (att_xx, aa, bb, pp, ffn_xx) per layer."""
    import torch
    device = weights["device"]
    n_embd = weights["n_embd"]
    n_att = weights["n_att"]
    return (
        [torch.zeros(1, n_embd, dtype=torch.float16, device=device) for _ in range(N_LAYER)],
        [torch.zeros(1, n_att, dtype=torch.float32, device=device) for _ in range(N_LAYER)],
        [torch.zeros(1, n_att, dtype=torch.float32, device=device) for _ in range(N_LAYER)],
        [torch.full((1, n_att), -1e30, dtype=torch.float32, device=device) for _ in range(N_LAYER)],
        [torch.zeros(1, n_embd, dtype=torch.float16, device=device) for _ in range(N_LAYER)],
    )


def forward_packed(weights: dict, input_ids, state):
    """Run all 12 layers packed (T = len(input_ids)). Returns
    logits [T, V]. Mutates state in place."""
    import torch
    import krunch_ac_cuda
    import torch.nn.functional as F
    device = weights["device"]
    n_embd = weights["n_embd"]
    layers = weights["layers"]
    emb_w = weights["emb_w"]
    ln_out_w = weights["ln_out_w"]
    ln_out_b = weights["ln_out_b"]
    head_w = weights["head_w"]

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.as_tensor(input_ids, dtype=torch.long, device=device)
    T = int(input_ids.shape[0])
    x = emb_w[input_ids].view(1, T, n_embd).contiguous()
    for i in range(N_LAYER):
        x = krunch_ac_cuda.rwkv4_layer_step_cpp(
            x.contiguous(),
            state[0][i], state[1][i], state[2][i], state[3][i], state[4][i],
            *layers[i],
        )
    # Per-row ln_out + det_matmul to match decoder shape exactly.
    x_flat = x.view(T, n_embd)
    xn = torch.cat([
        F.layer_norm(x_flat[t:t+1], (n_embd,), weight=ln_out_w, bias=ln_out_b)
        for t in range(T)
    ], dim=0)
    logits = krunch_ac_cuda.det_matmul(xn.contiguous(), head_w.contiguous())
    return logits  # [T, V]


def forward_stepped(weights: dict, last_token: int, state):
    """Run all 12 layers for one new token. Returns logits [V].
    Mutates state in place."""
    import torch
    import krunch_ac_cuda
    import torch.nn.functional as F
    device = weights["device"]
    n_embd = weights["n_embd"]
    layers = weights["layers"]
    emb_w = weights["emb_w"]
    ln_out_w = weights["ln_out_w"]
    ln_out_b = weights["ln_out_b"]
    head_w = weights["head_w"]

    x = emb_w[last_token].view(1, 1, n_embd).contiguous()
    for i in range(N_LAYER):
        x = krunch_ac_cuda.rwkv4_layer_step_cpp_t1(
            x.contiguous(),
            state[0][i], state[1][i], state[2][i], state[3][i], state[4][i],
            *layers[i],
        )
    xn = F.layer_norm(x.view(1, n_embd), (n_embd,),
                      weight=ln_out_w, bias=ln_out_b)
    logits = krunch_ac_cuda.det_matmul(xn.contiguous(), head_w.contiguous()).flatten()
    return logits  # [V]


def softmax_cdfs_per_row(logits_TxV):
    """Batched softmax + CDF via det_softmax_cdf kernel.
    Bit-identical between [T,V] and [1,V] invocation (verified) so
    encoder and decoder produce the same CDFs. ~20× faster than the
    per-row Python loop on T=1024."""
    import krunch_ac_cuda
    from krunch_ac.cdf import T as CDF_T
    return krunch_ac_cuda.det_softmax_cdf(logits_TxV.contiguous(), CDF_T)


def softmax_cdf_one_row(logits_V):
    """Single-row softmax + CDF for the stepped decoder path."""
    import krunch_ac_cuda
    from krunch_ac.cdf import T as CDF_T
    cdf = krunch_ac_cuda.det_softmax_cdf(
        logits_V.reshape(1, -1).contiguous(), CDF_T)
    return cdf[0].contiguous()


def cpp_path_enabled() -> bool:
    return os.environ.get("KRUNCH_CPP_PATH", "0") == "1"

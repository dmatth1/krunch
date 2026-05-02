"""Unit test: fused-layer kernel vs `rwkv4_step_ref::_layer_step`.

Loads the real RWKV-4-Pile-169M, picks layer 0, runs both the kernel
and the Python reference on identical inputs + state. Pass = max abs
logit diff < 0.5 (kernel is fp32-accumulated GEMV; ref uses torch's
matmul which may round differently).

Pre-req: `cd krunch_ac/cuda && python setup.py build_ext --inplace`
on the GPU host so `krunch_ac_cuda` exposes `rwkv4_layer_step`.

Usage (inside container):
    python /work/test_rwkv_step_kernel.py
"""

import os
import torch

os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

from krunch.inference import _load_rwkv, MODEL_PATH
from rwkv4_step_ref import _layer_step

import krunch_ac_cuda


def main():
    RWKV = _load_rwkv()
    print("loading RWKV-4-Pile-169M ...", flush=True)
    model = RWKV(model=str(MODEL_PATH).removesuffix(".pth"),
                 strategy="cuda fp16", verbose=False)
    print("loaded", flush=True)
    device = "cuda"

    args = model.args
    n_embd = args.n_embd
    n_att  = args.n_att
    print(f"n_embd={n_embd} n_att={n_att} n_layer={args.n_layer}", flush=True)
    assert n_embd == 768 and n_att == 768, \
        f"kernel expects 768/768, got {n_embd}/{n_att}"

    w = model.w
    LAYER = 0
    bbb = f"blocks.{LAYER}."
    att = f"blocks.{LAYER}.att."
    ffn = f"blocks.{LAYER}.ffn."

    # Random fp16 input, random initial state to exercise non-zero state path
    torch.manual_seed(42)
    x = torch.randn(n_embd, dtype=torch.float16, device=device) * 0.1
    att_xx_init = torch.randn(n_embd, dtype=torch.float16, device=device) * 0.1
    aa_init = torch.randn(n_att, dtype=torch.float32, device=device) * 0.1
    bb_init = torch.randn(n_att, dtype=torch.float32, device=device) * 0.1 + 1.0
    pp_init = torch.full((n_att,), -1e30, dtype=torch.float32, device=device)
    ffn_xx_init = torch.randn(n_embd, dtype=torch.float16, device=device) * 0.1

    # ---- Reference path ----
    x_ref = x.clone()
    att_xx_ref = att_xx_init.clone()
    aa_ref = aa_init.clone()
    bb_ref = bb_init.clone()
    pp_ref = pp_init.clone()
    ffn_xx_ref = ffn_xx_init.clone()

    x_ref_out, att_xx_ref_out, aa_ref_out, bb_ref_out, pp_ref_out, ffn_xx_ref_out = _layer_step(
        x_ref, att_xx_ref, aa_ref, bb_ref, pp_ref, ffn_xx_ref,
        w[bbb+'ln1.weight'], w[bbb+'ln1.bias'],
        w[bbb+'ln2.weight'], w[bbb+'ln2.bias'],
        w[att+'time_mix_k'].squeeze(),
        w[att+'time_mix_v'].squeeze(),
        w[att+'time_mix_r'].squeeze(),
        w[att+'time_decay'], w[att+'time_first'],
        w[att+'key.weight'], w[att+'value.weight'],
        w[att+'receptance.weight'], w[att+'output.weight'],
        w[ffn+'time_mix_k'].squeeze(),
        w[ffn+'time_mix_r'].squeeze(),
        w[ffn+'key.weight'], w[ffn+'value.weight'], w[ffn+'receptance.weight'],
    )

    # ---- Kernel path ----
    x_k = x.clone()
    x_k_out = torch.zeros_like(x_k)
    att_xx_k = att_xx_init.clone()
    aa_k = aa_init.clone()
    bb_k = bb_init.clone()
    pp_k = pp_init.clone()
    ffn_xx_k = ffn_xx_init.clone()

    krunch_ac_cuda.rwkv4_layer_step(
        x_k, x_k_out,
        att_xx_k, aa_k, bb_k, pp_k, ffn_xx_k,
        w[bbb+'ln1.weight'].contiguous(), w[bbb+'ln1.bias'].contiguous(),
        w[att+'time_mix_k'].squeeze().contiguous(),
        w[att+'time_mix_v'].squeeze().contiguous(),
        w[att+'time_mix_r'].squeeze().contiguous(),
        w[att+'time_decay'].contiguous(), w[att+'time_first'].contiguous(),
        w[att+'key.weight'].contiguous(),
        w[att+'value.weight'].contiguous(),
        w[att+'receptance.weight'].contiguous(),
        w[att+'output.weight'].contiguous(),
        w[bbb+'ln2.weight'].contiguous(), w[bbb+'ln2.bias'].contiguous(),
        w[ffn+'time_mix_k'].squeeze().contiguous(),
        w[ffn+'time_mix_r'].squeeze().contiguous(),
        w[ffn+'key.weight'].contiguous(),
        w[ffn+'value.weight'].contiguous(),
        w[ffn+'receptance.weight'].contiguous(),
    )
    torch.cuda.synchronize()

    # ---- Compare ----
    def diff(a, b, name):
        a32 = a.float()
        b32 = b.float()
        d = (a32 - b32).abs().max().item()
        print(f"DIFF {name} max_abs={d:.5f}  shape={tuple(a.shape)} dtype={a.dtype}", flush=True)
        return d

    print("--- kernel vs reference, layer 0 ---", flush=True)
    d_x   = diff(x_k_out,    x_ref_out,        "x_out")
    d_att = diff(att_xx_k,   att_xx_ref_out,   "att_xx")
    d_aa  = diff(aa_k,       aa_ref_out,       "aa")
    d_bb  = diff(bb_k,       bb_ref_out,       "bb")
    d_pp  = diff(pp_k,       pp_ref_out,       "pp")
    d_ffn = diff(ffn_xx_k,   ffn_xx_ref_out,   "ffn_xx")

    THRESH = 0.5
    pass_ = max(d_x, d_att, d_aa, d_bb, d_pp, d_ffn) < THRESH
    if pass_:
        print(f"PASS — all diffs < {THRESH}", flush=True)
    else:
        print(f"FAIL — at least one diff >= {THRESH}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Single-layer v1 vs v2 comparison from fresh state.
Isolates per-call kernel differences from multi-layer compounding."""
import os, sys
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import torch
import krunch_ac_cuda
import rwkv.model
from krunch.inference import _load_rwkv, MODEL_PATH
from krunch import cpp_path


def maxabs(a, b): return (a.float() - b.float()).abs().max().item()


def main():
    print("Loading model...")
    RWKV = _load_rwkv()
    model = RWKV(model=str(MODEL_PATH).removesuffix(".pth"), strategy="cuda fp16")
    weights = cpp_path.init_weights(model, "cuda")

    n_embd = weights["n_embd"]
    n_att = weights["n_att"]
    layers = weights["layers"]
    emb_w = weights["emb_w"]
    L = layers[0]  # just layer 0

    scratch = torch.empty(krunch_ac_cuda.v2_scratch_bytes(),
                          dtype=torch.uint8, device="cuda")

    # Fresh state for v1
    s1_att = torch.zeros(n_embd, dtype=torch.float16, device="cuda")
    s1_aa = torch.zeros(n_att, dtype=torch.float32, device="cuda")
    s1_bb = torch.zeros(n_att, dtype=torch.float32, device="cuda")
    s1_pp = torch.full((n_att,), -1e30, dtype=torch.float32, device="cuda")
    s1_ffn = torch.zeros(n_embd, dtype=torch.float16, device="cuda")

    # Fresh state for v2
    s2_att = torch.zeros(n_embd, dtype=torch.float16, device="cuda")
    s2_aa = torch.zeros(n_att, dtype=torch.float32, device="cuda")
    s2_bb = torch.zeros(n_att, dtype=torch.float32, device="cuda")
    s2_pp = torch.full((n_att,), -1e30, dtype=torch.float32, device="cuda")
    s2_ffn = torch.zeros(n_embd, dtype=torch.float16, device="cuda")

    x_in = emb_w[0].view(n_embd).clone()
    x1_out = torch.empty(n_embd, dtype=torch.float16, device="cuda")
    x2_out = torch.empty(n_embd, dtype=torch.float16, device="cuda")

    krunch_ac_cuda.rwkv4_layer_step(x_in, x1_out,
                                     s1_att, s1_aa, s1_bb, s1_pp, s1_ffn, *L)
    krunch_ac_cuda.rwkv4_layer_step_v2(x_in, x2_out,
                                        s2_att, s2_aa, s2_bb, s2_pp, s2_ffn,
                                        *L, scratch)
    torch.cuda.synchronize()

    print(f"\nLayer 0, fresh state, single-step:")
    print(f"  x_out diff:   {maxabs(x1_out, x2_out):.6e}  (v1 vs v2)")
    print(f"  att_xx diff:  {maxabs(s1_att, s2_att):.6e}")
    print(f"  aa diff:      {maxabs(s1_aa, s2_aa):.6e}")
    print(f"  bb diff:      {maxabs(s1_bb, s2_bb):.6e}")
    print(f"  pp diff:      {maxabs(s1_pp, s2_pp):.6e}")
    print(f"  ffn_xx diff:  {maxabs(s1_ffn, s2_ffn):.6e}")

    print(f"\nv1 x_out range: {x1_out.float().min():.3f}..{x1_out.float().max():.3f}")
    print(f"v2 x_out range: {x2_out.float().min():.3f}..{x2_out.float().max():.3f}")
    print(f"v1 aa range:    {s1_aa.min():.3f}..{s1_aa.max():.3f}")
    print(f"v2 aa range:    {s2_aa.min():.3f}..{s2_aa.max():.3f}")
    print(f"v1 pp range:    {s1_pp.min():.3f}..{s1_pp.max():.3f}")
    print(f"v2 pp range:    {s2_pp.min():.3f}..{s2_pp.max():.3f}")


if __name__ == "__main__":
    main()

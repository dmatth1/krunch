"""Verify v3 kernel's full layer step: M6 milestone.

All phases active (Phase 3 KVR matmul + Phase 10 final residual now
implemented). Compare x_out + state against torch full-layer reference.

Threshold: 5e-2 — accumulated fp16 noise across 4 matmuls + 2 LNs +
WKV + sigmoid + relu² + residual chain.
"""
import os, sys
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")

import torch
import krunch_ac_cuda


def maxabs(a, b): return (a.float() - b.float()).abs().max().item()


def torch_full_layer(x, att_xx_prev, ffn_xx_prev, aa, bb, pp,
                     ln1_w, ln1_b, tm_k, tm_v, tm_r,
                     time_decay, time_first,
                     Kw, Vw, Rw, Ow,
                     ln2_w, ln2_b, ffn_tm_k, ffn_tm_r,
                     ffn_Kw, ffn_Vw, ffn_Rw):
    """Full RWKV-4 layer, T=1 step, fp16 in/out, fp32 internal."""
    C = x.shape[-1]

    # LN1
    xx = torch.nn.functional.layer_norm(
        x.float(), (C,), weight=ln1_w.float(), bias=ln1_b.float()).to(torch.float16)
    # premix
    att_v = att_xx_prev.float()
    kx = (xx.float() * tm_k.float() + att_v * (1 - tm_k.float())).to(torch.float16)
    vx = (xx.float() * tm_v.float() + att_v * (1 - tm_v.float())).to(torch.float16)
    rx = (xx.float() * tm_r.float() + att_v * (1 - tm_r.float())).to(torch.float16)
    att_xx_new = xx
    # KVR
    k = (kx.float() @ Kw.float())
    v = (vx.float() @ Vw.float())
    r_pre = (rx.float() @ Rw.float()).to(torch.float16)
    # WKV
    u, w = time_first, time_decay
    ww = u.unsqueeze(0) + k
    p1 = torch.maximum(pp, ww)
    e1 = torch.exp(pp - p1); e2 = torch.exp(ww - p1)
    y = (e1 * aa + e2 * v) / (e1 * bb + e2)
    ww2 = w.unsqueeze(0) + pp
    p2 = torch.maximum(ww2, k)
    e1_2 = torch.exp(ww2 - p2); e2_2 = torch.exp(k - p2)
    aa_new = e1_2 * aa + e2_2 * v
    bb_new = e1_2 * bb + e2_2
    pp_new = p2
    # Ow + residual
    sig_r = torch.sigmoid(r_pre.float())
    pre = sig_r * y
    attn_out = pre @ Ow.float()
    x_attn = (x.float() + attn_out).to(torch.float16)
    # LN2
    xx2 = torch.nn.functional.layer_norm(
        x_attn.float(), (C,), weight=ln2_w.float(), bias=ln2_b.float())
    xx2_h = xx2.to(torch.float16)
    # ffn-premix
    fxv = ffn_xx_prev.float()
    fk = (xx2 * ffn_tm_k.float() + fxv * (1 - ffn_tm_k.float())).to(torch.float16)
    fr = (xx2 * ffn_tm_r.float() + fxv * (1 - ffn_tm_r.float())).to(torch.float16)
    ffn_xx_new = xx2_h
    # FFN
    K_act = torch.clamp(fk.float() @ ffn_Kw.float(), min=0.0).pow(2).to(torch.float16)
    R_act = torch.sigmoid(fr.float() @ ffn_Rw.float()).to(torch.float16)
    V_out = (K_act.float() @ ffn_Vw.float()).to(torch.float16)
    # Final residual
    x_out = (x_attn.float() + R_act.float() * V_out.float()).to(torch.float16)
    return x_out, att_xx_new, ffn_xx_new, aa_new, bb_new, pp_new


def main():
    device = "cuda"
    torch.manual_seed(17)
    B = 128
    C = 768
    n_att = 768
    n_ffn = 3072

    x_in = torch.randn(B, C, dtype=torch.float16, device=device) * 0.1
    att_xx_prev = torch.randn(B, C, dtype=torch.float16, device=device) * 0.05
    ffn_xx_prev = torch.randn(B, C, dtype=torch.float16, device=device) * 0.05
    aa_init = torch.randn(B, n_att, dtype=torch.float32, device=device) * 0.5
    bb_init = torch.randn(B, n_att, dtype=torch.float32, device=device).abs() + 0.1
    pp_init = torch.randn(B, n_att, dtype=torch.float32, device=device) * 0.5

    ln1_w = torch.randn(C, dtype=torch.float16, device=device) * 0.5 + 1.0
    ln1_b = torch.randn(C, dtype=torch.float16, device=device) * 0.05
    tm_k = torch.rand(C, dtype=torch.float16, device=device)
    tm_v = torch.rand(C, dtype=torch.float16, device=device)
    tm_r = torch.rand(C, dtype=torch.float16, device=device)
    time_decay = (-torch.rand(n_att, dtype=torch.float32, device=device) - 0.1)
    time_first = torch.randn(n_att, dtype=torch.float32, device=device) * 0.3
    Kw = torch.randn(C, n_att, dtype=torch.float16, device=device) * 0.05
    Vw = torch.randn(C, n_att, dtype=torch.float16, device=device) * 0.05
    Rw = torch.randn(C, C,     dtype=torch.float16, device=device) * 0.05
    Ow = torch.randn(n_att, C, dtype=torch.float16, device=device) * 0.05
    ln2_w = torch.randn(C, dtype=torch.float16, device=device) * 0.3 + 1.0
    ln2_b = torch.randn(C, dtype=torch.float16, device=device) * 0.05
    ffn_tm_k = torch.rand(C, dtype=torch.float16, device=device)
    ffn_tm_r = torch.rand(C, dtype=torch.float16, device=device)
    ffn_Kw = torch.randn(C, n_ffn, dtype=torch.float16, device=device) * 0.04
    ffn_Vw = torch.randn(n_ffn, C, dtype=torch.float16, device=device) * 0.04
    ffn_Rw = torch.randn(C, C, dtype=torch.float16, device=device) * 0.04

    # Reference
    x_out_ref, att_xx_after, ffn_xx_after, aa_after, bb_after, pp_after = torch_full_layer(
        x_in, att_xx_prev, ffn_xx_prev, aa_init.clone(), bb_init.clone(), pp_init.clone(),
        ln1_w, ln1_b, tm_k, tm_v, tm_r,
        time_decay, time_first,
        Kw, Vw, Rw, Ow,
        ln2_w, ln2_b, ffn_tm_k, ffn_tm_r,
        ffn_Kw, ffn_Vw, ffn_Rw)

    # Kernel state
    aa = aa_init.clone(); bb = bb_init.clone(); pp = pp_init.clone()
    att_xx = att_xx_prev.clone()
    ffn_xx = ffn_xx_prev.clone()
    x_out = torch.zeros(B, C, dtype=torch.float16, device=device)

    n_bytes = krunch_ac_cuda.v3_scratch_bytes(B)
    scratch = torch.zeros(n_bytes, dtype=torch.uint8, device=device)
    print(f"v3 scratch bytes (B={B}): {n_bytes}")

    krunch_ac_cuda.rwkv4_layer_step_v3(
        B, x_in, x_out,
        att_xx, aa, bb, pp, ffn_xx,
        ln1_w, ln1_b, tm_k, tm_v, tm_r,
        time_decay, time_first,
        Kw, Vw, Rw, Ow,
        ln2_w, ln2_b, ffn_tm_k, ffn_tm_r,
        ffn_Kw, ffn_Vw, ffn_Rw,
        scratch,
    )
    torch.cuda.synchronize()

    print(f"\nFull layer output diffs (B={B}):")
    print(f"  x_out    max-abs vs ref: {maxabs(x_out, x_out_ref):.6e}")
    print(f"  att_xx   max-abs vs ref: {maxabs(att_xx, att_xx_after):.6e}")
    print(f"  ffn_xx   max-abs vs ref: {maxabs(ffn_xx, ffn_xx_after):.6e}")
    print(f"  aa       max-abs vs ref: {maxabs(aa, aa_after):.6e}")
    print(f"  bb       max-abs vs ref: {maxabs(bb, bb_after):.6e}")
    print(f"  pp       max-abs vs ref: {maxabs(pp, pp_after):.6e}")

    THRESHOLD = 5e-2
    diffs = [
        ("x_out", maxabs(x_out, x_out_ref)),
        ("att_xx", maxabs(att_xx, att_xx_after)),
        ("ffn_xx", maxabs(ffn_xx, ffn_xx_after)),
        ("aa", maxabs(aa, aa_after)),
        ("bb", maxabs(bb, bb_after)),
        ("pp", maxabs(pp, pp_after)),
    ]
    pass_ = all(d < THRESHOLD for _, d in diffs)
    print(f"\nverdict: {'PASS' if pass_ else 'FAIL'} (threshold {THRESHOLD:.0e})")
    sys.exit(0 if pass_ else 1)


if __name__ == "__main__":
    main()

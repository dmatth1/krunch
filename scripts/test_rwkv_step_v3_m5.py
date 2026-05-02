"""Verify v3 kernel's Phase 8 (ffn_K + relu²), Phase 8b (ffn_R + sigmoid),
Phase 9 (ffn_V matmul). M5 milestone.

Strategy: feed real inputs through full chain, pre-fill k_acc/v_acc/r_pre
in scratch (Phase 3 still stubbed), validate ffn_K_act / ffn_R_act / ffn_V_out
out of scratch.

Threshold: 5e-2 — accumulated fp16 noise across Phase 5 matmul (768) +
Phase 8 matmul (768) + relu² + Phase 9 matmul (3072).
"""
import os, sys
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")

import torch
import krunch_ac_cuda


def maxabs(a, b): return (a.float() - b.float()).abs().max().item()


def main():
    device = "cuda"
    torch.manual_seed(13)
    B = 128
    C = 768
    n_att = 768
    n_ffn = 3072
    V3_B_MAX = 256
    V3_GRID_DIM = 6

    # Real inputs
    x_in = torch.randn(B, C, dtype=torch.float16, device=device) * 0.1
    att_xx_prev = torch.randn(B, C, dtype=torch.float16, device=device) * 0.05
    ln1_w = torch.randn(C, dtype=torch.float16, device=device) * 0.5 + 1.0
    ln1_b = torch.randn(C, dtype=torch.float16, device=device) * 0.05
    tm_k = torch.rand(C, dtype=torch.float16, device=device)
    tm_v = torch.rand(C, dtype=torch.float16, device=device)
    tm_r = torch.rand(C, dtype=torch.float16, device=device)

    k_ref = torch.randn(B, n_att, dtype=torch.float32, device=device) * 0.5
    v_ref = torch.randn(B, n_att, dtype=torch.float32, device=device) * 0.5
    r_pre_ref = torch.randn(B, n_att, dtype=torch.float16, device=device) * 0.3
    aa_init = torch.randn(B, n_att, dtype=torch.float32, device=device) * 0.5
    bb_init = torch.randn(B, n_att, dtype=torch.float32, device=device).abs() + 0.1
    pp_init = torch.randn(B, n_att, dtype=torch.float32, device=device) * 0.5
    time_decay = (-torch.rand(n_att, dtype=torch.float32, device=device) - 0.1)
    time_first = torch.randn(n_att, dtype=torch.float32, device=device) * 0.3

    Ow = torch.randn(n_att, C, dtype=torch.float16, device=device) * 0.05
    ln2_w = torch.randn(C, dtype=torch.float16, device=device) * 0.3 + 1.0
    ln2_b = torch.randn(C, dtype=torch.float16, device=device) * 0.05
    ffn_xx_prev = torch.randn(B, C, dtype=torch.float16, device=device) * 0.05
    ffn_tm_k = torch.rand(C, dtype=torch.float16, device=device)
    ffn_tm_r = torch.rand(C, dtype=torch.float16, device=device)
    ffn_Kw = torch.randn(C, n_ffn, dtype=torch.float16, device=device) * 0.04
    ffn_Vw = torch.randn(n_ffn, C, dtype=torch.float16, device=device) * 0.04
    ffn_Rw = torch.randn(C, C, dtype=torch.float16, device=device) * 0.04

    # Reference: full chain through M5
    # WKV
    u, w = time_first, time_decay
    ww = u.unsqueeze(0) + k_ref
    p1 = torch.maximum(pp_init, ww)
    e1 = torch.exp(pp_init - p1); e2 = torch.exp(ww - p1)
    y_ref = (e1 * aa_init + e2 * v_ref) / (e1 * bb_init + e2)
    # Phase 5: Ow + residual
    sig_r = torch.sigmoid(r_pre_ref.float())
    pre = sig_r * y_ref
    attn_out = pre @ Ow.float()
    x_attn_ref = (x_in.float() + attn_out).to(torch.float16)
    # Phase 6: LN2
    xx2 = torch.nn.functional.layer_norm(
        x_attn_ref.float(), (C,), weight=ln2_w.float(), bias=ln2_b.float())
    # Phase 7: ffn-premix
    ffn_xx_v = ffn_xx_prev.float()
    fk = (xx2 * ffn_tm_k.float() + ffn_xx_v * (1.0 - ffn_tm_k.float())).to(torch.float16)
    fr = (xx2 * ffn_tm_r.float() + ffn_xx_v * (1.0 - ffn_tm_r.float())).to(torch.float16)
    # Phase 8: ffn_K + relu²; Phase 8b: ffn_R + sigmoid
    ffn_K = fk.float() @ ffn_Kw.float()  # [B, n_ffn]
    ffn_K_act_ref = torch.clamp(ffn_K, min=0.0).pow(2).to(torch.float16)
    ffn_R = fr.float() @ ffn_Rw.float()  # [B, C]
    ffn_R_act_ref = torch.sigmoid(ffn_R).to(torch.float16)
    # Phase 9: ffn_V
    ffn_V_ref = (ffn_K_act_ref.float() @ ffn_Vw.float()).to(torch.float16)

    # Kernel state buffers
    aa = aa_init.clone(); bb = bb_init.clone(); pp = pp_init.clone()
    att_xx = att_xx_prev.clone()
    ffn_xx = ffn_xx_prev.clone()
    x_out = torch.zeros(B, C, dtype=torch.float16, device=device)

    # Unused
    def dummy(*shape):
        return torch.zeros(shape, dtype=torch.float16, device=device)
    Kw = dummy(C, n_att); Vw = dummy(C, n_att); Rw = dummy(C, C)

    n_bytes = krunch_ac_cuda.v3_scratch_bytes(B)
    scratch = torch.zeros(n_bytes, dtype=torch.uint8, device=device)
    print(f"v3 scratch bytes (B={B}): {n_bytes}")

    # Pre-fill k_acc, v_acc, r_pre
    ln_partials_floats = 2 * V3_B_MAX * V3_GRID_DIM
    halves_per_buf = V3_B_MAX * C
    kvrx_floats = (3 * halves_per_buf) // 2
    k_acc_off_f = ln_partials_floats + kvrx_floats
    v_acc_off_f = k_acc_off_f + V3_B_MAX * n_att
    r_pre_off_f = v_acc_off_f + V3_B_MAX * n_att
    y_buf_off_f = r_pre_off_f + (V3_B_MAX * C) // 2
    x_attn_off_f = y_buf_off_f + V3_B_MAX * n_att
    ln2_partials_off_f = x_attn_off_f + (V3_B_MAX * C) // 2
    ffn_kx_off_f = ln2_partials_off_f + 2 * V3_B_MAX * V3_GRID_DIM
    ffn_rx_off_f = ffn_kx_off_f + (V3_B_MAX * C) // 2
    ffn_K_act_off_f = ffn_rx_off_f + (V3_B_MAX * C) // 2
    ffn_V_out_off_f = ffn_K_act_off_f + (V3_B_MAX * n_ffn) // 2
    ffn_R_act_off_f = ffn_V_out_off_f + (V3_B_MAX * C) // 2

    scratch_f = scratch.view(torch.float32)
    scratch_h = scratch.view(torch.float16)
    scratch_f[k_acc_off_f : k_acc_off_f + V3_B_MAX * n_att].view(V3_B_MAX, n_att)[:B] = k_ref
    scratch_f[v_acc_off_f : v_acc_off_f + V3_B_MAX * n_att].view(V3_B_MAX, n_att)[:B] = v_ref
    r_pre_off_h = r_pre_off_f * 2
    scratch_h[r_pre_off_h : r_pre_off_h + V3_B_MAX * C].view(V3_B_MAX, C)[:B] = r_pre_ref

    # Launch
    krunch_ac_cuda.rwkv4_layer_step_v3(
        B, x_in, x_out,
        att_xx, aa, bb, pp, ffn_xx,
        ln1_w, ln1_b, tm_k, tm_v, tm_r,
        time_decay, time_first,
        Kw, Vw, Rw, Ow,
        ln2_w, ln2_b,
        ffn_tm_k, ffn_tm_r,
        ffn_Kw, ffn_Vw, ffn_Rw,
        scratch,
    )
    torch.cuda.synchronize()

    # Read M5 outputs
    ffn_K_act_off_h = ffn_K_act_off_f * 2
    ffn_V_out_off_h = ffn_V_out_off_f * 2
    ffn_R_act_off_h = ffn_R_act_off_f * 2
    ffn_K_act_view = scratch_h[ffn_K_act_off_h : ffn_K_act_off_h + V3_B_MAX * n_ffn].view(V3_B_MAX, n_ffn)
    ffn_V_view = scratch_h[ffn_V_out_off_h : ffn_V_out_off_h + V3_B_MAX * C].view(V3_B_MAX, C)
    ffn_R_act_view = scratch_h[ffn_R_act_off_h : ffn_R_act_off_h + V3_B_MAX * C].view(V3_B_MAX, C)

    K_act = ffn_K_act_view[:B]
    V_out = ffn_V_view[:B]
    R_act = ffn_R_act_view[:B]

    print(f"\nM5 phase output diffs (B={B}):")
    print(f"  ffn_K_act max-abs vs ref: {maxabs(K_act, ffn_K_act_ref):.6e}")
    print(f"  ffn_R_act max-abs vs ref: {maxabs(R_act, ffn_R_act_ref):.6e}")
    print(f"  ffn_V_out max-abs vs ref: {maxabs(V_out, ffn_V_ref):.6e}")

    THRESHOLD = 5e-2
    diffs = [maxabs(K_act, ffn_K_act_ref), maxabs(R_act, ffn_R_act_ref),
             maxabs(V_out, ffn_V_ref)]
    pass_ = all(d < THRESHOLD for d in diffs)
    print(f"\nverdict: {'PASS' if pass_ else 'FAIL'} (threshold {THRESHOLD:.0e})")
    sys.exit(0 if pass_ else 1)


if __name__ == "__main__":
    main()

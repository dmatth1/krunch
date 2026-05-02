"""Verify v3 kernel's Phase 5 (Ow + residual), Phase 6 (LN2),
Phase 7 (ffn-premix). M4 milestone.

Strategy:
- Provide real inputs for all weights touched by Phases 1, 2, 4-7.
- Pre-fill k_acc, v_acc, r_pre in scratch (Phase 3 still stubbed).
- Run kernel. Read x_attn, ffn_kx, ffn_rx out of scratch + ffn_xx state.
- Compare vs. pure-torch reference.

Threshold: 5e-3 abs (fp16 storage noise dominates after Phase 5 matmul).
"""
import os, sys
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")

import torch
import krunch_ac_cuda


def maxabs(a, b): return (a.float() - b.float()).abs().max().item()


def ref_full(x, ln1_w, ln1_b, tm_k, tm_v, tm_r, att_xx_prev,
             k_ref, v_ref, r_pre_ref,
             time_decay, time_first, aa_in, bb_in, pp_in,
             Ow, ln2_w, ln2_b, ffn_tm_k, ffn_tm_r, ffn_xx_prev):
    """Pure-torch ref for phases 1-7."""
    # LN1 (M2 already validated)
    xx = torch.nn.functional.layer_norm(
        x.float(), (x.shape[-1],), weight=ln1_w.float(), bias=ln1_b.float())
    xx_h = xx.to(x.dtype)

    # WKV (M3 already validated). Use pre-filled k_ref / v_ref.
    u = time_first
    w = time_decay
    ww = u.unsqueeze(0) + k_ref
    p1 = torch.maximum(pp_in, ww)
    e1 = torch.exp(pp_in - p1)
    e2 = torch.exp(ww - p1)
    y_ref = (e1 * aa_in + e2 * v_ref) / (e1 * bb_in + e2)
    ww2 = w.unsqueeze(0) + pp_in
    p2 = torch.maximum(ww2, k_ref)
    e1_2 = torch.exp(ww2 - p2)
    e2_2 = torch.exp(k_ref - p2)
    aa_after = e1_2 * aa_in + e2_2 * v_ref
    bb_after = e1_2 * bb_in + e2_2
    pp_after = p2

    # Phase 5: Ow + residual.
    # Ow stored [n_att, C] row-major. pre = sigmoid(r_pre) * y; out = pre @ Ow
    sig_r = torch.sigmoid(r_pre_ref.float())  # [B, n_att]
    pre = sig_r * y_ref                       # [B, n_att]
    attn_out = pre @ Ow.float()               # [B, C]
    x_attn_ref = (x.float() + attn_out).to(x.dtype)

    # Phase 6: LN2
    xx2 = torch.nn.functional.layer_norm(
        x_attn_ref.float(), (x.shape[-1],), weight=ln2_w.float(), bias=ln2_b.float())
    xx2_h = xx2.to(x.dtype)

    # Phase 7: ffn-premix
    ffn_xx_v = ffn_xx_prev.float()
    fk_ref = (xx2 * ffn_tm_k.float() + ffn_xx_v * (1.0 - ffn_tm_k.float())).to(x.dtype)
    fr_ref = (xx2 * ffn_tm_r.float() + ffn_xx_v * (1.0 - ffn_tm_r.float())).to(x.dtype)
    ffn_xx_after = xx2_h

    return x_attn_ref, fk_ref, fr_ref, ffn_xx_after


def main():
    device = "cuda"
    torch.manual_seed(11)
    B = 128
    C = 768
    n_att = 768
    n_ffn = 3072
    V3_B_MAX = 256
    V3_GRID_DIM = 6

    # Real inputs / weights
    x_in = torch.randn(B, C, dtype=torch.float16, device=device) * 0.1
    att_xx_prev = torch.randn(B, C, dtype=torch.float16, device=device) * 0.05
    ln1_w = torch.randn(C, dtype=torch.float16, device=device) * 0.5 + 1.0
    ln1_b = torch.randn(C, dtype=torch.float16, device=device) * 0.05
    tm_k = torch.rand(C, dtype=torch.float16, device=device)
    tm_v = torch.rand(C, dtype=torch.float16, device=device)
    tm_r = torch.rand(C, dtype=torch.float16, device=device)

    # Phase 3 stubbed — we pre-fill k_acc, v_acc, r_pre in scratch.
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

    # Reference
    x_attn_ref, fk_ref, fr_ref, ffn_xx_after = ref_full(
        x_in, ln1_w, ln1_b, tm_k, tm_v, tm_r, att_xx_prev,
        k_ref, v_ref, r_pre_ref,
        time_decay, time_first, aa_init.clone(), bb_init.clone(), pp_init.clone(),
        Ow, ln2_w, ln2_b, ffn_tm_k, ffn_tm_r, ffn_xx_prev)

    # Kernel inputs
    aa = aa_init.clone()
    bb = bb_init.clone()
    pp = pp_init.clone()
    att_xx = att_xx_prev.clone()
    ffn_xx = ffn_xx_prev.clone()
    x_out = torch.zeros(B, C, dtype=torch.float16, device=device)

    # Unused (Phase 3, FFN R/K/V stubbed)
    def dummy(*shape):
        return torch.zeros(shape, dtype=torch.float16, device=device)
    Kw = dummy(C, n_att); Vw = dummy(C, n_att); Rw = dummy(C, C)
    ffn_Kw = dummy(C, n_ffn); ffn_Vw = dummy(n_ffn, C); ffn_Rw = dummy(C, C)

    # Allocate scratch + pre-fill k_acc, v_acc, r_pre
    n_bytes = krunch_ac_cuda.v3_scratch_bytes(B)
    scratch = torch.zeros(n_bytes, dtype=torch.uint8, device=device)
    print(f"v3 scratch bytes (B={B}): {n_bytes}")

    # Offsets (mirror kernel layout)
    ln_partials_floats = 2 * V3_B_MAX * V3_GRID_DIM
    halves_per_buf = V3_B_MAX * C
    kvrx_floats = (3 * halves_per_buf) // 2
    k_acc_off_f = ln_partials_floats + kvrx_floats
    v_acc_off_f = k_acc_off_f + V3_B_MAX * n_att
    r_pre_off_f = v_acc_off_f + V3_B_MAX * n_att

    scratch_f = scratch.view(torch.float32)
    k_view = scratch_f[k_acc_off_f : k_acc_off_f + V3_B_MAX * n_att].view(V3_B_MAX, n_att)
    v_view = scratch_f[v_acc_off_f : v_acc_off_f + V3_B_MAX * n_att].view(V3_B_MAX, n_att)
    k_view[:B] = k_ref
    v_view[:B] = v_ref

    # r_pre in halves: write via half view
    scratch_h = scratch.view(torch.float16)
    r_pre_off_h = r_pre_off_f * 2  # floats→halves: ×2
    r_view = scratch_h[r_pre_off_h : r_pre_off_h + V3_B_MAX * C].view(V3_B_MAX, C)
    r_view[:B] = r_pre_ref

    # Launch v3 kernel
    krunch_ac_cuda.rwkv4_layer_step_v3(
        B, x_in, x_out,
        att_xx, aa, bb, pp, ffn_xx,
        ln1_w, ln1_b,
        tm_k, tm_v, tm_r,
        time_decay, time_first,
        Kw, Vw, Rw, Ow,
        ln2_w, ln2_b,
        ffn_tm_k, ffn_tm_r,
        ffn_Kw, ffn_Vw, ffn_Rw,
        scratch,
    )
    torch.cuda.synchronize()

    # Read x_attn, ffn_kx, ffn_rx out of scratch
    # x_attn comes after y_buf:
    y_buf_off_f = r_pre_off_f + (V3_B_MAX * C) // 2
    x_attn_off_f = y_buf_off_f + V3_B_MAX * n_att
    ln2_partials_off_f = x_attn_off_f + (V3_B_MAX * C) // 2
    ffn_kx_off_f = ln2_partials_off_f + 2 * V3_B_MAX * V3_GRID_DIM
    ffn_rx_off_f = ffn_kx_off_f + (V3_B_MAX * C) // 2

    x_attn_off_h = x_attn_off_f * 2
    ffn_kx_off_h = ffn_kx_off_f * 2
    ffn_rx_off_h = ffn_rx_off_f * 2

    x_attn_view = scratch_h[x_attn_off_h : x_attn_off_h + V3_B_MAX * C].view(V3_B_MAX, C)
    ffn_kx_view = scratch_h[ffn_kx_off_h : ffn_kx_off_h + V3_B_MAX * C].view(V3_B_MAX, C)
    ffn_rx_view = scratch_h[ffn_rx_off_h : ffn_rx_off_h + V3_B_MAX * C].view(V3_B_MAX, C)

    x_attn = x_attn_view[:B]
    ffn_kx = ffn_kx_view[:B]
    ffn_rx = ffn_rx_view[:B]

    print(f"\nM4 phase output diffs (B={B}):")
    print(f"  x_attn   max-abs vs ref: {maxabs(x_attn, x_attn_ref):.6e}")
    print(f"  ffn_kx   max-abs vs ref: {maxabs(ffn_kx, fk_ref):.6e}")
    print(f"  ffn_rx   max-abs vs ref: {maxabs(ffn_rx, fr_ref):.6e}")
    print(f"  ffn_xx   max-abs vs ref: {maxabs(ffn_xx, ffn_xx_after):.6e}")

    THRESHOLD = 5e-3
    diffs = [maxabs(x_attn, x_attn_ref), maxabs(ffn_kx, fk_ref),
             maxabs(ffn_rx, fr_ref), maxabs(ffn_xx, ffn_xx_after)]
    pass_ = all(d < THRESHOLD for d in diffs)
    print(f"\nverdict: {'PASS' if pass_ else 'FAIL'} (threshold {THRESHOLD:.0e})")
    sys.exit(0 if pass_ else 1)


if __name__ == "__main__":
    main()

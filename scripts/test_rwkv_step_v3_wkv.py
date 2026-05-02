"""Verify v3 kernel's Phase 4 WKV (M3 milestone).

Phase 3 (KVR matmul) is still stubbed in v3 — it doesn't write k_acc/v_acc.
This test pre-fills k_acc/v_acc in scratch with synthetic values, runs
the v3 kernel, and verifies that Phase 4 (WKV) output (y_buf) and state
updates (aa/bb/pp) match a Python reference implementation of the same
WKV recurrence.

Acceptable diff: < 1e-3 abs (WKV math is fp32 throughout — tighter than
the fp16-bracket phases).
"""
import os, sys
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")

import torch
import krunch_ac_cuda


def maxabs(a, b): return (a.float() - b.float()).abs().max().item()


def ref_wkv(k, v, time_decay, time_first, aa_in, bb_in, pp_in):
    """Pure-torch reference: WKV at B>1, T=1.
    k, v shape [B, n_att] fp32. time_decay/first shape [n_att] fp32.
    Returns (y, aa_new, bb_new, pp_new)."""
    u = time_first  # [n_att]
    w = time_decay  # [n_att] (already -exp(raw))

    # y = (exp(pp - p1) * aa + exp(u + k - p1) * v) / denom
    ww = u.unsqueeze(0) + k                          # [B, n_att]
    p1 = torch.maximum(pp_in, ww)
    e1 = torch.exp(pp_in - p1)
    e2 = torch.exp(ww - p1)
    y = (e1 * aa_in + e2 * v) / (e1 * bb_in + e2)

    # state update
    ww2 = w.unsqueeze(0) + pp_in
    p2 = torch.maximum(ww2, k)
    e1_2 = torch.exp(ww2 - p2)
    e2_2 = torch.exp(k - p2)
    aa_new = e1_2 * aa_in + e2_2 * v
    bb_new = e1_2 * bb_in + e2_2
    pp_new = p2

    return y, aa_new, bb_new, pp_new


def main():
    device = "cuda"
    torch.manual_seed(7)
    B = 128
    C = 768
    n_att = 768
    n_ffn = 3072
    V3_B_MAX = 256
    V3_GRID_DIM = 6

    # Synthetic k_acc, v_acc, state — values shaped like real RWKV-4 layer step
    k_ref = torch.randn(B, n_att, dtype=torch.float32, device=device) * 0.5
    v_ref = torch.randn(B, n_att, dtype=torch.float32, device=device) * 0.5
    aa_init = torch.randn(B, n_att, dtype=torch.float32, device=device) * 0.5
    bb_init = torch.randn(B, n_att, dtype=torch.float32, device=device).abs() + 0.1
    pp_init = torch.randn(B, n_att, dtype=torch.float32, device=device) * 0.5
    time_decay = (-torch.rand(n_att, dtype=torch.float32, device=device) - 0.1)
    time_first = torch.randn(n_att, dtype=torch.float32, device=device) * 0.3

    # Compute reference (from a clone of state, since kernel mutates in place)
    y_ref, aa_after, bb_after, pp_after = ref_wkv(
        k_ref, v_ref, time_decay, time_first,
        aa_init.clone(), bb_init.clone(), pp_init.clone())

    # Set up kernel inputs / state
    aa = aa_init.clone()
    bb = bb_init.clone()
    pp = pp_init.clone()

    # Other state buffers (untouched in M3)
    att_xx = torch.zeros(B, C, dtype=torch.float16, device=device)
    ffn_xx = torch.zeros(B, C, dtype=torch.float16, device=device)
    x_in = torch.zeros(B, C, dtype=torch.float16, device=device)
    x_out = torch.zeros(B, C, dtype=torch.float16, device=device)

    # Weights — only LN1 + premix weights matter for Phases 1+2 (ignored for M3 verification)
    ln1_w = torch.ones(C, dtype=torch.float16, device=device)
    ln1_b = torch.zeros(C, dtype=torch.float16, device=device)
    tm_k = torch.zeros(C, dtype=torch.float16, device=device)
    tm_v = torch.zeros(C, dtype=torch.float16, device=device)
    tm_r = torch.zeros(C, dtype=torch.float16, device=device)

    def dummy(*shape):
        return torch.zeros(shape, dtype=torch.float16, device=device)

    ln2_w = dummy(C); ln2_b = dummy(C)
    ffn_tm_k = dummy(C); ffn_tm_r = dummy(C)
    Kw = dummy(C, n_att); Vw = dummy(C, n_att); Rw = dummy(C, C); Ow = dummy(n_att, C)
    ffn_Kw = dummy(C, n_ffn); ffn_Vw = dummy(n_ffn, C); ffn_Rw = dummy(C, C)

    # Allocate scratch
    n_bytes = krunch_ac_cuda.v3_scratch_bytes(B)
    scratch = torch.zeros(n_bytes, dtype=torch.uint8, device=device)
    print(f"v3 scratch bytes (B={B}): {n_bytes}")

    # Pre-fill k_acc and v_acc in scratch.
    # Layout (offsets in floats):
    #   ln_partials:    0
    #   ln_partial_sq:  V3_B_MAX * V3_GRID_DIM
    #   kx (halves):    2 * V3_B_MAX * V3_GRID_DIM (in floats) = ...
    #   vx, rx after kx
    # After 3 × V3_B_MAX × C halves of kx/vx/rx:
    #   k_acc (floats)
    #   v_acc
    #   r_pre (halves)
    #   y_buf (floats)
    ln_partials_floats = 2 * V3_B_MAX * V3_GRID_DIM
    halves_per_buf = V3_B_MAX * C
    kvrx_halves = 3 * halves_per_buf
    kvrx_floats = kvrx_halves // 2  # halves to floats: divide by 2

    # k_acc starts at offset (in floats)
    k_acc_off_floats = ln_partials_floats + kvrx_floats
    v_acc_off_floats = k_acc_off_floats + V3_B_MAX * n_att

    scratch_f = scratch.view(torch.float32)
    # Write k_ref into scratch[k_acc_off_floats : k_acc_off_floats + B*n_att]
    # Note: kernel expects [B_MAX, n_att] but we only fill [B, n_att].
    k_acc_view = scratch_f[k_acc_off_floats : k_acc_off_floats + V3_B_MAX * n_att].view(V3_B_MAX, n_att)
    v_acc_view = scratch_f[v_acc_off_floats : v_acc_off_floats + V3_B_MAX * n_att].view(V3_B_MAX, n_att)
    k_acc_view[:B] = k_ref
    v_acc_view[:B] = v_ref

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

    # Read y_buf from scratch
    y_buf_off_floats = (v_acc_off_floats + V3_B_MAX * n_att
                       + (V3_B_MAX * C) // 2)  # past r_pre halves
    y_view = scratch_f[y_buf_off_floats : y_buf_off_floats + V3_B_MAX * n_att].view(V3_B_MAX, n_att)
    y = y_view[:B]

    print(f"\nWKV phase output diffs (B={B}):")
    print(f"  y       max-abs vs ref: {maxabs(y, y_ref):.6e}")
    print(f"  aa      max-abs vs ref: {maxabs(aa, aa_after):.6e}")
    print(f"  bb      max-abs vs ref: {maxabs(bb, bb_after):.6e}")
    print(f"  pp      max-abs vs ref: {maxabs(pp, pp_after):.6e}")

    THRESHOLD = 1e-3
    diffs = [maxabs(y, y_ref), maxabs(aa, aa_after),
             maxabs(bb, bb_after), maxabs(pp, pp_after)]
    pass_ = all(d < THRESHOLD for d in diffs)
    print(f"\nverdict: {'PASS' if pass_ else 'FAIL'} (threshold {THRESHOLD:.0e})")
    sys.exit(0 if pass_ else 1)


if __name__ == "__main__":
    main()

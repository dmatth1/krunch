"""Verify v3 kernel's LN1 + premix phases (M2 milestone).

Calls rwkv4_layer_step_v3 with random weights + inputs at B=128.
Reads kx/vx/rx out of scratch buffer. Compares against pure-torch
reference implementation. Acceptable: max-abs < 5e-3 (fp16 noise).

State updates (att_xx) verified separately.
"""
import os, sys, time
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")

import torch
import krunch_ac_cuda


def maxabs(a, b): return (a.float() - b.float()).abs().max().item()


def ref_ln_premix(x, ln1_w, ln1_b, tm_k, tm_v, tm_r, att_xx_prev):
    """Pure-torch reference: LN1 (per-batch) + time-mix premix.
    Returns (kx, vx, rx, att_xx_new)."""
    # LN over last dim (channel)
    xx = torch.nn.functional.layer_norm(
        x.float(), (x.shape[-1],), weight=ln1_w.float(), bias=ln1_b.float())
    xx = xx.to(x.dtype)
    att_xx_v = att_xx_prev.to(torch.float32)
    xx_f = xx.float()
    tm_k_f = tm_k.float()
    tm_v_f = tm_v.float()
    tm_r_f = tm_r.float()
    kx = xx_f * tm_k_f + att_xx_v * (1 - tm_k_f)
    vx = xx_f * tm_v_f + att_xx_v * (1 - tm_v_f)
    rx = xx_f * tm_r_f + att_xx_v * (1 - tm_r_f)
    return kx.to(x.dtype), vx.to(x.dtype), rx.to(x.dtype), xx.to(x.dtype)


def main():
    device = "cuda"
    torch.manual_seed(0)
    B = 128
    C = 768
    n_att = 768
    n_ffn = 3072

    # Random inputs + weights matching RWKV-4 shapes
    x_in = torch.randn(B, C, dtype=torch.float16, device=device) * 0.1
    att_xx_prev = torch.randn(B, C, dtype=torch.float16, device=device) * 0.05

    ln1_w = torch.randn(C, dtype=torch.float16, device=device) * 0.5 + 1.0
    ln1_b = torch.randn(C, dtype=torch.float16, device=device) * 0.05
    tm_k = torch.rand(C, dtype=torch.float16, device=device)
    tm_v = torch.rand(C, dtype=torch.float16, device=device)
    tm_r = torch.rand(C, dtype=torch.float16, device=device)

    # Other weights (unused in M2 — pass dummies)
    def dummy(*shape):
        return torch.zeros(shape, dtype=torch.float16, device=device)

    ln2_w = dummy(C); ln2_b = dummy(C)
    ffn_tm_k = dummy(C); ffn_tm_r = dummy(C)
    Kw = dummy(C, n_att); Vw = dummy(C, n_att); Rw = dummy(C, C); Ow = dummy(n_att, C)
    ffn_Kw = dummy(C, n_ffn); ffn_Vw = dummy(n_ffn, C); ffn_Rw = dummy(C, C)
    time_decay = torch.zeros(n_att, dtype=torch.float32, device=device)
    time_first = torch.zeros(n_att, dtype=torch.float32, device=device)

    # State buffers
    att_xx = att_xx_prev.clone()
    aa = torch.zeros(B, n_att, dtype=torch.float32, device=device)
    bb = torch.zeros(B, n_att, dtype=torch.float32, device=device)
    pp = torch.full((B, n_att), -1e30, dtype=torch.float32, device=device)
    ffn_xx = torch.zeros(B, C, dtype=torch.float16, device=device)

    x_out = torch.zeros(B, C, dtype=torch.float16, device=device)

    # Allocate v3 scratch
    n_bytes = krunch_ac_cuda.v3_scratch_bytes(B)
    scratch = torch.zeros(n_bytes, dtype=torch.uint8, device=device)
    print(f"v3 scratch bytes (B={B}): {n_bytes}")

    # Launch v3
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

    # Compute reference
    ref_kx, ref_vx, ref_rx, ref_att_xx = ref_ln_premix(
        x_in, ln1_w, ln1_b, tm_k, tm_v, tm_r, att_xx_prev)

    # Read kx, vx, rx out of scratch
    # Layout: [ln_partials | ln_partial_sq | kx | vx | rx]
    # Sizes:  2 * V3_B_MAX * V3_GRID_DIM floats + 3 * V3_B_MAX * N_EMBD halves
    V3_B_MAX = 256
    V3_GRID_DIM = 6
    N_EMBD = 768
    kx_offset_floats = 2 * V3_B_MAX * V3_GRID_DIM
    kx_offset_bytes = kx_offset_floats * 4
    halves_per_buf = V3_B_MAX * N_EMBD

    scratch_view = scratch.view(torch.uint8)
    kx_bytes = scratch_view[kx_offset_bytes : kx_offset_bytes + halves_per_buf * 2]
    vx_bytes = scratch_view[kx_offset_bytes + halves_per_buf * 2 :
                            kx_offset_bytes + halves_per_buf * 4]
    rx_bytes = scratch_view[kx_offset_bytes + halves_per_buf * 4 :
                            kx_offset_bytes + halves_per_buf * 6]
    kx_full = kx_bytes.view(torch.float16).reshape(V3_B_MAX, N_EMBD)
    vx_full = vx_bytes.view(torch.float16).reshape(V3_B_MAX, N_EMBD)
    rx_full = rx_bytes.view(torch.float16).reshape(V3_B_MAX, N_EMBD)
    kx = kx_full[:B]
    vx = vx_full[:B]
    rx = rx_full[:B]

    print(f"\nLN1 + premix output diffs (B={B}):")
    print(f"  kx       max-abs vs torch ref: {maxabs(kx, ref_kx):.6e}")
    print(f"  vx       max-abs vs torch ref: {maxabs(vx, ref_vx):.6e}")
    print(f"  rx       max-abs vs torch ref: {maxabs(rx, ref_rx):.6e}")
    print(f"  att_xx   max-abs vs ref:        {maxabs(att_xx, ref_att_xx):.6e}")

    THRESHOLD = 5e-3  # fp16 noise band
    diffs = [maxabs(kx, ref_kx), maxabs(vx, ref_vx), maxabs(rx, ref_rx),
             maxabs(att_xx, ref_att_xx)]
    pass_ = all(d < THRESHOLD for d in diffs)
    print(f"\nverdict: {'PASS' if pass_ else 'FAIL'} (threshold {THRESHOLD:.0e})")
    sys.exit(0 if pass_ else 1)


if __name__ == "__main__":
    main()

"""Microbench: time one v3 layer step at B=128, single GPU layer.

Establishes scalar v3 baseline. Compare against cpp_path single-layer
estimate to know how much speedup WMMA upgrades must deliver.
"""
import os, sys, time
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
import torch
import krunch_ac_cuda


def main():
    device = "cuda"
    torch.manual_seed(0)
    B = 128
    C = 768
    n_att = 768
    n_ffn = 3072

    x_in = torch.randn(B, C, dtype=torch.float16, device=device) * 0.1
    att_xx = torch.randn(B, C, dtype=torch.float16, device=device) * 0.05
    ffn_xx = torch.randn(B, C, dtype=torch.float16, device=device) * 0.05
    aa = torch.randn(B, n_att, dtype=torch.float32, device=device) * 0.5
    bb = torch.randn(B, n_att, dtype=torch.float32, device=device).abs() + 0.1
    pp = torch.randn(B, n_att, dtype=torch.float32, device=device) * 0.5
    x_out = torch.zeros(B, C, dtype=torch.float16, device=device)

    ln1_w = torch.randn(C, dtype=torch.float16, device=device) * 0.5 + 1.0
    ln1_b = torch.randn(C, dtype=torch.float16, device=device) * 0.05
    tm_k = torch.rand(C, dtype=torch.float16, device=device)
    tm_v = torch.rand(C, dtype=torch.float16, device=device)
    tm_r = torch.rand(C, dtype=torch.float16, device=device)
    time_decay = -torch.rand(n_att, dtype=torch.float32, device=device) - 0.1
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

    n_bytes = krunch_ac_cuda.v3_scratch_bytes(B)
    scratch = torch.zeros(n_bytes, dtype=torch.uint8, device=device)

    def call_once():
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

    # Warmup
    for _ in range(3):
        call_once()
    torch.cuda.synchronize()

    # Time
    N = 20
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(N):
        call_once()
    end.record()
    torch.cuda.synchronize()
    ms_total = start.elapsed_time(end)
    ms_per = ms_total / N

    print(f"v3 single-layer step @ B={B}: {ms_per:.3f} ms/call ({ms_total:.1f} ms / {N} iters)")
    print(f"For 12-layer model, est wall: {ms_per * 12:.1f} ms/step")


if __name__ == "__main__":
    main()

"""Isolate where packed (T=2) and stepped (T=1 twice) diverge.

Runs ONE layer of the C++ RWKV-4 forward both ways with identical
inputs. Compares per-token output and post-call state (aa, bb, pp,
att_xx, ffn_xx) bit-exactly. First divergence pinpoints which op is
shape-dependent.
"""
import os
os.environ["KRUNCH_DETERMINISTIC_MATMUL"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"

import torch
import krunch_ac_cuda as M
# Importing rwkv.model with RWKV_CUDA_ON=1 builds + loads wkv_cuda,
# which registers torch.ops.rwkv.wkv_forward.
import rwkv.model  # noqa: F401

def maxabs(a, b):
    return (a.float() - b.float()).abs().max().item()

def main():
    device = "cuda"
    torch.manual_seed(123)
    C = 768
    n_att = 768
    n_ffn = 3072
    T = 32

    # Inputs
    x_pack = torch.randn(1, T, C, dtype=torch.float16, device=device) * 0.1
    x_step0 = x_pack[:, 0:1, :].contiguous()
    x_step1 = x_pack[:, 1:2, :].contiguous()

    # Weights
    def w(*shape, scale=0.05):
        return torch.randn(*shape, dtype=torch.float16, device=device) * scale
    ln1_w = torch.ones(C, dtype=torch.float16, device=device)
    ln1_b = torch.zeros(C, dtype=torch.float16, device=device)
    ln2_w = torch.ones(C, dtype=torch.float16, device=device)
    ln2_b = torch.zeros(C, dtype=torch.float16, device=device)
    tm_k = w(C); tm_v = w(C); tm_r = w(C)
    time_decay = torch.randn(n_att, dtype=torch.float32, device=device) * 0.1 - 1.0
    time_first = torch.randn(n_att, dtype=torch.float32, device=device) * 0.1
    Kw = w(C, n_att); Vw = w(C, n_att); Rw = w(C, C); Ow = w(n_att, C)
    ffn_tm_k = w(C); ffn_tm_r = w(C)
    ffn_Kw = w(C, n_ffn); ffn_Vw = w(n_ffn, C); ffn_Rw = w(C, C)

    # State init
    def state_init():
        return (
            torch.zeros(1, C, dtype=torch.float16, device=device),  # att_xx
            torch.zeros(1, n_att, dtype=torch.float32, device=device),  # aa
            torch.zeros(1, n_att, dtype=torch.float32, device=device),  # bb
            torch.full((1, n_att), -1e30, dtype=torch.float32, device=device),  # pp
            torch.zeros(1, C, dtype=torch.float16, device=device),  # ffn_xx
        )

    # Stepped: T calls of T=1
    sa_xx, sa, sb, sp, sf_xx = state_init()
    y_s_all = []
    for t in range(T):
        x_t = x_pack[:, t:t+1, :].contiguous()
        y_t = M.rwkv4_layer_step_cpp_t1(
            x_t, sa_xx, sa, sb, sp, sf_xx,
            ln1_w, ln1_b, tm_k, tm_v, tm_r,
            time_decay, time_first, Kw, Vw, Rw, Ow,
            ln2_w, ln2_b, ffn_tm_k, ffn_tm_r,
            ffn_Kw, ffn_Vw, ffn_Rw)
        y_s_all.append(y_t)
    y_s = torch.cat(y_s_all, dim=1)  # [1, T, C]
    s_final = (sa_xx.clone(), sa.clone(), sb.clone(), sp.clone(), sf_xx.clone())

    # Packed: one T=2 call
    pa_xx, pa, pb, pp_, pf_xx = state_init()
    y_p = M.rwkv4_layer_step_cpp(
        x_pack, pa_xx, pa, pb, pp_, pf_xx,
        ln1_w, ln1_b, tm_k, tm_v, tm_r,
        time_decay, time_first, Kw, Vw, Rw, Ow,
        ln2_w, ln2_b, ffn_tm_k, ffn_tm_r,
        ffn_Kw, ffn_Vw, ffn_Rw)
    p_final = (pa_xx.clone(), pa.clone(), pb.clone(), pp_.clone(), pf_xx.clone())

    # Compare per-timestep
    for t in range(T):
        d = maxabs(y_s[:, t:t+1, :], y_p[:, t:t+1, :])
        if d > 0:
            print(f"y t={t}: {d:.6e}")
    print(f"y total max: {maxabs(y_s, y_p):.6e}")
    names = ["att_xx", "aa", "bb", "pp", "ffn_xx"]
    for n, a, b in zip(names, s_final, p_final):
        d = maxabs(a, b)
        print(f"final {n:>8}: {d:.6e}")

if __name__ == "__main__":
    main()

"""Run the FULL 12-layer model with REAL weights both ways and compare
per-step output bit-exactly. Uses the same code as the e2e roundtrip
but skips the AC stage so we can pinpoint where state diverges.
"""
import os
# Inherit KRUNCH_DETERMINISTIC_MATMUL from caller; default off so we can
# probe whether cuBLAS-only is shape-stable at our layer shapes.
os.environ.setdefault("RWKV_CUDA_ON", "1")

import torch, torch.nn.functional as F
from krunch.inference import _load_rwkv, MODEL_PATH, BOS_TOKEN
import krunch_ac_cuda
import sys

N_LAYER = 12

def maxabs(a, b):
    return (a.float() - b.float()).abs().max().item()


def get_layer_weights(model, device):
    w = model.w
    layers = []
    def fix(t, dt=None):
        t = t.to(device).contiguous()
        if dt is not None and t.dtype != dt: t = t.to(dtype=dt).contiguous()
        return t
    for i in range(N_LAYER):
        bbb = f"blocks.{i}."; att = f"blocks.{i}.att."; ffn = f"blocks.{i}.ffn."
        layers.append([
            fix(w[bbb+'ln1.weight'], torch.float16), fix(w[bbb+'ln1.bias'], torch.float16),
            fix(w[att+'time_mix_k'].squeeze(), torch.float16),
            fix(w[att+'time_mix_v'].squeeze(), torch.float16),
            fix(w[att+'time_mix_r'].squeeze(), torch.float16),
            fix(w[att+'time_decay'], torch.float32),
            fix(w[att+'time_first'], torch.float32),
            fix(w[att+'key.weight'], torch.float16),
            fix(w[att+'value.weight'], torch.float16),
            fix(w[att+'receptance.weight'], torch.float16),
            fix(w[att+'output.weight'], torch.float16),
            fix(w[bbb+'ln2.weight'], torch.float16), fix(w[bbb+'ln2.bias'], torch.float16),
            fix(w[ffn+'time_mix_k'].squeeze(), torch.float16),
            fix(w[ffn+'time_mix_r'].squeeze(), torch.float16),
            fix(w[ffn+'key.weight'], torch.float16),
            fix(w[ffn+'value.weight'], torch.float16),
            fix(w[ffn+'receptance.weight'], torch.float16),
        ])
    return (layers,
            fix(w['emb.weight'], torch.float16),
            fix(w['ln_out.weight'], torch.float16),
            fix(w['ln_out.bias'], torch.float16),
            fix(w['head.weight'], torch.float16))


def fresh_state(device, n_embd, n_att):
    return ([torch.zeros(1, n_embd, dtype=torch.float16, device=device) for _ in range(N_LAYER)],
            [torch.zeros(1, n_att, dtype=torch.float32, device=device) for _ in range(N_LAYER)],
            [torch.zeros(1, n_att, dtype=torch.float32, device=device) for _ in range(N_LAYER)],
            [torch.full((1, n_att), -1e30, dtype=torch.float32, device=device) for _ in range(N_LAYER)],
            [torch.zeros(1, n_embd, dtype=torch.float16, device=device) for _ in range(N_LAYER)])


def main():
    RWKV = _load_rwkv()
    model = RWKV(model=str(MODEL_PATH).removesuffix(".pth"), strategy="cuda fp16", verbose=False)
    device = "cuda"
    n_embd = model.args.n_embd
    n_att = model.args.n_att
    layers, emb_w, ln_out_w, ln_out_b, head_w = get_layer_weights(model, device)

    inputs = [BOS_TOKEN, 510, 3158, 8516, 30013, 27287, 689, 253, 22658, 4370,
              15, 831, 310, 247, 1071, 273, 253, 36407, 3204, 11454,
              13800, 47618, 15, 844, 403, 3515, 247, 12243, 14, 42611, 11940]
    T = len(inputs)
    idx = torch.tensor(inputs, dtype=torch.long, device=device)

    # ============== PACKED (compress) ==============
    state_p = fresh_state(device, n_embd, n_att)
    x = emb_w[idx].view(1, T, n_embd).contiguous()
    for i in range(N_LAYER):
        x = krunch_ac_cuda.rwkv4_layer_step_cpp(
            x.contiguous(),
            state_p[0][i], state_p[1][i], state_p[2][i], state_p[3][i], state_p[4][i],
            *layers[i],
        )
    x_packed_layer_out = x.clone()  # [1, T, n_embd]

    # ============== STEPPED (decompress) ==============
    state_s = fresh_state(device, n_embd, n_att)
    x_stepped_per_t = []
    for t in range(T):
        xt = emb_w[idx[t]].view(1, 1, n_embd).contiguous()
        for i in range(N_LAYER):
            xt = krunch_ac_cuda.rwkv4_layer_step_cpp_t1(
                xt.contiguous(),
                state_s[0][i], state_s[1][i], state_s[2][i], state_s[3][i], state_s[4][i],
                *layers[i],
            )
        x_stepped_per_t.append(xt)
    x_stepped_layer_out = torch.cat(x_stepped_per_t, dim=1)

    # Find first divergent timestep
    print(f"T={T}, comparing layer-stack output [1, T, {n_embd}]")
    first_bad = -1
    for t in range(T):
        d = maxabs(x_packed_layer_out[:, t:t+1, :], x_stepped_layer_out[:, t:t+1, :])
        if d > 0:
            print(f"  t={t}: max_abs={d:.6e}")
            if first_bad < 0:
                first_bad = t
                if t > 5: break  # stop after a few
    if first_bad < 0:
        print("  ALL TIMESTEPS BIT-IDENTICAL.")
    else:
        print(f"  First divergence at t={first_bad}.")

    # Now check the post-layer pipeline that the AC roundtrip uses.
    print("\n=== Post-layer pipeline diff ===")
    # Encoder: batched ln_out + head
    x_flat = x_packed_layer_out.view(T, n_embd)
    xn_batched = F.layer_norm(x_flat, (n_embd,), weight=ln_out_w, bias=ln_out_b)
    logits_batched = krunch_ac_cuda.det_matmul(xn_batched.contiguous(), head_w.contiguous())

    # Decoder: per-row ln_out + head
    xn_per_row_list = []
    logits_per_row_list = []
    for t in range(T):
        xn_t = F.layer_norm(x_stepped_layer_out[:, t, :].contiguous(), (n_embd,),
                             weight=ln_out_w, bias=ln_out_b)
        xn_per_row_list.append(xn_t)
        lg_t = krunch_ac_cuda.det_matmul(xn_t.contiguous(), head_w.contiguous())
        logits_per_row_list.append(lg_t)
    xn_per_row = torch.cat(xn_per_row_list, dim=0)
    logits_per_row = torch.cat(logits_per_row_list, dim=0)

    print(f"x_packed_vs_stepped layer-out diff: {maxabs(x_flat, x_stepped_layer_out.view(T, n_embd)):.6e}")
    print(f"ln_out batched vs per-row diff:     {maxabs(xn_batched, xn_per_row):.6e}")
    print(f"logits batched vs per-row diff:     {maxabs(logits_batched, logits_per_row):.6e}")
    # Find argmax divergence (proxies AC bin)
    am_b = logits_batched.argmax(dim=-1)
    am_r = logits_per_row.argmax(dim=-1)
    n_diff_argmax = (am_b != am_r).sum().item()
    print(f"argmax mismatches: {n_diff_argmax}/{T}")

    # CDF check: encoder runs probs_to_cdf_gpu on [T, V], decoder on [1, V].
    from krunch_ac.gpu_encode import probs_to_cdf_gpu
    probs_batched = torch.softmax(logits_batched.float(), dim=-1)
    cdfs_batched = probs_to_cdf_gpu(probs_batched).contiguous()  # [T, V+1]
    cdfs_per_row_list = []
    for t in range(T):
        cdfs_per_row_list.append(probs_to_cdf_gpu(probs_batched[t:t+1]).contiguous())
    cdfs_per_row = torch.cat(cdfs_per_row_list, dim=0)
    diff_cdf = (cdfs_batched.long() - cdfs_per_row.long()).abs().max().item()
    print(f"CDF batched vs per-row max int diff: {diff_cdf}")
    n_rows_diff = ((cdfs_batched != cdfs_per_row).any(dim=-1)).sum().item()
    print(f"CDF rows differing: {n_rows_diff}/{T}")

    # Compare final per-layer state
    print("\nFinal per-layer state diff (packed vs stepped):")
    for i in range(N_LAYER):
        d_axx = maxabs(state_p[0][i], state_s[0][i])
        d_aa = maxabs(state_p[1][i], state_s[1][i])
        d_bb = maxabs(state_p[2][i], state_s[2][i])
        d_pp = maxabs(state_p[3][i], state_s[3][i])
        d_fxx = maxabs(state_p[4][i], state_s[4][i])
        if max(d_axx, d_aa, d_bb, d_pp, d_fxx) > 0:
            print(f"  L{i}: att_xx={d_axx:.3e} aa={d_aa:.3e} bb={d_bb:.3e} pp={d_pp:.3e} ffn_xx={d_fxx:.3e}")

if __name__ == "__main__":
    main()

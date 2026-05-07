"""Verify cross-chunk batched stepped forward bit-equality.

Run B=3 independent token sequences two ways:
  A) Sequential: each chunk decoded independently with B=1 calls
  B) Batched: all B chunks stepped together with one [B, 1, C] call

If layer step + state evolution is correctly per-chunk-independent,
A and B produce IDENTICAL per-chunk outputs and state.

Uses real RWKV-4-Pile-169M weights so we hit realistic numerics.
"""
import os
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import torch
import krunch_ac_cuda
from krunch.inference import _load_rwkv, MODEL_PATH, BOS_TOKEN
from krunch import cpp_path

N_LAYER = 12
N_STEPS = 16


def maxabs(a, b): return (a.float() - b.float()).abs().max().item()


def main():
    RWKV = _load_rwkv()
    model = RWKV(model=str(MODEL_PATH).removesuffix(".pth"),
                 strategy="cuda fp16", verbose=False)
    weights = cpp_path.init_weights(model, "cuda")
    n_embd = weights["n_embd"]
    n_att = weights["n_att"]
    layers = weights["layers"]
    emb_w = weights["emb_w"]

    # 3 independent token sequences
    seqs = [
        [BOS_TOKEN, 510, 3158, 8516, 30013, 27287, 689, 253, 22658, 4370,
         15, 831, 310, 247, 1071, 273][:N_STEPS],
        [BOS_TOKEN, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
         100, 200, 300, 400, 500, 600][:N_STEPS],
        [BOS_TOKEN, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15][:N_STEPS],
    ]
    B = len(seqs)
    T = N_STEPS

    # ============== A: Sequential, B independent runs ==============
    seq_outs = []  # list of [T, n_embd] per chunk
    seq_states = []  # final state per chunk
    for b in range(B):
        state = cpp_path.fresh_state(weights)
        outs = []
        for t in range(T):
            tok = seqs[b][t]
            x = emb_w[tok].view(1, 1, n_embd).contiguous()
            for i in range(N_LAYER):
                x = krunch_ac_cuda.rwkv4_layer_step_cpp(
                    x.contiguous(),
                    state[0][i], state[1][i], state[2][i], state[3][i], state[4][i],
                    *layers[i],
                )
            outs.append(x.view(n_embd).clone())
        seq_outs.append(torch.stack(outs))  # [T, n_embd]
        seq_states.append([[s.clone() for s in state[k]] for k in range(5)])

    # ============== B: Batched B-way stepped ==============
    # State tensors become [B, ...]
    bat_att_xx = [torch.zeros(B, n_embd, dtype=torch.float16, device="cuda") for _ in range(N_LAYER)]
    bat_aa = [torch.zeros(B, n_att, dtype=torch.float32, device="cuda") for _ in range(N_LAYER)]
    bat_bb = [torch.zeros(B, n_att, dtype=torch.float32, device="cuda") for _ in range(N_LAYER)]
    bat_pp = [torch.full((B, n_att), -1e30, dtype=torch.float32, device="cuda") for _ in range(N_LAYER)]
    bat_ffn_xx = [torch.zeros(B, n_embd, dtype=torch.float16, device="cuda") for _ in range(N_LAYER)]

    bat_outs = []
    for t in range(T):
        # x: [B, 1, n_embd]  — embeddings of seqs[*][t]
        toks = torch.tensor([seqs[b][t] for b in range(B)], dtype=torch.long, device="cuda")
        x = emb_w[toks].view(B, 1, n_embd).contiguous()
        for i in range(N_LAYER):
            x = krunch_ac_cuda.rwkv4_layer_step_cpp(
                x.contiguous(),
                bat_att_xx[i], bat_aa[i], bat_bb[i], bat_pp[i], bat_ffn_xx[i],
                *layers[i],
            )
        bat_outs.append(x.view(B, n_embd).clone())  # [B, n_embd]
    bat_outs = torch.stack(bat_outs, dim=1)  # [B, T, n_embd]

    # Compare per chunk
    print(f"B={B} T={T}: comparing batched vs sequential outputs and state\n")
    all_ok = True
    for b in range(B):
        out_diff = maxabs(bat_outs[b], seq_outs[b])
        ax_diff = maxabs(bat_att_xx[N_LAYER-1][b:b+1], seq_states[b][0][N_LAYER-1])
        aa_diff = maxabs(bat_aa[N_LAYER-1][b:b+1], seq_states[b][1][N_LAYER-1])
        bb_diff = maxabs(bat_bb[N_LAYER-1][b:b+1], seq_states[b][2][N_LAYER-1])
        pp_diff = maxabs(bat_pp[N_LAYER-1][b:b+1], seq_states[b][3][N_LAYER-1])
        fx_diff = maxabs(bat_ffn_xx[N_LAYER-1][b:b+1], seq_states[b][4][N_LAYER-1])
        ok = (out_diff == 0 and ax_diff == 0 and aa_diff == 0
              and bb_diff == 0 and pp_diff == 0 and fx_diff == 0)
        all_ok &= ok
        print(f"chunk {b}: out_diff={out_diff:.3e} att_xx={ax_diff:.3e} "
              f"aa={aa_diff:.3e} bb={bb_diff:.3e} pp={pp_diff:.3e} "
              f"ffn_xx={fx_diff:.3e}  {'PASS' if ok else 'FAIL'}")
    print()
    print("ALL PASS — batched == sequential bit-identically" if all_ok
          else "SOMETHING DIFFERS — fix before relying on cross-chunk batched decode")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

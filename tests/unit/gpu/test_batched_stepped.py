"""Cross-chunk batched stepped forward must match B=1 sequential bit-exactly.

Run B=3 independent token sequences two ways:
  A) Sequential — each chunk decoded independently with B=1 calls
  B) Batched    — all B chunks stepped together with one [B, 1, C] call

If layer step + state evolution is correctly per-chunk-independent,
A and B produce IDENTICAL per-chunk outputs and state. This is the
invariant `decompress_chunks_batched` depends on.

Uses real RWKV-4-Pile-169M weights so we hit production numerics.
"""
import os
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import pytest
import torch
import krunch_ac_cuda
from krunch.inference import _load_rwkv, MODEL_PATH, BOS_TOKEN
from krunch import cpp_path

N_LAYER = 12
N_STEPS = 16


def _max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


@pytest.fixture(scope="module")
def model_weights():
    RWKV = _load_rwkv()
    model = RWKV(model=str(MODEL_PATH).removesuffix(".pth"),
                 strategy="cuda fp16", verbose=False)
    return cpp_path.init_weights(model, "cuda")


def _run_sequential(weights, seqs):
    n_embd = weights["n_embd"]
    layers = weights["layers"]
    emb_w = weights["emb_w"]
    seq_outs, seq_states = [], []
    for tokens in seqs:
        state = cpp_path.fresh_state(weights)
        outs = []
        for tok in tokens:
            x = emb_w[tok].view(1, 1, n_embd).contiguous()
            for i in range(N_LAYER):
                x = krunch_ac_cuda.rwkv4_layer_step_cpp(
                    x.contiguous(),
                    state[0][i], state[1][i], state[2][i],
                    state[3][i], state[4][i],
                    *layers[i],
                )
            outs.append(x.view(n_embd).clone())
        seq_outs.append(torch.stack(outs))
        seq_states.append([[s.clone() for s in state[k]] for k in range(5)])
    return seq_outs, seq_states


def _run_batched(weights, seqs):
    n_embd = weights["n_embd"]
    n_att = weights["n_att"]
    layers = weights["layers"]
    emb_w = weights["emb_w"]
    B, T = len(seqs), len(seqs[0])
    att_xx = [torch.zeros(B, n_embd, dtype=torch.float16, device="cuda") for _ in range(N_LAYER)]
    aa = [torch.zeros(B, n_att, dtype=torch.float32, device="cuda") for _ in range(N_LAYER)]
    bb = [torch.zeros(B, n_att, dtype=torch.float32, device="cuda") for _ in range(N_LAYER)]
    pp = [torch.full((B, n_att), -1e30, dtype=torch.float32, device="cuda") for _ in range(N_LAYER)]
    ffn_xx = [torch.zeros(B, n_embd, dtype=torch.float16, device="cuda") for _ in range(N_LAYER)]
    bat_outs = []
    for t in range(T):
        toks = torch.tensor([seqs[b][t] for b in range(B)],
                            dtype=torch.long, device="cuda")
        x = emb_w[toks].view(B, 1, n_embd).contiguous()
        for i in range(N_LAYER):
            x = krunch_ac_cuda.rwkv4_layer_step_cpp(
                x.contiguous(),
                att_xx[i], aa[i], bb[i], pp[i], ffn_xx[i],
                *layers[i],
            )
        bat_outs.append(x.view(B, n_embd).clone())
    bat_outs = torch.stack(bat_outs, dim=1)
    return bat_outs, (att_xx, aa, bb, pp, ffn_xx)


SEQS = [
    [BOS_TOKEN, 510, 3158, 8516, 30013, 27287, 689, 253, 22658, 4370,
     15, 831, 310, 247, 1071, 273][:N_STEPS],
    [BOS_TOKEN, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
     100, 200, 300, 400, 500, 600][:N_STEPS],
    [BOS_TOKEN, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15][:N_STEPS],
]


@pytest.mark.parametrize("b", [0, 1, 2])
def test_batched_matches_sequential(model_weights, b):
    seq_outs, seq_states = _run_sequential(model_weights, SEQS)
    bat_outs, (att_xx, aa, bb, pp, ffn_xx) = _run_batched(model_weights, SEQS)

    assert _max_abs(bat_outs[b], seq_outs[b]) == 0.0, "output diverged"
    assert _max_abs(att_xx[N_LAYER - 1][b:b + 1],
                    seq_states[b][0][N_LAYER - 1]) == 0.0, "att_xx diverged"
    assert _max_abs(aa[N_LAYER - 1][b:b + 1],
                    seq_states[b][1][N_LAYER - 1]) == 0.0, "aa diverged"
    assert _max_abs(bb[N_LAYER - 1][b:b + 1],
                    seq_states[b][2][N_LAYER - 1]) == 0.0, "bb diverged"
    assert _max_abs(pp[N_LAYER - 1][b:b + 1],
                    seq_states[b][3][N_LAYER - 1]) == 0.0, "pp diverged"
    assert _max_abs(ffn_xx[N_LAYER - 1][b:b + 1],
                    seq_states[b][4][N_LAYER - 1]) == 0.0, "ffn_xx diverged"

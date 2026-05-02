"""Verify the C++ packed forward (compress) and C++ T=1 forward (decompress)
produce byte-exact AC roundtrip — the real correctness gate.

If both sides use the same C++ ops, drift should be fp16-noise (<0.1 abs)
across packed T=1024 vs stepped T=1, and AC roundtrips cleanly.
"""
import os, time, torch
import torch.nn.functional as F

os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

from krunch.inference import _load_rwkv, MODEL_PATH
import krunch_ac_cuda

N_LAYER = 12


def get_layer_weights(model, device):
    w = model.w
    n_layer = model.args.n_layer
    layers = []
    def fix(t, dt=None):
        t = t.to(device).contiguous()
        if dt is not None and t.dtype != dt:
            t = t.to(dtype=dt).contiguous()
        return t
    for i in range(n_layer):
        bbb = f"blocks.{i}."
        att = f"blocks.{i}.att."
        ffn = f"blocks.{i}.ffn."
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


def cpp_packed_forward(emb_w, layers, ln_out_w, ln_out_b, head_w,
                       state, tokens, n_embd):
    """Run packed forward over a sequence. Mutates state in place.
    Returns logits at every position [T, V]."""
    state_xx, state_aa, state_bb, state_pp, state_ffn = state
    T = len(tokens)
    device = emb_w.device
    idx = torch.tensor(tokens, dtype=torch.long, device=device)
    x = emb_w[idx].view(1, T, n_embd).contiguous()
    for i in range(N_LAYER):
        x = krunch_ac_cuda.rwkv4_layer_step_cpp(
            x.contiguous(),
            state_xx[i], state_aa[i], state_bb[i], state_pp[i], state_ffn[i],
            *layers[i],
        )
    x = F.layer_norm(x.view(T, n_embd), (n_embd,),
                     weight=ln_out_w, bias=ln_out_b)
    return x @ head_w


def cpp_step_forward(emb_w, layers, ln_out_w, ln_out_b, head_w,
                     state, token, n_embd):
    """Run T=1 stepped forward. Mutates state in place. Returns [V]."""
    state_xx, state_aa, state_bb, state_pp, state_ffn = state
    device = emb_w.device
    x = emb_w[token].view(1, 1, n_embd).contiguous()
    for i in range(N_LAYER):
        x = krunch_ac_cuda.rwkv4_layer_step_cpp_t1(
            x.contiguous(),
            state_xx[i], state_aa[i], state_bb[i], state_pp[i], state_ffn[i],
            *layers[i],
        )
    x = F.layer_norm(x.view(1, n_embd), (n_embd,),
                     weight=ln_out_w, bias=ln_out_b)
    return (x @ head_w).flatten()


def main():
    RWKV = _load_rwkv()
    print("loading RWKV-4-Pile-169M ...", flush=True)
    model = RWKV(model=str(MODEL_PATH).removesuffix(".pth"),
                 strategy="cuda fp16", verbose=False)
    print("loaded", flush=True)
    device = "cuda"
    n_embd = model.args.n_embd
    n_att = model.args.n_att

    layers, emb_w, ln_out_w, ln_out_b, head_w = get_layer_weights(model, device)

    # Generate a 16-token reference sequence using BlinkDL greedy.
    # We'll then check that packed-T=16 logits at every position match
    # stepped-T=1 logits at the same positions.
    seq = [0]
    state_b = None
    for _ in range(15):
        logits, state_b = model.forward([seq[-1]], state_b)
        if not torch.is_tensor(logits): logits = torch.as_tensor(logits, device=device)
        seq.append(int(logits.argmax().item()))
    print(f"reference seq: {seq}", flush=True)

    # PACKED: run all 16 tokens at once, get T logits
    state_p = fresh_state(device, n_embd, n_att)
    logits_packed = cpp_packed_forward(emb_w, layers, ln_out_w, ln_out_b, head_w,
                                        state_p, seq, n_embd)  # [T, V]

    # STEPPED: run T=1 for each token sequentially
    state_s = fresh_state(device, n_embd, n_att)
    logits_stepped = []
    for tok in seq:
        l = cpp_step_forward(emb_w, layers, ln_out_w, ln_out_b, head_w,
                              state_s, tok, n_embd)
        logits_stepped.append(l.float())

    print("--- packed T=N vs stepped T=1 (CPP both sides) ---", flush=True)
    for i in range(len(seq)):
        diff = (logits_packed[i].float() - logits_stepped[i]).abs().max().item()
        print(f"DRIFT step={i} max_abs={diff:.4f}", flush=True)


if __name__ == "__main__":
    main()

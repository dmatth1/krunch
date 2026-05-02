"""Bench C++ packed forward (T=1024) vs BlinkDL packed.

Compress's hot path is packed forward over SEQ_BATCH=1024 tokens. To not
regress compress, the C++ packed must match or beat BlinkDL's 13.5 ms/call
on T4 (measured earlier).
"""
import os, time, torch
import torch.nn.functional as F

os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

from krunch.inference import _load_rwkv, MODEL_PATH
import krunch_ac_cuda

N_LAYER = 12
T = 1024
N_WARMUP = 8
N_ITERS = 50


def bench(label, fn, n=N_ITERS):
    for _ in range(N_WARMUP): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    print(f"BENCH {label} per-call-ms={dt*1000:.3f}", flush=True)
    return dt


def get_layer_weights(model, device):
    w = model.w
    layers = []
    def fix(t, dt=None):
        t = t.to(device).contiguous()
        if dt is not None and t.dtype != dt: t = t.to(dtype=dt).contiguous()
        return t
    for i in range(N_LAYER):
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

    # State for compress: B=1
    def fresh_state():
        return ([torch.zeros(1, n_embd, dtype=torch.float16, device=device) for _ in range(N_LAYER)],
                [torch.zeros(1, n_att, dtype=torch.float32, device=device) for _ in range(N_LAYER)],
                [torch.zeros(1, n_att, dtype=torch.float32, device=device) for _ in range(N_LAYER)],
                [torch.full((1, n_att), -1e30, dtype=torch.float32, device=device) for _ in range(N_LAYER)],
                [torch.zeros(1, n_embd, dtype=torch.float16, device=device) for _ in range(N_LAYER)])

    # Bench: C++ packed T=1024
    state_p = fresh_state()
    tokens = torch.zeros(T, dtype=torch.long, device=device)
    def fn_cpp_packed():
        x = emb_w[tokens].view(1, T, n_embd).contiguous()
        for i in range(N_LAYER):
            x = krunch_ac_cuda.rwkv4_layer_step_cpp(
                x.contiguous(),
                state_p[0][i], state_p[1][i], state_p[2][i], state_p[3][i], state_p[4][i],
                *layers[i],
            )
        x = F.layer_norm(x.view(T, n_embd), (n_embd,), weight=ln_out_w, bias=ln_out_b)
        return x @ head_w
    bench("cpp_packed_t1024", fn_cpp_packed)

    # Bench: BlinkDL packed (~T=1024)
    state_b = None
    tok_list = list(range(T))
    def fn_blinkdl_packed():
        nonlocal state_b
        state_b = None  # reset for a fair per-call comparison
        logits, state_b = model.forward(tok_list, state_b)
    bench("blinkdl_packed_t1024", fn_blinkdl_packed)


if __name__ == "__main__":
    main()

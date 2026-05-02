"""C++ layer orchestration: correctness vs forward_batched, speed vs BlinkDL.

Calls `krunch_ac_cuda.rwkv4_layer_step_cpp_t1` once per layer × 12 layers
to do a full single-token forward. Compares logits to BlinkDL and bench
the per-token wall.
"""
import os, time, torch

os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

from krunch.inference import _load_rwkv, MODEL_PATH
import krunch_ac_cuda

N_WARMUP = 16
N_ITERS = 200


def main():
    RWKV = _load_rwkv()
    print("loading RWKV-4-Pile-169M ...", flush=True)
    model = RWKV(model=str(MODEL_PATH).removesuffix(".pth"),
                 strategy="cuda fp16", verbose=False)
    print("loaded", flush=True)
    device = "cuda"

    args = model.args
    n_layer = args.n_layer
    n_embd = args.n_embd
    n_att = args.n_att
    w = model.w

    # Pre-extract weights per layer. Force everything to CUDA — rwkv's
    # strategy may leave some scalar params on CPU.
    def fix(t, dtype=None):
        t = t.to(device).contiguous()
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype=dtype).contiguous()
        return t
    layer_w = []
    for i in range(n_layer):
        bbb = f"blocks.{i}."
        att = f"blocks.{i}.att."
        ffn = f"blocks.{i}.ffn."
        layer_w.append([
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
    # Also fix ln_out + head + emb
    ln_out_w = fix(w['ln_out.weight'], torch.float16)
    ln_out_b = fix(w['ln_out.bias'], torch.float16)
    emb_w = fix(w['emb.weight'], torch.float16)
    head_w = fix(w['head.weight'], torch.float16)

    # Per-layer state (B=1)
    def fresh_state():
        return ([torch.zeros(1, n_embd, dtype=torch.float16, device=device) for _ in range(n_layer)],
                [torch.zeros(1, n_att, dtype=torch.float32, device=device) for _ in range(n_layer)],
                [torch.zeros(1, n_att, dtype=torch.float32, device=device) for _ in range(n_layer)],
                [torch.full((1, n_att), -1e30, dtype=torch.float32, device=device) for _ in range(n_layer)],
                [torch.zeros(1, n_embd, dtype=torch.float16, device=device) for _ in range(n_layer)])

    state_xx, state_aa, state_bb, state_pp, state_ffn = fresh_state()

    def cpp_full_forward(token_id):
        x = emb_w[token_id].view(1, 1, n_embd).contiguous()
        for i in range(n_layer):
            x = krunch_ac_cuda.rwkv4_layer_step_cpp_t1(
                x.contiguous(),
                state_xx[i], state_aa[i], state_bb[i], state_pp[i], state_ffn[i],
                *layer_w[i],
            )
        x = torch.nn.functional.layer_norm(
            x.view(1, n_embd), (n_embd,),
            weight=ln_out_w, bias=ln_out_b)
        return x @ head_w

    # Correctness vs BlinkDL
    print("--- correctness ---", flush=True)
    state_blink = None
    for tok in [0, 100, 1000, 5, 42, 9999]:
        # Reset our state to zero before each tok for clean compare
        sa, sb, sc, sd, se = fresh_state()
        for i in range(n_layer):
            state_xx[i].copy_(sa[i]); state_aa[i].copy_(sb[i])
            state_bb[i].copy_(sc[i]); state_pp[i].copy_(sd[i]); state_ffn[i].copy_(se[i])
        logits_cpp = cpp_full_forward(tok).flatten().float()

        state_blink = None  # Reset BlinkDL state
        logits_blink, state_blink = model.forward([tok], state_blink)
        if not torch.is_tensor(logits_blink):
            logits_blink = torch.as_tensor(logits_blink, device=device)
        logits_blink = logits_blink.flatten().float()

        diff = (logits_cpp - logits_blink).abs().max().item()
        print(f"DIFF tok={tok} max_abs={diff:.4f}", flush=True)

    # Bench
    print("--- bench ---", flush=True)
    state_xx, state_aa, state_bb, state_pp, state_ffn = fresh_state()
    last = 0
    def fn_cpp():
        nonlocal last
        logits = cpp_full_forward(last)
        last = int(logits.argmax().item())
    for _ in range(N_WARMUP): fn_cpp()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS): fn_cpp()
    torch.cuda.synchronize()
    dt_cpp = (time.perf_counter() - t0) / N_ITERS
    print(f"BENCH cpp_full per-token-ms={dt_cpp*1000:.3f}", flush=True)

    state_b = None
    last = 0
    def fn_blink():
        nonlocal state_b, last
        logits, state_b = model.forward([last], state_b)
        last = int(logits.argmax().item()) if torch.is_tensor(logits) else 0
    for _ in range(N_WARMUP): fn_blink()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS): fn_blink()
    torch.cuda.synchronize()
    dt_b = (time.perf_counter() - t0) / N_ITERS
    print(f"BENCH blinkdl per-token-ms={dt_b*1000:.3f}", flush=True)
    print(f"SPEEDUP cpp/blinkdl = {dt_b/dt_cpp:.2f}x", flush=True)


if __name__ == "__main__":
    main()

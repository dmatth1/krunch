"""Speed bench: 12-layer call of `rwkv4_layer_step_kernel` vs BlinkDL per-token.

Calls the kernel 12 times per token (one per layer) using the real model
weights. Final layer norm + head GEMV are skipped — they're a small fixed
overhead not on the critical path. Goal: prove single-process per-token
wall is materially faster than BlinkDL's 7.5 ms/token.
"""
import os
import time
import torch

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
    n_embd = args.n_embd
    n_att  = args.n_att
    n_layer = args.n_layer
    w = model.w

    # Pre-collect weights and contiguous-ify ONCE (kernel requires contiguous).
    layer_weights = []
    for i in range(n_layer):
        bbb = f"blocks.{i}."
        att = f"blocks.{i}.att."
        ffn = f"blocks.{i}.ffn."
        layer_weights.append((
            w[bbb+'ln1.weight'].contiguous(), w[bbb+'ln1.bias'].contiguous(),
            w[att+'time_mix_k'].squeeze().contiguous(),
            w[att+'time_mix_v'].squeeze().contiguous(),
            w[att+'time_mix_r'].squeeze().contiguous(),
            w[att+'time_decay'].contiguous(), w[att+'time_first'].contiguous(),
            w[att+'key.weight'].contiguous(),
            w[att+'value.weight'].contiguous(),
            w[att+'receptance.weight'].contiguous(),
            w[att+'output.weight'].contiguous(),
            w[bbb+'ln2.weight'].contiguous(), w[bbb+'ln2.bias'].contiguous(),
            w[ffn+'time_mix_k'].squeeze().contiguous(),
            w[ffn+'time_mix_r'].squeeze().contiguous(),
            w[ffn+'key.weight'].contiguous(),
            w[ffn+'value.weight'].contiguous(),
            w[ffn+'receptance.weight'].contiguous(),
        ))

    # State per layer
    state_xx  = [torch.zeros(n_embd, dtype=torch.float16, device=device) for _ in range(n_layer)]
    state_aa  = [torch.zeros(n_att, dtype=torch.float32, device=device) for _ in range(n_layer)]
    state_bb  = [torch.zeros(n_att, dtype=torch.float32, device=device) for _ in range(n_layer)]
    state_pp  = [torch.full((n_att,), -1e30, dtype=torch.float32, device=device)
                 for _ in range(n_layer)]
    state_ffn = [torch.zeros(n_embd, dtype=torch.float16, device=device) for _ in range(n_layer)]

    # Pre-allocated x ping-pong buffers (kernel reads x_in, writes x_out)
    x_a = torch.zeros(n_embd, dtype=torch.float16, device=device)
    x_b = torch.zeros(n_embd, dtype=torch.float16, device=device)

    def step():
        # Embedding lookup → x_a (use BOS=0 for bench)
        x_a.copy_(w['emb.weight'][0])
        x_in = x_a
        x_out = x_b
        for i in range(n_layer):
            (ln1_w, ln1_b, tm_k, tm_v, tm_r, td, tf,
             Kw, Vw, Rw, Ow,
             ln2_w, ln2_b, ffk, ffr, fK, fV, fR) = layer_weights[i]
            krunch_ac_cuda.rwkv4_layer_step(
                x_in, x_out,
                state_xx[i], state_aa[i], state_bb[i], state_pp[i], state_ffn[i],
                ln1_w, ln1_b, tm_k, tm_v, tm_r, td, tf,
                Kw, Vw, Rw, Ow,
                ln2_w, ln2_b, ffk, ffr, fK, fV, fR,
            )
            x_in, x_out = x_out, x_in

    for _ in range(N_WARMUP):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / N_ITERS
    print(f"BENCH kernel_12layers per-token-ms={dt*1000:.3f}", flush=True)

    # Compare to BlinkDL per-token
    state = None
    last = 0
    def fn_blinkdl():
        nonlocal state, last
        logits, state = model.forward([last], state)
        last = int(logits.argmax().item()) if torch.is_tensor(logits) else 0
    for _ in range(N_WARMUP):
        fn_blinkdl()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        fn_blinkdl()
    torch.cuda.synchronize()
    dt_b = (time.perf_counter() - t0) / N_ITERS
    print(f"BENCH blinkdl_per_token per-token-ms={dt_b*1000:.3f}", flush=True)
    print(f"SPEEDUP kernel/blinkdl = {dt_b/dt:.2f}x", flush=True)


if __name__ == "__main__":
    main()

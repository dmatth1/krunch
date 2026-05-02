"""Profile compress forward_packed_window: breaks the per-window wall into
embedding / layer loop / final LN / head matmul components.

Decision-feeder for T3.2 lever (multi-warp det_matmul_tc). If head matmul
≥30% of forward wall, build the multi-warp kernel. If <10%, skip.

Run on real RWKV-4-Pile-169M weights, T=1024 (production SEQ_BATCH).
Reports per-component ms/call + share of forward wall.
"""
import os, sys, time
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import torch
import torch.nn.functional as F
import krunch_ac_cuda
import rwkv.model

from krunch.inference import _load_rwkv  # noqa
from krunch import cpp_path

N_LAYER = 12
T = 1024
N_WARMUP = 5
N_ITERS = 30


def bench(label, fn, total=None):
    for _ in range(N_WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        fn()
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) / N_ITERS * 1000
    share = f" ({100*dt_ms/total:.1f}% of forward)" if total else ""
    print(f"  {label:30s}  {dt_ms:7.3f} ms/call{share}", flush=True)
    return dt_ms


def main():
    device = "cuda"
    print("Loading RWKV-4-Pile-169M...", flush=True)
    from krunch.inference import MODEL_PATH
    RWKV = _load_rwkv()
    model_path_no_ext = str(MODEL_PATH).removesuffix(".pth")
    model = RWKV(model=model_path_no_ext, strategy="cuda fp16")
    weights = cpp_path.init_weights(model, device)
    n_embd = weights["n_embd"]
    layers = weights["layers"]
    emb_w = weights["emb_w"]
    ln_out_w = weights["ln_out_w"]
    ln_out_b = weights["ln_out_b"]
    head_w = weights["head_w"]

    # Realistic input: random tokens (token IDs don't change matmul shapes).
    input_ids = torch.randint(0, 50000, (T,), dtype=torch.long, device=device)

    # ---- Component 1: embedding lookup + reshape ----
    def fn_emb():
        x = emb_w[input_ids].view(1, T, n_embd).contiguous()
        return x

    # Pre-compute x for downstream stages.
    x_after_emb = fn_emb()

    # ---- Component 2: full layer-stack loop (12 layers) on a fresh state ----
    def fn_layers():
        state = cpp_path.fresh_state(weights)
        x = x_after_emb
        for i in range(N_LAYER):
            x = krunch_ac_cuda.rwkv4_layer_step_cpp(
                x.contiguous(),
                state[0][i], state[1][i], state[2][i], state[3][i], state[4][i],
                *layers[i],
            )
        return x

    x_after_layers = fn_layers()
    x_flat = x_after_layers.view(T, n_embd).contiguous()

    # ---- Component 3: final layer_norm (T, n_embd) ----
    def fn_ln():
        return F.layer_norm(x_flat, (n_embd,), weight=ln_out_w, bias=ln_out_b)

    xn = fn_ln().contiguous()
    head_w_c = head_w.contiguous()

    # ---- Component 4: head matmul — det_matmul (M=T, K=768, N=50277) ----
    def fn_head():
        return krunch_ac_cuda.det_matmul(xn, head_w_c)

    # ---- Component 0: full window forward (the production call) ----
    def fn_full():
        state = cpp_path.fresh_state(weights)
        return cpp_path.forward_packed_window(weights, input_ids, state, 0, T)

    print(f"\nCompress per-window breakdown (T={T}, A10G/T4):")
    full_ms = bench("forward_packed_window (full)", fn_full)
    print()
    bench("embedding lookup",     fn_emb,    total=full_ms)
    bench("layer loop (12 layers)", fn_layers, total=full_ms)
    bench("final layer_norm",     fn_ln,     total=full_ms)
    bench("head det_matmul",      fn_head,   total=full_ms)

    print(f"\nshape recap: head matmul is M={T}, K={n_embd}, "
          f"N={head_w.shape[-1]} (det_matmul_tc, 1 warp/block)")


if __name__ == "__main__":
    main()

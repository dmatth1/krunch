"""De-risk measurement Round 2 (V1_PLAN Tier 3).

Round 1 (T4) findings:
  - BlinkDL T=1:                      7.5 ms/token (current decompress)
  - forward_batched eager T=1:       12.0 ms/token (slower than BlinkDL)
  - BlinkDL T=1024 packed:           13.5 ms/call → 0.013 ms/token (compress)
  - DRIFT BlinkDL vs forward_batched: 2.6-7.1 abs (BIG — different ops)
  - DRIFT fbPacked vs fbStepped:      0.016-0.063 abs (TINY — same code path)
  - torch.compile on forward_batched: FakeTensor error on gemm_fp16_cublas
    (custom op opaque to dynamo).

Round 2 explores:
  - With `KRUNCH_PLAIN_MATMUL=1` (plain `@` instead of gemm_fp16_cublas),
    does forward_batched compile cleanly?
  - Speed of plain-matmul forward_batched at T=1 vs T=1024.
  - Drift: forward_batched-plain T=1 vs T=1024 (would AC roundtrip work
    if we use plain forward_batched on both encode + decode?).

If KRUNCH_PLAIN_MATMUL forward_batched compiles AND beats BlinkDL T=1
AND drift between compiled-T=1 and eager-T=1024 is < 0.1 abs, we have a
winning pure-torch path that closes the gap WITHOUT a custom CUDA kernel.
"""

import os
import time
import torch

os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")
os.environ["KRUNCH_PLAIN_MATMUL"] = "1"  # force compile-friendly path
os.environ["KRUNCH_PURE_WKV"] = "1"      # pure-torch WKV for compile

from krunch.inference import _load_rwkv, MODEL_PATH
from krunch.batched_rwkv4 import forward_batched, init_state_batched

N_WARMUP = 8
N_ITERS = 100


def bench(label, fn, n_iters=N_ITERS):
    for _ in range(N_WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_iters
    print(f"BENCH {label} per-call-ms={dt*1000:.3f}", flush=True)
    return dt


def logits_at(model, tokens_list, fwd_fn, packed=False):
    """Run model forward across tokens_list. fwd_fn is either:
       - BlinkDL: model.forward([tok], state) per-token
       - forward_batched: pass [B,T] tensor. If packed=True, do T=full single call.
    Returns list of per-step logits tensors."""
    device = "cuda"
    if packed:
        state = init_state_batched(model, B=1, device=device)
        tokens = torch.tensor([tokens_list], dtype=torch.long, device=device)
        logits_full, _ = fwd_fn(model, tokens, state, full_output=True)
        return [logits_full[0, t].float().clone() for t in range(len(tokens_list))]
    else:
        state = init_state_batched(model, B=1, device=device)
        tok_t = torch.zeros(1, 1, dtype=torch.long, device=device)
        out = []
        for tok in tokens_list:
            tok_t.fill_(tok)
            logits, state = fwd_fn(model, tok_t, state, full_output=False)
            out.append(logits.flatten().float().clone())
        return out


def main():
    RWKV = _load_rwkv()
    model_path_no_ext = str(MODEL_PATH).removesuffix(".pth")
    print("loading RWKV-4-Pile-169M ...", flush=True)
    model = RWKV(model=model_path_no_ext, strategy="cuda fp16", verbose=False)
    print("loaded", flush=True)

    device = "cuda"
    print(f"KRUNCH_PLAIN_MATMUL={os.environ.get('KRUNCH_PLAIN_MATMUL')}", flush=True)

    # Baseline: BlinkDL T=1 (current decompress)
    state_a = None
    last_a = 0
    def fn_a():
        nonlocal state_a, last_a
        logits, state_a = model.forward([last_a], state_a)
        last_a = int(logits.argmax().item()) if torch.is_tensor(logits) else 0
    bench("A_blinkdl_t1", fn_a)

    # forward_batched with PLAIN MATMUL (env-toggled), T=1 eager
    state_b = init_state_batched(model, B=1, device=device)
    last_b = torch.zeros(1, 1, dtype=torch.long, device=device)
    def fn_b():
        nonlocal state_b, last_b
        logits, state_b = forward_batched(model, last_b, state_b, full_output=False)
        last_b.fill_(int(logits.argmax(dim=-1).item()))
    bench("B_fb_plainmm_eager_t1", fn_b)

    # forward_batched plain T=1024 packed (eager)
    state_c = init_state_batched(model, B=1, device=device)
    tokens_c = torch.zeros(1, 1024, dtype=torch.long, device=device)
    def fn_c():
        forward_batched(model, tokens_c, state_c, full_output=True)
    bench("C_fb_plainmm_eager_t1024_packed", fn_c, n_iters=20)

    # torch.compile on forward_batched with plain mm — does it work now?
    print("\ncompiling forward_batched (default mode) with KRUNCH_PLAIN_MATMUL=1 ...", flush=True)
    try:
        fb_compiled = torch.compile(forward_batched, fullgraph=False, dynamic=False)
        # Warm up the compile by running it once
        state_d = init_state_batched(model, B=1, device=device)
        last_d = torch.zeros(1, 1, dtype=torch.long, device=device)
        _ = fb_compiled(model, last_d, state_d, full_output=False)
        torch.cuda.synchronize()
        print("compile OK on T=1 path", flush=True)
        def fn_d():
            nonlocal state_d, last_d
            logits, state_d = fb_compiled(model, last_d, state_d, full_output=False)
            last_d.fill_(int(logits.argmax(dim=-1).item()))
        bench("D_fb_plainmm_compiled_t1", fn_d)
    except Exception as e:
        print(f"BENCH D_fb_plainmm_compiled_t1 FAILED: {type(e).__name__}: {e}", flush=True)

    # torch.compile reduce-overhead (CUDA graphs)
    print("\ncompiling forward_batched (reduce-overhead) with KRUNCH_PLAIN_MATMUL=1 ...", flush=True)
    try:
        fb_compiled_ro = torch.compile(forward_batched, mode="reduce-overhead",
                                        fullgraph=False, dynamic=False)
        state_e = init_state_batched(model, B=1, device=device)
        last_e = torch.zeros(1, 1, dtype=torch.long, device=device)
        _ = fb_compiled_ro(model, last_e, state_e, full_output=False)
        torch.cuda.synchronize()
        print("compile OK on T=1 path (reduce-overhead)", flush=True)
        def fn_e():
            nonlocal state_e, last_e
            logits, state_e = fb_compiled_ro(model, last_e, state_e, full_output=False)
            last_e.fill_(int(logits.argmax(dim=-1).item()))
        bench("E_fb_plainmm_compiled_RO_t1", fn_e)
    except Exception as e:
        print(f"BENCH E_fb_plainmm_compiled_RO_t1 FAILED: {type(e).__name__}: {e}", flush=True)

    # Drift between forward_batched plain T=1024 packed vs T=1 stepped (both eager)
    print("\n--- DRIFT: forward_batched plain mm packed T=full vs stepped T=1 ---", flush=True)
    seq = [0]
    state = None
    for _ in range(16):
        logits, state = model.forward([seq[-1]], state)
        if not torch.is_tensor(logits):
            logits = torch.as_tensor(logits, device=device)
        seq.append(int(logits.argmax().item()))
    print(f"reference seq: {seq}", flush=True)
    logits_packed = logits_at(model, seq[:-1], forward_batched, packed=True)
    logits_stepped = logits_at(model, seq[:-1], forward_batched, packed=False)
    for i, (a, b) in enumerate(zip(logits_packed, logits_stepped)):
        diff = (a - b).abs().max().item()
        print(f"DRIFT_fbPacked_vs_fbStepped step={i} max_abs={diff:.4f}", flush=True)


if __name__ == "__main__":
    main()

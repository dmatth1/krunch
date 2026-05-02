"""Verify rwkv4_layer_step_v2 (multi-block cooperative kernel) matches
v1 (and rwkv4_step_ref) within fp16 noise.

Steps:
  1. Run 4 sequential token-step forwards through ALL 12 layers using v1
     (`launch_rwkv4_layer_step` via `rwkv4_layer_step` pybind).
  2. Run the same 4-token sequence through v2 (`rwkv4_layer_step_v2`).
  3. Compare per-step x_out + final state across all 12 layers.

Acceptance: max-abs diff per output ≤ 0.1 (fp16 noise from different
reduction order across blocks). v1 reference itself drifts ~0.016 from
the pure-torch _layer_step reference.

Then microbench v2 vs v1 for one full token (12-layer pass).
"""
import os, sys, time
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import torch
import krunch_ac_cuda
import rwkv.model

from krunch.inference import _load_rwkv, MODEL_PATH
from krunch import cpp_path


def maxabs(a, b): return (a.float() - b.float()).abs().max().item()


def run_v1(weights, last_token, state):
    """One token forward through all 12 layers using v1 single-block kernel."""
    n_embd = weights["n_embd"]
    n_att  = weights["n_att"]
    layers = weights["layers"]
    emb_w  = weights["emb_w"]

    device = weights["device"]
    x_in  = torch.empty(n_embd, dtype=torch.float16, device=device)
    x_out = torch.empty(n_embd, dtype=torch.float16, device=device)
    x_in.copy_(emb_w[last_token].view(n_embd))

    for i in range(12):
        L = layers[i]
        # rwkv4_layer_step pybind takes the same args list as v2.
        krunch_ac_cuda.rwkv4_layer_step(
            x_in, x_out,
            state[0][i].view(n_embd), state[1][i].view(n_att),
            state[2][i].view(n_att), state[3][i].view(n_att),
            state[4][i].view(n_embd),
            *L,
        )
        x_in, x_out = x_out, x_in  # ping-pong
    return x_in.clone()


def run_v2(weights, last_token, state, scratch):
    """One token forward through all 12 layers using v2 cooperative kernel."""
    n_embd = weights["n_embd"]
    n_att  = weights["n_att"]
    layers = weights["layers"]
    emb_w  = weights["emb_w"]

    device = weights["device"]
    x_in  = torch.empty(n_embd, dtype=torch.float16, device=device)
    x_out = torch.empty(n_embd, dtype=torch.float16, device=device)
    x_in.copy_(emb_w[last_token].view(n_embd))

    for i in range(12):
        L = layers[i]
        krunch_ac_cuda.rwkv4_layer_step_v2(
            x_in, x_out,
            state[0][i].view(n_embd), state[1][i].view(n_att),
            state[2][i].view(n_att), state[3][i].view(n_att),
            state[4][i].view(n_embd),
            *L,
            scratch,
        )
        x_in, x_out = x_out, x_in
    return x_in.clone()


def main():
    print("Loading RWKV-4-Pile-169M...", flush=True)
    RWKV = _load_rwkv()
    model = RWKV(model=str(MODEL_PATH).removesuffix(".pth"), strategy="cuda fp16")
    weights = cpp_path.init_weights(model, "cuda")

    n_bytes = krunch_ac_cuda.v2_scratch_bytes()
    print(f"v2 scratch: {n_bytes} bytes")
    scratch = torch.empty(n_bytes, dtype=torch.uint8, device="cuda")

    # Run 4 tokens through both paths. State is mutated in place; need fresh
    # copies for each run.
    BOS = 0
    tokens = [BOS, 100, 200, 300]

    state_v1 = cpp_path.fresh_state(weights)
    state_v2 = cpp_path.fresh_state(weights)

    print("\nPer-token output diff (v1 vs v2):")
    last_v1 = BOS
    last_v2 = BOS
    for t, tok in enumerate(tokens):
        out_v1 = run_v1(weights, last_v1, state_v1)
        out_v2 = run_v2(weights, last_v2, state_v2, scratch)
        d = maxabs(out_v1, out_v2)
        print(f"  step {t} tok={tok:5d}: max-abs = {d:.6e}")
        last_v1 = tok
        last_v2 = tok

    print("\nFinal state diffs (over all 12 layers):")
    for s_idx, name in enumerate(["att_xx", "aa", "bb", "pp", "ffn_xx"]):
        max_d = max(
            (state_v1[s_idx][i].float() - state_v2[s_idx][i].float()).abs().max().item()
            for i in range(12)
        )
        print(f"  {name:8s}: max-abs over 12 layers = {max_d:.6e}")

    # Microbench
    print("\nMicrobench (one full 12-layer token forward):")
    N_WARM = 10
    N_ITERS = 200

    last = tokens[-1]
    for _ in range(N_WARM): run_v1(weights, last, state_v1)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(N_ITERS): run_v1(weights, last, state_v1)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    v1_ms = (t1 - t0) / N_ITERS * 1000
    print(f"  v1 (single-block):    {v1_ms:7.3f} ms/token")

    for _ in range(N_WARM): run_v2(weights, last, state_v2, scratch)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(N_ITERS): run_v2(weights, last, state_v2, scratch)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    v2_ms = (t1 - t0) / N_ITERS * 1000
    print(f"  v2 (multi-block 24x32): {v2_ms:7.3f} ms/token  (speedup vs v1: {v1_ms/v2_ms:.2f}×)")

    # cpp_path.forward_stepped (production path) for reference
    state_ref = cpp_path.fresh_state(weights)
    def fn_cpp():
        return cpp_path.forward_stepped(weights, last, state_ref)
    for _ in range(N_WARM): fn_cpp()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(N_ITERS): fn_cpp()
    torch.cuda.synchronize(); t1 = time.perf_counter()
    cpp_ms = (t1 - t0) / N_ITERS * 1000
    print(f"  cpp_path.forward_stepped (incl. head+CDF): {cpp_ms:7.3f} ms/token")


if __name__ == "__main__":
    main()

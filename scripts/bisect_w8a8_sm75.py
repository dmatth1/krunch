"""Bisect where encoder packed and decoder stepped diverge with
KRUNCH_INT8_W8A8=1 on sm_75 (T4).

Runs the same short token sequence through:
  A) forward_packed_window — what the encoder uses
  B) forward_stepped (M=1) — what the decoder uses
both starting from identical fresh state, both with W8A8 weights.

Prints per-layer post-step output diffs (or, if they agree at the
layer level, the per-call dispatch differences). The first diverging
layer points at the broken kernel.

Run inside the published Docker image:
  docker run --rm --gpus all \\
    -e KRUNCH_INT8_W8A8=1 -v /tmp:/tmp \\
    --entrypoint /opt/conda/bin/python \\
    ghcr.io/dmatth1/krunch:latest \\
    /tmp/bisect_w8a8_sm75.py
"""
import os
os.environ.setdefault("KRUNCH_INT8_W8A8", "1")
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import torch
import krunch_ac_cuda
from krunch.inference import _load_rwkv, MODEL_PATH, BOS_TOKEN
from krunch import cpp_path

N_LAYER = 12
TOKENS = [BOS_TOKEN, 510, 3158, 8516, 30013, 27287, 689, 253]


def maxabs(a, b):
    return (a.float() - b.float()).abs().max().item()


def main():
    print("=== W8A8 sm_75 bisect: where do encoder packed and "
          "decoder stepped diverge? ===\n")

    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name} sm_{p.major}{p.minor}")
    print(f"KRUNCH_INT8_W8A8 = {os.environ.get('KRUNCH_INT8_W8A8')}\n")

    RWKV = _load_rwkv()
    model = RWKV(model=str(MODEL_PATH).removesuffix(".pth"),
                 strategy="cuda fp16", verbose=False)
    weights = cpp_path.init_weights(model, "cuda")

    # Sanity: confirm W8A8 weight bundle was populated.
    has_w8a8 = "w8a8_int8" in weights
    print(f"weights['w8a8_int8'] present: {has_w8a8}\n")

    T = len(TOKENS)
    input_t = torch.as_tensor(TOKENS, dtype=torch.long, device="cuda")

    # =======================================================================
    # Part 1: top-level logits diff. Run packed (full window) vs stepped
    # (one token at a time) and compare logits[t] for t in 0..T-1.
    # =======================================================================
    print("--- Part 1: logits divergence ---")
    state_p = cpp_path.fresh_state(weights)
    packed_logits = cpp_path.forward_packed_window(
        weights, input_t, state_p, off=0, n=T
    )

    state_s = cpp_path.fresh_state(weights)
    stepped_logits = []
    for tok in TOKENS:
        stepped_logits.append(
            cpp_path.forward_stepped(weights, tok, state_s).clone()
        )
    stepped_logits = torch.stack(stepped_logits)

    print(f"  packed_logits  shape={tuple(packed_logits.shape)}")
    print(f"  stepped_logits shape={tuple(stepped_logits.shape)}")
    for t in range(T):
        diff = maxabs(packed_logits[t], stepped_logits[t])
        marker = "" if diff == 0.0 else " ← DIVERGED"
        print(f"  t={t} (tok={TOKENS[t]:5d}): max|packed - stepped| = {diff:.4e}{marker}")
    overall = maxabs(packed_logits, stepped_logits)
    print(f"  overall max diff: {overall:.4e}")

    if overall == 0.0:
        print("\n*** Logits agree. Bug is downstream of the forward — "
              "look at CDF / AC kernel paths. ***")
        return

    # =======================================================================
    # Part 2: layer-by-layer divergence. For each layer i in 0..11, compare
    # the layer-step output of the packed path (at first token) vs the
    # stepped path. Identifies which layer first diverges.
    # =======================================================================
    print("\n--- Part 2: per-layer divergence (first token) ---")
    n_embd = weights["n_embd"]
    layers = weights["layers"]
    emb_w = weights["emb_w"]

    # Packed path replays internally — call forward_packed_window with n=1
    # and capture intermediate layer outputs by re-running per-layer
    # rwkv4_layer_step_cpp at T=1 (which mirrors what packed does for n=1).
    # For T=1 packed and T=1 stepped should converge to the same kernel
    # (rwkv4_layer_step_cpp), so this part isolates the W8A8 dispatch
    # at M=1.
    state_p = cpp_path.fresh_state(weights)
    state_s = cpp_path.fresh_state(weights)

    x_p = emb_w[TOKENS[0]].view(1, 1, n_embd).contiguous()
    x_s = emb_w[TOKENS[0]].view(1, 1, n_embd).contiguous()
    for i in range(N_LAYER):
        x_p = krunch_ac_cuda.rwkv4_layer_step_cpp(
            x_p.contiguous(),
            state_p[0][i], state_p[1][i], state_p[2][i],
            state_p[3][i], state_p[4][i],
            *layers[i],
        )
        x_s = krunch_ac_cuda.rwkv4_layer_step_cpp(
            x_s.contiguous(),
            state_s[0][i], state_s[1][i], state_s[2][i],
            state_s[3][i], state_s[4][i],
            *layers[i],
        )
        diff = maxabs(x_p, x_s)
        print(f"  layer {i:2d}: max|x_p - x_s| = {diff:.4e}")
    print("(both paths above use rwkv4_layer_step_cpp at T=1; if these "
          "agree, the kernel is deterministic at M=1 — divergence is "
          "in the M>=2 packed path.)")

    # =======================================================================
    # Part 3: forward_packed_window at full T vs single-step replay.
    # Specifically isolates whether the W8A8 packed dispatch at T=8 and
    # M=8 (still M < 256, so should NOT route to W8A8 packed) is the
    # culprit.
    # =======================================================================
    print("\n--- Part 3: full-T packed vs full-T per-token-stepped ---")
    state_p = cpp_path.fresh_state(weights)
    state_s = cpp_path.fresh_state(weights)
    packed = cpp_path.forward_packed_window(
        weights, input_t, state_p, off=0, n=T
    )
    stepped = []
    for tok in TOKENS:
        stepped.append(
            cpp_path.forward_stepped(weights, tok, state_s).clone()
        )
    stepped = torch.stack(stepped)
    for t in range(T):
        diff = maxabs(packed[t], stepped[t])
        marker = "" if diff < 1e-3 else " ← DIVERGED"
        print(f"  t={t}: max diff = {diff:.4e}{marker}")
    print(f"  overall: {maxabs(packed, stepped):.4e}")


if __name__ == "__main__":
    main()

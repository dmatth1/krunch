"""Bit-exact unit test for multi-window forward_packed_window state carry.

After the streaming-compress change (2026-04-30) added the SEQ_BATCH
window loop in `cpp_path.forward_packed_window`, T3 hit a CRC mismatch
on a multi-chunk WildChat sample. Same wrong CRC at B=8 and B=128 →
bug is not a B-scale issue but a numerics mismatch between streaming
compress and stepped decompress.

This test isolates the streaming step. With T = 3 × SEQ_BATCH and
real RWKV-4-Pile-169M weights, run the same input two ways:

  A) `forward_packed_window` once with T = full length (one window)
  B) `forward_packed_window` 3 times with T = SEQ_BATCH each, state
     carrying forward (the streaming compress code path)

If (A) and (B) produce bit-identical per-token logits, the streaming
compress is correct and the bug is elsewhere (probably in the
batched-decompress glue: input_buf concat, base offsets, AC state
init for B>1).

If they disagree, the streaming compress has a state-carry bug — fix
points to inside `forward_packed_window` or the per-layer state copy
semantics.
"""
import os
os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

import torch
from krunch.inference import _load_rwkv, MODEL_PATH, BOS_TOKEN
from krunch import cpp_path


def maxabs(a, b):
    return (a.float() - b.float()).abs().max().item()


def main():
    SEQ_BATCH = 1024
    N_WINDOWS = 3
    T = SEQ_BATCH * N_WINDOWS

    RWKV = _load_rwkv()
    print("loading model...", flush=True)
    model = RWKV(model=str(MODEL_PATH).removesuffix(".pth"),
                 strategy="cuda fp16", verbose=False)
    weights = cpp_path.init_weights(model, "cuda")

    torch.manual_seed(0)
    # Use a real-ish sequence: BOS + random valid tokens.
    rng = torch.Generator(device="cuda").manual_seed(42)
    input_ids = torch.cat([
        torch.tensor([BOS_TOKEN], dtype=torch.long, device="cuda"),
        torch.randint(0, 50000, (T - 1,), generator=rng,
                       dtype=torch.long, device="cuda"),
    ])

    # ============== A) single big window ==============
    state_a = cpp_path.fresh_state(weights)
    with torch.no_grad():
        logits_a = cpp_path.forward_packed_window(
            weights, input_ids, state_a, off=0, n=T)  # [T, V]

    # ============== B) N_WINDOWS sequential windows ==============
    state_b = cpp_path.fresh_state(weights)
    chunks = []
    with torch.no_grad():
        for w in range(N_WINDOWS):
            off = w * SEQ_BATCH
            chunks.append(cpp_path.forward_packed_window(
                weights, input_ids, state_b, off=off, n=SEQ_BATCH))
    logits_b = torch.cat(chunks, dim=0)  # [T, V]

    # Compare per-window
    print(f"T={T} ({N_WINDOWS} × SEQ_BATCH={SEQ_BATCH})")
    all_zero = True
    for w in range(N_WINDOWS):
        lo, hi = w * SEQ_BATCH, (w + 1) * SEQ_BATCH
        d = maxabs(logits_a[lo:hi], logits_b[lo:hi])
        print(f"  window {w}: max abs diff = {d:.6e} {'OK' if d == 0 else 'MISMATCH'}")
        if d != 0:
            all_zero = False

    # State equality across all 12 layers, all 5 components
    print("Final state equality:")
    state_ok = True
    for k in range(5):
        names = ["att_xx", "aa", "bb", "pp", "ffn_xx"][k]
        for i in range(len(state_a[k])):
            d = maxabs(state_a[k][i], state_b[k][i])
            if d != 0:
                print(f"  layer {i:>2} {names}: diff {d:.6e}  MISMATCH")
                state_ok = False
    if state_ok:
        print("  all layers, all components: identical")

    raise SystemExit(0 if all_zero and state_ok else 1)


if __name__ == "__main__":
    main()

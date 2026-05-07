"""
Bisect the W8A8 encoder/decoder M-stability bug.

Question: encoder uses W8A8 layer step at M=T=large (packed). Decoder
uses W8A8 layer step at M=B=16 (stepped). Microbench confirmed the
W8A8 matmul kernel is bit-exact across M (0.0 diff M=1024[:16] vs
M=16). Yet E2E roundtrip fails. So the M-instability is somewhere
DOWNSTREAM of the matmul: in sigmoid, layer_norm, premix, WKV, or
the per-row dequant interaction.

Test approach:
  1. Tokenize 1 small chunk (e.g., 64 tokens).
  2. Run forward_packed_window(W8A8, T=64) → save logits[64, V].
  3. Run forward_stepped_batched(W8A8, B=1) for 64 steps with the
     ground-truth tokens (NOT AC-decoded) → save logits[64, V].
  4. Compare per-position. Find the first position where logits diverge
     beyond fp16 noise (≥ 0.5 abs).
  5. Also save per-LAYER outputs by hooking into the layer step.
     Identify which layer first diverges.

If layer 0 outputs match but layer 1+ diverge → M-instability is in
something AFTER layer 0 (state evolution? specific op?).

Run on GPU instance:
  docker run --rm --gpus all \\
    -e KRUNCH_CPP_PATH=1 -e KRUNCH_DETERMINISTIC_MATMUL=1 \\
    -e KRUNCH_OWN_WKV=1 -e KRUNCH_INT8_W8A8=1 \\
    -v /tmp/krunch_ac/cuda/krunch_ac_cuda.cpython-311-x86_64-linux-gnu.so:/app/krunch_ac_cuda.cpython-311-x86_64-linux-gnu.so:ro \\
    -v /tmp/inference.py:/app/krunch/inference.py:ro \\
    -v /tmp/cpp_path.py:/app/krunch/cpp_path.py:ro \\
    -v /tmp/bisect_w8a8_mstability.py:/tmp/bisect.py \\
    -v /tmp/sample.bin:/tmp/sample.bin \\
    --entrypoint /opt/conda/bin/python \\
    ghcr.io/dmatth1/krunch:latest \\
    /tmp/bisect.py --sample /tmp/sample.bin
"""
import argparse
import os
import sys

os.environ.setdefault("KRUNCH_DETERMINISTIC_MATMUL", "1")
os.environ.setdefault("KRUNCH_OWN_WKV", "1")
os.environ.setdefault("KRUNCH_CPP_PATH", "1")
os.environ.setdefault("KRUNCH_INT8_W8A8", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")
os.environ.setdefault("RWKV_JIT_ON", "1")


BOS_TOKEN = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--n_tokens", type=int, default=64,
                     help="Number of tokens to test (small for clarity)")
    args = ap.parse_args()

    # Use first ~256 bytes of the sample = ~64 tokens. Just need a few
    # tokens — looking for first-divergence position.
    with open(args.sample, "rb") as f:
        text = f.read(1024).decode("utf-8", errors="replace")

    import torch
    from krunch.inference import engine
    from krunch import cpp_path

    print("Loading model...", flush=True)
    engine.load()
    tok = engine._tokenizer.encode(text).ids[:args.n_tokens]
    print(f"Tokens: {len(tok)} (first 10: {tok[:10]})", flush=True)

    weights = cpp_path.init_weights(engine._model, engine._device)
    has_w8a8 = "w8a8_int8" in weights
    print(f"weights bundle has w8a8_int8: {has_w8a8}", flush=True)
    if not has_w8a8:
        sys.exit("FAIL: KRUNCH_INT8_W8A8=1 not enabled in weights bundle")

    full_input = [BOS_TOKEN] + tok[:-1]
    T = len(full_input)
    full_input_t = torch.as_tensor(
        full_input, dtype=torch.long, device=engine._device)

    # === ENCODER PATH: forward_packed_window at T=64 ===
    print(f"\n=== ENCODER (W8A8 packed, T={T}) ===", flush=True)
    enc_state = cpp_path.fresh_state(weights)
    enc_logits = cpp_path.forward_packed_window(
        weights, full_input_t, enc_state, off=0, n=T)
    torch.cuda.synchronize()
    print(f"  enc_logits shape: {tuple(enc_logits.shape)}, "
          f"dtype: {enc_logits.dtype}")

    # === DECODER PATH: forward_stepped_batched at B=1, called T times ===
    print(f"\n=== DECODER (W8A8 stepped, B=1, T iterations) ===", flush=True)
    dec_state = cpp_path.fresh_state_batched(weights, B=1)
    dec_logits_list = []
    for t in range(T):
        # Feed the GROUND-TRUTH previous token (matches encoder's input).
        last = torch.tensor([full_input[t]], dtype=torch.long,
                             device=engine._device)
        out = cpp_path.forward_stepped_batched(weights, last, dec_state)
        dec_logits_list.append(out[0])  # [V]
    torch.cuda.synchronize()
    dec_logits = torch.stack(dec_logits_list, dim=0)  # [T, V]
    print(f"  dec_logits shape: {tuple(dec_logits.shape)}, "
          f"dtype: {dec_logits.dtype}")

    # === COMPARE ===
    print(f"\n=== COMPARE ===")
    diff = (enc_logits.to(torch.float32) - dec_logits.to(torch.float32)).abs()
    per_pos_max = diff.max(dim=1).values  # [T]
    per_pos_mean = diff.mean(dim=1)
    print(f"  Per-position max-abs diff (first 16):")
    for t in range(min(16, T)):
        marker = " "
        if per_pos_max[t].item() > 0.5:
            marker = "← DIVERGES"
        elif per_pos_max[t].item() > 0.05:
            marker = "← NOISE"
        print(f"    t={t:3d}  max_abs={per_pos_max[t].item():8.4f}  "
              f"mean_abs={per_pos_mean[t].item():8.5f}  {marker}")
    print(f"  Overall max-abs: {diff.max().item():.4f}")
    print(f"  First position with max_abs > 0.5: ",
          end="")
    bad = (per_pos_max > 0.5).nonzero(as_tuple=True)[0]
    if len(bad) == 0:
        print("none (W8A8 is M/B-stable!)")
    else:
        print(f"t={bad[0].item()}")

    # Also compare argmax (which token would AC pick at each position)
    enc_argmax = enc_logits.argmax(dim=1)
    dec_argmax = dec_logits.argmax(dim=1)
    mismatches = (enc_argmax != dec_argmax).nonzero(as_tuple=True)[0]
    print(f"\n  Argmax mismatches: {len(mismatches)} / {T} positions")
    if len(mismatches) > 0:
        print(f"  First mismatch: t={mismatches[0].item()}, "
              f"enc={enc_argmax[mismatches[0]].item()}, "
              f"dec={dec_argmax[mismatches[0]].item()}")


if __name__ == "__main__":
    main()

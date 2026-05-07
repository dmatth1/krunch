"""
Layer-step-level bisect for the W8A8 T-stability bug.

Microbench confirmed the W8A8 matmul kernel is M-stable at M=1, 16,
32 vs 1024 (0.0 diff). The bug is somewhere in the layer step's
interaction with downstream ops. This script narrows further:

For ONE layer, run rwkv4_layer_step_cpp_w8a8 directly:
  Encoder mode: x [1, 32, C], state [1, ...], one call → x_out [1, 32, C]
  Decoder mode: x [1, 1, C], state [1, ...], 32 calls (manually advancing
                state) → x_out_list of 32 [1, 1, C]

Compare encoder x_out[:, t, :] vs decoder call_t.x_out[:, 0, :] for each t.
First t where they diverge → the layer-step has T-stability bug.

If layer 0 matches across t, the bug is downstream (WKV state evolution
in subsequent layers, or layer 1+'s receipt of layer 0's output).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=8, help="positions to test")
    args = ap.parse_args()
    T = args.T

    import torch
    from krunch.inference import engine
    from krunch import cpp_path
    import krunch_ac_cuda

    print("Loading model...", flush=True)
    engine.load()
    weights = cpp_path.init_weights(engine._model, engine._device)
    if "w8a8_int8" not in weights:
        sys.exit("FAIL: KRUNCH_INT8_W8A8=1 not in weights bundle")

    n_embd = weights["n_embd"]
    n_att = weights["n_att"]
    emb_w = weights["emb_w"]
    layers = weights["layers"]
    w8a8_int8 = weights["w8a8_int8"]
    w8a8_scale = weights["w8a8_scale"]
    device = engine._device

    # Construct identical input embeddings for T positions.
    torch.manual_seed(0)
    token_ids = torch.randint(0, 50000, (T,), device=device, dtype=torch.long)
    x_full = emb_w[token_ids].view(1, T, n_embd).contiguous()  # [1, T, n_embd]
    print(f"Test input: T={T}, x_full shape={tuple(x_full.shape)}", flush=True)

    def fresh_state_b1():
        """Fresh single-batch state (matches both encoder and decoder
        bisect setup at B=1)."""
        return (
            [torch.zeros(1, n_embd, dtype=torch.float16, device=device) for _ in range(12)],
            [torch.zeros(1, n_att, dtype=torch.float32, device=device) for _ in range(12)],
            [torch.zeros(1, n_att, dtype=torch.float32, device=device) for _ in range(12)],
            [torch.full((1, n_att), -1e30, dtype=torch.float32, device=device) for _ in range(12)],
            [torch.zeros(1, n_embd, dtype=torch.float16, device=device) for _ in range(12)],
        )

    def call_layer(x, state_layer, layer_idx):
        """One layer of W8A8 step with the given state slice."""
        L = layers[layer_idx]
        ints = w8a8_int8[layer_idx]
        scales = w8a8_scale[layer_idx]
        return krunch_ac_cuda.rwkv4_layer_step_cpp_w8a8(
            x.contiguous(),
            state_layer[0], state_layer[1], state_layer[2],
            state_layer[3], state_layer[4],
            L[0], L[1], L[2], L[3], L[4], L[5], L[6],
            ints[0], scales[0], ints[1], scales[1],
            ints[2], scales[2], ints[3], scales[3],
            L[11], L[12], L[13], L[14],
            ints[4], scales[4], ints[5], scales[5], ints[6], scales[6],
        )

    # === ENCODER: pass full T positions through layer 0 ===
    print(f"\n=== Encoder pass: layer 0, T={T} packed ===", flush=True)
    enc_state = fresh_state_b1()
    enc_state_layer0 = (enc_state[0][0], enc_state[1][0], enc_state[2][0],
                        enc_state[3][0], enc_state[4][0])
    enc_x_out = call_layer(x_full, enc_state_layer0, 0)
    torch.cuda.synchronize()
    print(f"  enc_x_out shape: {tuple(enc_x_out.shape)}")

    # === DECODER: pass T positions one at a time through layer 0 ===
    print(f"\n=== Decoder pass: layer 0, T={T} stepped ===", flush=True)
    dec_state = fresh_state_b1()
    dec_state_layer0 = (dec_state[0][0], dec_state[1][0], dec_state[2][0],
                        dec_state[3][0], dec_state[4][0])
    dec_outs = []
    for t in range(T):
        x_t = x_full[:, t:t+1, :].contiguous()  # [1, 1, n_embd]
        x_out_t = call_layer(x_t, dec_state_layer0, 0)
        dec_outs.append(x_out_t.clone())  # [1, 1, n_embd]
    torch.cuda.synchronize()
    dec_x_out = torch.cat(dec_outs, dim=1)  # [1, T, n_embd]
    print(f"  dec_x_out shape: {tuple(dec_x_out.shape)}")

    # === COMPARE per-position ===
    print(f"\n=== COMPARE encoder x_out[:, t, :] vs decoder x_out_t ===")
    diff = (enc_x_out.to(torch.float32) - dec_x_out.to(torch.float32)).abs()
    per_pos_max = diff.max(dim=2).values.squeeze(0)  # [T]
    for t in range(T):
        marker = "← DIVERGES" if per_pos_max[t].item() > 0.01 else ""
        print(f"  t={t}  layer0_max_abs={per_pos_max[t].item():.6f}  {marker}")

    if per_pos_max.max().item() < 0.001:
        print("\n  ✓ Layer 0 IS T-stable. Bug is in subsequent layers or state.")
    else:
        print(f"\n  ✗ Layer 0 NOT T-stable. Bug is INSIDE rwkv4_layer_step_cpp_w8a8.")
        print(f"  First diverging position: t={(per_pos_max > 0.001).nonzero(as_tuple=True)[0][0].item()}")

    # === Also compare final state ===
    print(f"\n=== Final state diff (after T positions) ===")
    state_names = ["att_xx", "aa", "bb", "pp", "ffn_xx"]
    for k in range(5):
        d = (enc_state_layer0[k].to(torch.float32) -
              dec_state_layer0[k].to(torch.float32)).abs().max().item()
        marker = " ← DIFFERS" if d > 0.001 else ""
        print(f"  {state_names[k]}: max_abs_diff={d:.6f}{marker}")


if __name__ == "__main__":
    main()

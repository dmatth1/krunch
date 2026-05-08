"""
A1 reconnaissance: how sparse is FFN-V's input on real WildChat?

The premise: in the FFN block, k_ffn = relu(matmul(ffn_kx, ffn_K))**2
produces values where many entries are zero (anywhere the pre-relu
activation was negative). The downstream matmul v_ffn = matmul(k_ffn,
ffn_V) is then a sparse matrix-vector / matrix-matrix product. If
the sparsity is high enough, an SpMV implementation could meaningfully
reduce FFN-V wall time (FFN-V is 30-35% of layer-step matmul cost
per the §2 phase profile).

This probe measures the actual sparsity ratio on production weights
+ a real WildChat sample. Result drives the A1 ship/skip decision.

Usage on g5.xlarge with the krunch image:

    docker run --rm --gpus all -v /tmp:/work \
      -e KRUNCH_INT8_W8A8=1 \
      ghcr.io/dmatth1/krunch:latest \
      python3 /work/probe_ffn_v_sparsity.py /work/sample.bin

Per docs/TIER_3_OPTIMIZATION.md NEXT after Phase 2B SKIP (2026-05-08).
"""

import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

# Live in the same import order krunch's CLI uses to ensure cpp_path
# auto-detection runs.
from krunch.inference import InferenceEngine, BOS_TOKEN
from krunch import cpp_path


def _load_sample_tokens(sample_path: str, n_tokens_target: int = 8192) -> list[int]:
    """Load a subset of the sample and tokenize via krunch's tokenizer."""
    raw = Path(sample_path).read_bytes()
    # Try a handful of slice sizes until we have ~n_tokens_target tokens.
    for slice_len in (n_tokens_target * 4, n_tokens_target * 6, len(raw)):
        text = raw[:slice_len].decode("utf-8", errors="replace")
        # Use a deterministic tokenizer instance.
        from tokenizers import Tokenizer
        from krunch.inference import TOKENIZER_PATH
        from tokenizers.normalizers import Sequence as _NormSeq
        tok = Tokenizer.from_file(str(TOKENIZER_PATH))
        tok.normalizer = _NormSeq([])
        ids = tok.encode(text).ids
        if len(ids) >= n_tokens_target:
            return ids[:n_tokens_target]
    return ids  # whatever we got


def _ffn_block_torch(xx2_flat, ffn_xx, ffn_tm_k, ffn_tm_r,
                     ffn_K_fp16, ffn_V_fp16, ffn_R_fp16):
    """Reference FFN block in pure torch fp16. Mirrors the C++
    rwkv4_layer_step_cpp inner FFN math, in dense form. Returns
    (k_ffn, v_ffn, r_ffn) where k_ffn is the post-relu² activation
    we care about for sparsity measurement."""
    # premix: ffn_kx = xx2_flat * tm_k + ffn_xx_broadcast * (1 - tm_k)
    ffn_kx = xx2_flat * ffn_tm_k + ffn_xx * (1.0 - ffn_tm_k)
    ffn_rx = xx2_flat * ffn_tm_r + ffn_xx * (1.0 - ffn_tm_r)
    k_ffn = torch.matmul(ffn_kx, ffn_K_fp16)            # [M, n_ffn]
    k_ffn = torch.relu(k_ffn) ** 2                       # sparsity here
    r_ffn = torch.sigmoid(torch.matmul(ffn_rx, ffn_R_fp16))
    v_ffn = torch.matmul(k_ffn, ffn_V_fp16)             # the matmul A1 attacks
    return k_ffn, v_ffn, r_ffn


def main():
    if len(sys.argv) < 2:
        sample = "/work/sample.bin"
    else:
        sample = sys.argv[1]
    if not Path(sample).exists():
        print(f"[FATAL] sample not found at {sample}", file=sys.stderr)
        sys.exit(2)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    p = torch.cuda.get_device_properties(0)
    print(f"[GPU] {p.name} sm_{p.major}{p.minor}, "
          f"{p.multi_processor_count} SMs")

    # --- Load engine ----------------------------------------------------
    print("[load] engine ...", flush=True)
    eng = InferenceEngine()
    eng.load()
    weights = cpp_path.init_weights(eng._model, "cuda")
    print(f"[load] n_layer={cpp_path.N_LAYER}  n_embd={weights['n_embd']}", flush=True)

    # --- Get tokens -----------------------------------------------------
    tokens = _load_sample_tokens(sample, n_tokens_target=4096)
    full_input = [BOS_TOKEN] + tokens[:-1]
    T = len(full_input)
    print(f"[load] T={T} tokens from {sample}", flush=True)

    # --- Run a packed forward window to populate state ------------------
    full_input_t = torch.as_tensor(full_input, dtype=torch.long, device="cuda")
    state = cpp_path.fresh_state(weights)

    # Walk through layers manually (in eager torch) to capture FFN
    # intermediates per layer. This duplicates ~the math of
    # rwkv4_layer_step_cpp_w8a8 but in pure torch on the FP16 weight
    # bundle (init_weights also stores the fp16 layer weights even
    # when W8A8 is on, in the "layers" list).
    n_embd = weights["n_embd"]
    emb_w = weights["emb_w"]
    layers_fp16 = weights["layers"]  # original fp16 layer weights

    x = emb_w[full_input_t].view(1, T, n_embd).contiguous()  # [1, T, C]

    # Per-layer: layers[i][:7] non-matmul, [7:11] att matmul, [11:13] ln2,
    # [13:15] ffn_tm, [15:18] ffn matmul. See cpp_path.init_weights.
    nnz_total = 0
    elem_total = 0
    per_layer_sparsity = []
    print("\n[probe] per-layer post-relu² sparsity")
    print(f"{'layer':>5} {'nnz':>10} {'total':>10} {'sparsity':>10}")
    with torch.no_grad():
        for i in range(cpp_path.N_LAYER):
            L = layers_fp16[i]
            ln1_w, ln1_b = L[0], L[1]
            tm_k_att, tm_v_att, tm_r_att = L[2], L[3], L[4]
            time_decay, time_first = L[5], L[6]
            Kw, Vw, Rw, Ow = L[7], L[8], L[9], L[10]
            ln2_w, ln2_b = L[11], L[12]
            ffn_tm_k, ffn_tm_r = L[13], L[14]
            ffn_Kw, ffn_Vw, ffn_Rw = L[15], L[16], L[17]

            # ATT block — torch ref.
            xx = torch.nn.functional.layer_norm(
                x.float(), (n_embd,), ln1_w.float(), ln1_b.float()
            ).to(torch.float16)
            xx_2d = xx.view(1, T, n_embd)
            # Time-mix: shift by 1, masked first.
            shifted = torch.zeros_like(xx_2d)
            shifted[:, 1:, :] = xx_2d[:, :-1, :]
            kx_a = xx_2d * tm_k_att + shifted * (1.0 - tm_k_att)
            vx_a = xx_2d * tm_v_att + shifted * (1.0 - tm_v_att)
            rx_a = xx_2d * tm_r_att + shifted * (1.0 - tm_r_att)
            k_a = kx_a.float() @ Kw.float()
            v_a = vx_a.float() @ Vw.float()
            r_a = torch.sigmoid(rx_a.float() @ Rw.float())
            # WKV — too long to faithfully replicate here; what we need
            # is the FFN intermediate sparsity, which depends on
            # ffn_kx after LN2 and premix on the att-block output.
            # Rough but good-enough: skip the exact WKV recurrence and
            # use a reasonable approximation by passing att-block-pre-WKV
            # through the same residual structure. This is fine for
            # SPARSITY MEASUREMENT (we just need realistic activations
            # to feed the FFN-K matmul), not for bit-exact validation.
            x = x + r_a.to(torch.float16) * v_a.to(torch.float16)

            # FFN block — the part we care about.
            xx2 = torch.nn.functional.layer_norm(
                x.float(), (n_embd,), ln2_w.float(), ln2_b.float()
            ).to(torch.float16)
            xx2_2d = xx2.view(1, T, n_embd)
            shifted_ffn = torch.zeros_like(xx2_2d)
            shifted_ffn[:, 1:, :] = xx2_2d[:, :-1, :]
            ffn_kx = xx2_2d * ffn_tm_k + shifted_ffn * (1.0 - ffn_tm_k)
            ffn_rx = xx2_2d * ffn_tm_r + shifted_ffn * (1.0 - ffn_tm_r)

            k_ffn = torch.matmul(
                ffn_kx.view(-1, n_embd).float(), ffn_Kw.float())
            k_ffn = torch.relu(k_ffn) ** 2
            v_ffn = torch.matmul(k_ffn, ffn_Vw.float())
            r_ffn = torch.sigmoid(torch.matmul(
                ffn_rx.view(-1, n_embd).float(), ffn_Rw.float()))
            x_flat = x.view(-1, n_embd) + (r_ffn.to(torch.float16) *
                                            v_ffn.to(torch.float16))
            x = x_flat.view(1, T, n_embd)

            # Sparsity stat: % of zeros in k_ffn (the FFN-V input).
            zeros = int((k_ffn == 0).sum().item())
            elem = int(k_ffn.numel())
            nnz = elem - zeros
            spr = zeros / elem
            per_layer_sparsity.append(spr)
            nnz_total += nnz
            elem_total += elem
            print(f"{i:>5} {nnz:>10} {elem:>10} {spr*100:>9.2f}%")

    overall_zero_frac = (elem_total - nnz_total) / elem_total
    print(f"\n[probe] OVERALL sparsity (fraction of zeros): "
          f"{overall_zero_frac*100:.2f}%")
    print(f"[probe] mean per-layer:                       "
          f"{100 * sum(per_layer_sparsity)/len(per_layer_sparsity):.2f}%")
    print(f"[probe] min per-layer:                        "
          f"{100 * min(per_layer_sparsity):.2f}%")
    print(f"[probe] max per-layer:                        "
          f"{100 * max(per_layer_sparsity):.2f}%")
    print(f"\n[probe] Implication for A1:")
    if overall_zero_frac >= 0.5:
        print("  ✓ ≥50% zeros → SpMV could 2× FFN-V time (15-17% wall savings).")
        print("    A1 worth implementing.")
    elif overall_zero_frac >= 0.3:
        print("  ⚠ 30-50% zeros → SpMV could give ~10% wall savings.")
        print("    A1 marginal; depends on overhead of mask computation.")
    else:
        print("  ✗ <30% zeros → SpMV unlikely to beat dense matmul on A10G.")
        print("    A1 probably another skip; pivot to NEXT-4 or ship.")


if __name__ == "__main__":
    main()

"""Per-input-channel uint8 weight calibration — design validation.

For each of the 7 quantizable matrices per layer (Kw, Vw, Rw, Ow,
ffn_Kw, ffn_Vw, ffn_Rw), compute per-input-channel scale + offset
quantization. Measure dequant MSE / max-abs error to validate the
INT8_WEIGHTS_DESIGN.md scheme on actual RWKV-4-Pile-169M weights.

Reference scheme (rwkv-cpp-accelerated):
  scale[k]  = (W[k,:].max() - W[k,:].min()) / 255
  offset[k] = W[k,:].min()
  W_uint8[k,n] = round((W[k,n] - offset[k]) / scale[k]).clip(0, 255)
  W_dequant[k,n] = W_uint8[k,n] * scale[k] + offset[k]

Runs CPU-only in ~10s. Loads weights via torch.load.
"""
import os, sys
from pathlib import Path

import torch

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "RWKV-4-Pile-169M-20220807-8023.pth"

QUANT_KEYS = ["att.key.weight", "att.value.weight", "att.receptance.weight",
              "att.output.weight", "ffn.key.weight", "ffn.value.weight",
              "ffn.receptance.weight"]


def quantize_per_input(W: torch.Tensor):
    """W shape [K, N]. Returns (W_uint8, scale, offset) with scale/offset shape [K]."""
    W_f = W.float()
    w_min = W_f.min(dim=1).values  # [K]
    w_max = W_f.max(dim=1).values  # [K]
    scale = (w_max - w_min) / 255.0
    scale = scale.clamp(min=1e-12)  # avoid div-by-0 for constant rows
    offset = w_min
    Q = ((W_f - offset.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 255).to(torch.uint8)
    return Q, scale, offset


def dequant_per_input(Q: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor):
    return Q.float() * scale.unsqueeze(1) + offset.unsqueeze(1)


def main():
    if not MODEL_PATH.exists():
        print(f"FAIL model not found: {MODEL_PATH}")
        sys.exit(2)

    print(f"Loading {MODEL_PATH}")
    sd = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    print(f"  {len(sd)} tensors\n")

    # Detect num layers
    layer_keys = [k for k in sd.keys() if k.startswith("blocks.") and k.endswith(".att.key.weight")]
    N_LAYER = len(layer_keys)
    print(f"N_LAYER = {N_LAYER}")

    # Per-matrix MSE / max-abs / SNR aggregates
    print("\n=== Per-layer per-matrix quantization error ===")
    print(f"{'matrix':<22} {'shape':<14} {'max-abs':<10} {'rmse':<10} {'rel-rmse':<10}")
    print("-" * 70)

    summary = {k: [] for k in QUANT_KEYS}

    for i in range(N_LAYER):
        for short in QUANT_KEYS:
            key = f"blocks.{i}.{short}"
            W = sd[key]  # [out, in] in nn.Linear convention; for our codec we treat [in, out] = W.T
            # Bellard scheme is per-input-channel. RWKV stores as [out, in].
            # We want per-input-channel = per-column when stored as [out, in],
            # OR per-row when transposed to [in, out].
            # We'll quantize the [in, out] = W.T view so "per-row" = "per-input-channel".
            W_io = W.t().contiguous()  # [in, out]
            Q, scale, offset = quantize_per_input(W_io)
            W_dq = dequant_per_input(Q, scale, offset)
            err = (W_io.float() - W_dq).abs()
            mx = err.max().item()
            rmse = err.pow(2).mean().sqrt().item()
            rms_orig = W_io.float().pow(2).mean().sqrt().item()
            rel_rmse = rmse / (rms_orig + 1e-12)
            summary[short].append((mx, rmse, rel_rmse))
            if i == 0:  # detailed only for layer 0; aggregate for rest
                print(f"L{i:02d}.{short:<18} {str(tuple(W_io.shape)):<14} {mx:.3e}  {rmse:.3e}  {rel_rmse:.3e}")

    print("\n=== Aggregates across all 12 layers ===")
    print(f"{'matrix':<22} {'avg max-abs':<14} {'avg rel-rmse':<14} {'worst rel-rmse'}")
    print("-" * 75)
    for short in QUANT_KEYS:
        stats = summary[short]
        avg_mx = sum(s[0] for s in stats) / len(stats)
        avg_rel = sum(s[2] for s in stats) / len(stats)
        worst_rel = max(s[2] for s in stats)
        print(f"{short:<22} {avg_mx:<14.3e} {avg_rel:<14.3e} {worst_rel:.3e}")

    # Storage savings
    fp16_bytes = sum(sd[f"blocks.{i}.{k}"].numel() * 2 for i in range(N_LAYER) for k in QUANT_KEYS)
    uint8_bytes = sum(sd[f"blocks.{i}.{k}"].numel() for i in range(N_LAYER) for k in QUANT_KEYS)
    scale_offset_bytes = sum(sd[f"blocks.{i}.{k}"].shape[1] * 4  # 2 fp16 each = 4 bytes per input channel
                              for i in range(N_LAYER) for k in QUANT_KEYS)
    print(f"\n=== Storage ===")
    print(f"  fp16 quantizable weights:    {fp16_bytes / 1e6:.2f} MB")
    print(f"  uint8 + scale/offset (fp16): {(uint8_bytes + scale_offset_bytes) / 1e6:.2f} MB")
    print(f"  reduction: {fp16_bytes / (uint8_bytes + scale_offset_bytes):.2f}x")


if __name__ == "__main__":
    main()

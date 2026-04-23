"""One-shot converter: RWKV-4 pretrained .pth → safetensors.

BlinkDL's pretrained RWKV checkpoints ship as PyTorch `.pth` files
(zip + pickle + raw tensors). Pickle is Python-specific and messy to
parse from Rust. Safetensors is a flat binary + JSON header format
designed exactly for cross-language weight loading.

This script runs ONCE per model release:

    python scripts/convert_rwkv_pth_to_safetensors.py \\
        RWKV-4-Pile-169M-20220807-8023.pth \\
        RWKV-4-Pile-169M-20220807-8023.safetensors

The output is loadable from Rust via the `safetensors` crate.
Bit-exact: every fp32/fp16 tensor value is preserved verbatim.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_pth", type=Path)
    p.add_argument("output_safetensors", type=Path)
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"],
                   help="Cast to this dtype before saving. Default: fp32 "
                        "(matches the published checkpoint and preserves "
                        "every bit). Use fp16 to halve file size.")
    args = p.parse_args()

    print(f"[convert] loading {args.input_pth}...")
    state = torch.load(str(args.input_pth), map_location="cpu", weights_only=False)

    # Optional dtype cast
    cast = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    if cast != torch.float32:
        print(f"[convert] casting to {args.dtype}")
        state = {k: v.to(cast) if v.is_floating_point() else v for k, v in state.items()}

    # Validate shape assumptions for RWKV-4-Pile-169M
    expected_vocab = 50277
    expected_embd = 768
    expected_layers = 12
    emb_shape = state["emb.weight"].shape
    head_shape = state["head.weight"].shape
    n_blocks = sum(1 for k in state if k.startswith("blocks.") and k.endswith(".ln1.weight"))
    print(f"[convert] vocab={emb_shape[0]}  n_embd={emb_shape[1]}  n_layer={n_blocks}  "
          f"head={head_shape}")
    if emb_shape != (expected_vocab, expected_embd):
        print(f"[convert] WARN: emb shape {emb_shape} != expected ({expected_vocab}, {expected_embd})",
              file=sys.stderr)
    if n_blocks != expected_layers:
        print(f"[convert] WARN: n_layer {n_blocks} != expected {expected_layers}", file=sys.stderr)

    # Every tensor must be contiguous for safetensors
    state = {k: v.contiguous() for k, v in state.items()}

    save_file(state, str(args.output_safetensors))
    size_mb = args.output_safetensors.stat().st_size / 1e6
    print(f"[convert] wrote {args.output_safetensors} ({size_mb:.1f} MB, {len(state)} tensors)")


if __name__ == "__main__":
    main()

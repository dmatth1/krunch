"""Convert an L3TC PyTorch checkpoint to the Rust-friendly binary format.

Reads a `.pth` file from the L3TC reference implementation, applies
HiRA merging (W = W + B @ A) on the key/value/receptance projections,
squeezes singleton batch dims on the time_mix vectors, renames
blocks.0.ln0.* to top-level ln0.* (matching the inference model's
layout), and writes everything out as a single flat binary file.

Format (all little-endian):

    magic: b"L3TC" (4 bytes)
    version: u32 (currently 1)
    n_tensors: u32
    For each tensor:
        name_len: u32
        name: utf-8 bytes
        ndim: u32
        dims: u32[ndim]
        dtype: u32 (0 = f32)
        data_len: u64 (bytes)
        data: raw bytes (little-endian f32)
    Trailer:
        magic: b"END!" (4 bytes)

The trailer makes it trivial to validate that we read the whole file.
The format carries no compression, no endianness switches, no padding.
It exists purely to be a dead-simple drop-in for Rust to read without
any pickle knowledge.

This script runs inside the L3TC venv (PyTorch is required). Run from
the repository root:

    cd vendor/L3TC && source .venv/bin/activate
    python ../../l3tc-rust/scripts/convert_checkpoint.py \\
        --input checkpoints/l3tc_checkpoints/l3tc_200k_bpe16k_c999_checkpoint0019.pth \\
        --config config/l3tc/l3tc_200k.py \\
        --output ../../l3tc-rust/checkpoints/l3tc_200k.bin
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import torch

MAGIC = b"L3TC"
TRAILER = b"END!"
VERSION = 1
DTYPE_F32 = 0


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load a PyTorch checkpoint and return just the model state dict."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def apply_hira_merge(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Merge HiRA branches into the base weights.

    For each block, for each of (att, ffn) × (key, value, receptance):
        W = W + B @ A
    The A and B tensors are then removed from the state dict.
    """
    merged: dict[str, torch.Tensor] = {}
    # Collect keys that look like HiRA branches so we can skip them
    # after merging.
    hira_keys: set[str] = set()

    for name, tensor in sd.items():
        # HiRA A/B tensors have suffixes like ".key_A.weight" or
        # ".key_B.weight" — recognise them and do the merge when we
        # encounter the base weight.
        if not name.endswith(".weight"):
            merged[name] = tensor
            continue
        base = name[: -len(".weight")]

        # If this is the base (e.g. blocks.0.att.key), check for A/B
        if any(base.endswith(f".{proj}") for proj in ("key", "value", "receptance")):
            a_key = f"{base}_A.weight"
            b_key = f"{base}_B.weight"
            if a_key in sd and b_key in sd:
                a = sd[a_key]
                b = sd[b_key]
                # W is (out, in); A is (rank, in); B is (out, rank)
                # W + B @ A → (out, in)
                merged_w = tensor + b @ a
                merged[name] = merged_w
                hira_keys.add(a_key)
                hira_keys.add(b_key)
                continue

        merged[name] = tensor

    # Drop the A/B tensors we've merged
    for k in hira_keys:
        merged.pop(k, None)

    return merged


def rename_top_level_ln0(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Move blocks.0.ln0.* to top-level ln0.* (matches inference class layout)."""
    out: dict[str, torch.Tensor] = {}
    for name, tensor in sd.items():
        if name == "blocks.0.ln0.weight":
            out["ln0.weight"] = tensor
        elif name == "blocks.0.ln0.bias":
            out["ln0.bias"] = tensor
        else:
            out[name] = tensor
    return out


def squeeze_time_mix(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Squeeze the (1, 1, n_embed) time_mix_* tensors down to (n_embed,).

    The training checkpoint stores these with leading singleton dims
    to make broadcasting work in the training forward pass. At
    inference time we just want the flat vector.
    """
    out: dict[str, torch.Tensor] = {}
    for name, tensor in sd.items():
        if "time_mix_" in name and tensor.dim() == 3:
            out[name] = tensor.squeeze(0).squeeze(0).contiguous()
        else:
            out[name] = tensor
    return out


def drop_unused(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop tensors that the inference path does not consume.

    The training checkpoint may contain tensors like `time_mix_g` that
    are referenced during training but whose outputs are discarded
    in the inference forward pass. Including them in the Rust binary
    is harmless but wastes a few bytes and clutters the Rust-side
    weight map.
    """
    # time_mix_g exists in the checkpoint but xg is computed and never
    # used in the RWKV_TimeMix.forward() of rwkv_tc_hira_infer.py. We
    # keep it anyway for now so we can cross-check tensor coverage,
    # but it's marked in the log.
    return sd


def write_tensor(f, name: str, tensor: torch.Tensor) -> None:
    """Write one tensor record in the binary format."""
    # Ensure contiguous f32 (promote f16/bf16 if needed)
    tensor = tensor.detach().to(torch.float32).contiguous()
    name_bytes = name.encode("utf-8")
    dims = tuple(tensor.shape)
    data = tensor.numpy().tobytes()

    f.write(struct.pack("<I", len(name_bytes)))
    f.write(name_bytes)
    f.write(struct.pack("<I", len(dims)))
    for d in dims:
        f.write(struct.pack("<I", d))
    f.write(struct.pack("<I", DTYPE_F32))
    f.write(struct.pack("<Q", len(data)))
    f.write(data)


def write_binary(tensors: dict[str, torch.Tensor], out_path: Path) -> None:
    """Write the full binary file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(tensors)))
        for name in sorted(tensors.keys()):
            write_tensor(f, name, tensors[name])
        f.write(TRAILER)


def extract_model_config(config_path: Path) -> dict[str, int]:
    """Best-effort parse of an L3TC config .py file for hyperparameters."""
    text = config_path.read_text()
    cfg: dict[str, int] = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key in ("num_hidden_layer", "hidden_size", "intermediate_size", "rwkv_rank"):
            try:
                cfg[key] = int(value)
            except ValueError:
                pass
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="input .pth")
    parser.add_argument("--config", type=Path, required=True, help="L3TC config .py")
    parser.add_argument("--output", type=Path, required=True, help="output .bin")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2

    print(f"loading: {args.input}")
    sd = load_state_dict(args.input)
    print(f"  raw tensors: {len(sd)}")

    cfg = extract_model_config(args.config)
    print(f"  config: {cfg}")

    sd = apply_hira_merge(sd)
    print(f"  after HiRA merge: {len(sd)} tensors")

    sd = rename_top_level_ln0(sd)
    sd = squeeze_time_mix(sd)
    sd = drop_unused(sd)

    if args.verbose:
        print("  final tensor list:")
        for name in sorted(sd.keys()):
            shape = tuple(sd[name].shape)
            print(f"    {name:60s} {shape}")

    total_bytes = 0
    for name, t in sd.items():
        total_bytes += t.numel() * 4  # f32

    print(f"  total f32 bytes: {total_bytes:,}")
    write_binary(sd, args.output)
    size = args.output.stat().st_size
    print(f"wrote: {args.output} ({size:,} bytes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

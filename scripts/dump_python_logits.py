"""Dump per-token logits from the Python L3TC-200K reference for diff
against the Rust port.

Loads the L3TC checkpoint directly (no SLConfig dependency), runs the
same per-token forward loop the production compressor.py uses, and
writes a flat binary dump that the Rust diff side can read back.

Output format (little-endian):

    tokens.bin:
        n_tokens: u32
        tokens:   u32 * n_tokens

    logits.bin:
        n_tokens: u32
        vocab:    u32
        logits:   f32 * n_tokens * vocab   (row major: row i = logits at step i)

The first input token is BOS (id=2). At step i, we forward `tokens[i]`
through the model and dump the resulting logits — these are the
distributions over the *next* token. Decoding `tokens[i+1]` against
this distribution is what the AC encodes.

Run from the L3TC venv:

    cd vendor/L3TC && source .venv/bin/activate
    python ../../scripts/dump_python_logits.py \
        --checkpoint checkpoints/l3tc_checkpoints/l3tc_200k_bpe16k_c999_checkpoint0019.pth \
        --tokenizer dictionary/vocab_enwik8_bpe_16384_0.999/spm_enwik8_bpe_16384_0.999.model \
        --input ../../bench/corpora/enwik6 \
        --segment-bytes 4096 \
        --max-tokens 256 \
        --out-dir /tmp/l3tc_python_dump
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import sentencepiece as spm

# Make the L3TC sources importable so we can grab the model class.
HERE = Path(__file__).resolve().parent
L3TC_DIR = HERE.parent / "vendor" / "L3TC"
sys.path.insert(0, str(L3TC_DIR))

from models.RWKV_V4.rwkv_tc_hira_infer import RWKV_TC_HIRA_Infer_For_Script  # noqa: E402

# L3TC-200K hyperparameters from vendor/L3TC/config/l3tc/l3tc_200k.py
HIDDEN_SIZE = 96
NUM_LAYERS = 2
INTERMEDIATE_SIZE = 96
RWKV_RANK = 4
VOCAB_SIZE = 16384  # SPM vocab from the L3TC dictionary

BOS_ID = 2  # matches compressor.py preprocess


def load_model(checkpoint_path: Path) -> torch.nn.Module:
    """Load the L3TC-200K checkpoint into the For_Script inference model.

    Mirrors the load logic in vendor/L3TC/scripts/compressor.py:
      1. torch.load (weights_only=False for old checkpoints)
      2. squeeze time_mix_* tensors that have a leading singleton dim
      3. HiRA merge: W = W + B @ A for key/value/receptance in att/ffn
      4. Move blocks.0.ln0.* to top-level ln0.*
    """
    print(f"loading checkpoint: {checkpoint_path}")
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = raw.get("model", raw)

    # Strip any "module." prefix from DDP-trained checkpoints.
    cleaned = OrderedDict()
    for k, v in sd.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v
    sd = cleaned

    # Squeeze (1, 1, n_embed) time_mix_* down to (1, n_embed) —
    # exactly what compressor.py does (`value[0]`). The model class
    # expects shape (1, n_embed) so broadcasting works in the
    # forward pass.
    for key in list(sd.keys()):
        if "att.time_mix" in key or "ffn.time_mix" in key:
            v = sd[key]
            if v.dim() == 3:
                sd[key] = v[0]

    # HiRA merge: W = W + B @ A for key/value/receptance.
    for key in list(sd.keys()):
        if not key.endswith(".weight"):
            continue
        base = key[: -len(".weight")]
        for proj in ("key", "value", "receptance"):
            if base.endswith("." + proj):
                a_key = f"{base}_A.weight"
                b_key = f"{base}_B.weight"
                if a_key in sd and b_key in sd:
                    sd[key] = sd[key] + sd[b_key] @ sd[a_key]
                    sd.pop(a_key)
                    sd.pop(b_key)
                break

    # Move blocks.0.ln0.* to top-level ln0.*
    if "blocks.0.ln0.weight" in sd:
        sd["ln0.weight"] = sd.pop("blocks.0.ln0.weight")
        sd["ln0.bias"] = sd.pop("blocks.0.ln0.bias")

    model = RWKV_TC_HIRA_Infer_For_Script(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        intermediate_size=INTERMEDIATE_SIZE,
        rwkv_rank=RWKV_RANK,
    )
    load_out = model.load_state_dict(sd, strict=False)
    print(f"  loaded: {load_out}")
    model.eval()
    return model


def encode_first_segment(
    tokenizer: spm.SentencePieceProcessor,
    input_path: Path,
    segment_bytes: int,
    max_tokens: int,
) -> list[int]:
    """Tokenize the first `segment_bytes` of `input_path` exactly the
    way compressor.py's preprocess does it: prepend BOS=2, then SPM
    pieces. Truncate to `max_tokens` if asked."""
    raw = input_path.read_bytes()[:segment_bytes]
    text = raw.decode("utf-8", errors="strict")
    proto = tokenizer.encode(text, out_type="immutable_proto")
    tokens = [BOS_ID]
    for piece in proto.pieces:
        if piece.begin == piece.end:
            continue
        tokens.append(piece.id)
    if max_tokens > 0:
        tokens = tokens[:max_tokens]
    print(f"  tokens: {len(tokens)} (first 10: {tokens[:10]})")
    return tokens


def dump_logits(
    model: torch.nn.Module,
    tokens: list[int],
    out_dir: Path,
) -> None:
    """Forward each token through the model and write per-token logits
    to `out_dir/logits.bin` in the format described at the top of this
    file."""
    out_dir.mkdir(parents=True, exist_ok=True)

    n_tokens = len(tokens)
    vocab = VOCAB_SIZE

    # Write tokens.bin (input token IDs).
    with (out_dir / "tokens.bin").open("wb") as f:
        f.write(struct.pack("<I", n_tokens))
        for t in tokens:
            f.write(struct.pack("<I", t))

    # Forward and dump.
    device = torch.device("cpu")
    hidden = model.forward_initialzation(batch_size=1, device=device)

    out_path = out_dir / "logits.bin"
    with out_path.open("wb") as f:
        f.write(struct.pack("<I", n_tokens))
        f.write(struct.pack("<I", vocab))

        with torch.no_grad():
            for i, tok in enumerate(tokens):
                input_t = torch.tensor([tok], dtype=torch.long, device=device)
                logits, hidden = model(input_t, hidden)
                # logits shape: (1, vocab_size)
                row = logits[0].cpu().numpy().astype("<f4").tobytes()
                assert len(row) == vocab * 4
                f.write(row)

                if (i + 1) % 32 == 0 or i == n_tokens - 1:
                    print(f"  forward {i + 1}/{n_tokens}")

    print(f"wrote: {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--segment-bytes", type=int, default=4096)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/l3tc_python_dump"))
    args = p.parse_args()

    if not args.checkpoint.exists():
        print(f"checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    if not args.tokenizer.exists():
        print(f"tokenizer not found: {args.tokenizer}", file=sys.stderr)
        return 2
    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2

    print(f"loading tokenizer: {args.tokenizer}")
    sp = spm.SentencePieceProcessor()
    sp.Load(str(args.tokenizer))

    model = load_model(args.checkpoint)

    tokens = encode_first_segment(sp, args.input, args.segment_bytes, args.max_tokens)
    dump_logits(model, tokens, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

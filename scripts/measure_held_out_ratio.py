"""Measure an entropy-bound compression ratio for a held-out text
corpus under a trained RWKV-v4 model.

Used by the training entrypoint (cdk/docker/training/train_entrypoint.sh)
during Spike 1 to decide `codec=l3tc` vs `codec=zstd_fallback` and to
emit the headline ratio number in v{N}.metadata.json.

This is the *theoretical* entropy bound:
    bits_per_token = -mean(log2 p_true[t])   (over all positions in held-out)
    ratio = (bits_per_byte) / 8 = (bits_per_token * tokens / total_bytes) / 8

Actual coded bytes (after arithmetic coding) are within ~1% of this
bound on sequences longer than a few KB. For spike go/no-go, the
entropy bound is a faithful stand-in and avoids needing the Rust
runtime inside the Python training container.

Usage:
    python scripts/measure_held_out_ratio.py \\
        --val-file /tmp/val.txt \\
        --checkpoint /tmp/checkpoint_final.pth \\
        --tokenizer /tmp/spm.model
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import sentencepiece as spm


def load_model(checkpoint_path: Path, num_layers: int, vocab_size: int):
    """Load the RWKV model using the same factory as train_l3tc_phase11.

    We import at call time so that this script can be used standalone
    without pulling vendor/L3TC/ into sys.path globally."""
    sys.path.insert(0, "/app")  # so `scripts.*` resolves inside the container
    sys.path.insert(0, "/app/vendor/L3TC")
    try:
        # train_l3tc_phase11.py defines the model architecture we need.
        # Importing it triggers its module-level code but we just need
        # the model class + factory.
        from scripts.train_l3tc_phase11 import build_model  # type: ignore
    except ImportError:
        # Fallback — try alternate import locations
        raise

    device = torch.device("cpu")  # load on CPU first; caller moves to GPU
    model = build_model(device, num_layers=num_layers, vocab_size=vocab_size)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def iterate_segments(tokens: list[int], segment_len: int):
    """Yield (input, target) segments of length `segment_len`. Last
    partial segment is dropped (trivial bias for long val files)."""
    for i in range(0, len(tokens) - segment_len, segment_len):
        yield tokens[i : i + segment_len]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val-file", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--segment-len", type=int, default=2048)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # Load held-out bytes + tokenize.
    raw = args.val_file.read_bytes()
    original_bytes = len(raw)
    text = raw.decode("utf-8", errors="replace")

    sp = spm.SentencePieceProcessor()
    sp.Load(str(args.tokenizer))
    tokens = sp.EncodeAsIds(text)
    total_tokens = len(tokens)
    if total_tokens < args.segment_len + 1:
        print(
            f"ERROR: held-out too small ({total_tokens} tokens, need > {args.segment_len}) — "
            f"no meaningful entropy estimate possible.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Load model + run inference to accumulate log-likelihood.
    model = load_model(args.checkpoint, args.num_layers, args.vocab_size)
    model = model.to(args.device)

    total_neg_log2 = 0.0
    total_positions = 0

    with torch.no_grad():
        for seg in iterate_segments(tokens, args.segment_len):
            inp = torch.tensor(seg[:-1], dtype=torch.long, device=args.device).unsqueeze(0)
            tgt = torch.tensor(seg[1:], dtype=torch.long, device=args.device).unsqueeze(0)

            # Forward pass: expected output shape (1, seq, vocab).
            # The train_l3tc_phase11.build_model output may be a tuple.
            out = model(inp)
            if isinstance(out, (tuple, list)):
                out = out[0]
            logits = out  # (1, seq, vocab)

            log_probs = F.log_softmax(logits.float(), dim=-1)
            # Gather log-prob of the true next token at each position.
            gathered = log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (1, seq)
            # -log2 p = -ln(p) / ln(2). We have log_softmax = ln p.
            seg_nll_nats = -gathered.sum().item()
            seg_neg_log2 = seg_nll_nats / math.log(2)

            total_neg_log2 += seg_neg_log2
            total_positions += inp.numel()

    bits_per_token = total_neg_log2 / max(1, total_positions)
    # Approximation: bytes-per-token ratio = original_bytes / total_tokens.
    # Real coding operates on tokens so this is exact.
    bytes_per_token = original_bytes / max(1, total_tokens)
    # Ratio = compressed_bytes / original_bytes = (bits_per_token / 8) / bytes_per_token
    ratio = (bits_per_token / 8.0) / max(1e-12, bytes_per_token)

    # Emit a single line for easy capture: RATIO=<float>.
    # Also emit human-readable summary to stderr for logs.
    print(f"{ratio:.6f}")
    print(
        f"[measure] tokens={total_tokens}  "
        f"bits/token={bits_per_token:.4f}  "
        f"bytes/token={bytes_per_token:.4f}  "
        f"original_bytes={original_bytes}  "
        f"ratio={ratio:.6f}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

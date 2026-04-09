"""Compute the theoretical entropy-optimal ratio from dumped logits.

The Python L3TC ratio of 0.1665 on enwik6 is reported as 'compressed
bytes / input bytes' but the underlying number comes from
`compressor.py`'s `entropy_sum` (the sum of -log2(p[next_token]) over
all token positions). The actual AC write path is commented out in
compressor.py. So 'Python ratio' is the theoretical entropy lower
bound, not real coded bytes.

This script computes the same entropy from a dumped logits.bin to
let us compare apples to apples between Python and Rust forward
passes.

Usage:
    python scripts/compute_entropy.py --dump-dir /tmp/l3tc_rust_dump --input-bytes 4096
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def load_logits(path: Path) -> np.ndarray:
    data = path.read_bytes()
    n_tokens = struct.unpack_from("<I", data, 0)[0]
    vocab = struct.unpack_from("<I", data, 4)[0]
    return np.frombuffer(data, dtype="<f4", offset=8).reshape(n_tokens, vocab)


def load_tokens(path: Path) -> np.ndarray:
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 0)[0]
    return np.frombuffer(data, dtype="<u4", offset=4, count=n)


def compute_entropy(logits: np.ndarray, tokens: np.ndarray) -> tuple[float, int]:
    """For each position i in [0, n-1], compute -log2 P(tokens[i+1] | logits[i])
    using a stable softmax. Return (total bits, count)."""
    n = logits.shape[0]
    total_bits = 0.0
    count = 0
    for i in range(n - 1):
        x = logits[i]
        m = float(x.max())
        ex = np.exp(x - m)
        s = float(ex.sum())
        # log2 P(tokens[i+1])
        p = ex[int(tokens[i + 1])] / s
        if p <= 0.0:
            print(f"warn: zero prob at step {i}", file=sys.stderr)
            continue
        total_bits += -np.log2(p)
        count += 1
    return total_bits, count


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump-dir", type=Path, default=Path("/tmp/l3tc_rust_dump"))
    p.add_argument(
        "--input-bytes",
        type=int,
        default=4096,
        help="Original number of input bytes the dump represents (used "
        "to report a 'ratio' figure comparable to Python's 0.1665).",
    )
    args = p.parse_args()

    tokens = load_tokens(args.dump_dir / "tokens.bin")
    logits = load_logits(args.dump_dir / "logits.bin")
    print(f"loaded {logits.shape[0]} tokens × {logits.shape[1]} vocab")

    total_bits, count = compute_entropy(logits, tokens)
    total_bytes = total_bits / 8.0
    print(f"entropy total: {total_bits:.2f} bits over {count} prediction steps")
    print(f"entropy bytes: {total_bytes:.2f}")
    print(f"avg bits/token: {total_bits / count:.4f}")
    if args.input_bytes > 0:
        print(f"entropy ratio (vs {args.input_bytes} bytes): {total_bytes / args.input_bytes:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

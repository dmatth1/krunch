"""Diff per-token logit dumps from the Python L3TC reference and the
Rust port. Reads two `logits.bin` files in the format produced by
`dump_python_logits.py` and `l3tc dump-logits` and reports per-token
L_inf and L2 differences. Halts at the first token that exceeds the
threshold and prints a small slice of the worst-disagreeing entries.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def load_logits(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    n_tokens = struct.unpack_from("<I", data, 0)[0]
    vocab = struct.unpack_from("<I", data, 4)[0]
    expected = 8 + n_tokens * vocab * 4
    if len(data) != expected:
        raise ValueError(
            f"{path}: size mismatch, header says {n_tokens}x{vocab} f32 "
            f"({expected} bytes), file is {len(data)} bytes"
        )
    arr = np.frombuffer(data, dtype="<f4", offset=8).reshape(n_tokens, vocab)
    return arr, np.array([n_tokens, vocab])


def load_tokens(path: Path) -> np.ndarray:
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 0)[0]
    return np.frombuffer(data, dtype="<u4", offset=4, count=n)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--python-dir", type=Path, default=Path("/tmp/l3tc_python_dump"))
    p.add_argument("--rust-dir", type=Path, default=Path("/tmp/l3tc_rust_dump"))
    p.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="Per-token L_inf threshold; first token over this is the "
        "divergence point.",
    )
    p.add_argument(
        "--show-top",
        type=int,
        default=8,
        help="At the divergence token, print the top-K entries with the "
        "biggest absolute disagreement.",
    )
    args = p.parse_args()

    py_tokens = load_tokens(args.python_dir / "tokens.bin")
    rs_tokens = load_tokens(args.rust_dir / "tokens.bin")
    if not np.array_equal(py_tokens, rs_tokens):
        print("ERROR: token sequences differ", file=sys.stderr)
        print(f"  python: {py_tokens[:20]}...", file=sys.stderr)
        print(f"  rust:   {rs_tokens[:20]}...", file=sys.stderr)
        return 2

    py, py_shape = load_logits(args.python_dir / "logits.bin")
    rs, rs_shape = load_logits(args.rust_dir / "logits.bin")
    if py.shape != rs.shape:
        print(f"ERROR: shape mismatch python={py.shape} rust={rs.shape}", file=sys.stderr)
        return 2

    print(f"comparing {py.shape[0]} tokens × {py.shape[1]} vocab")
    print(f"threshold (per-token L_inf): {args.threshold}")
    print()
    print(f"{'tok':>4} {'id':>6} {'L_inf':>12} {'L2':>12} {'argmax_py':>10} {'argmax_rs':>10}")

    div_idx = -1
    for i in range(py.shape[0]):
        diff = py[i] - rs[i]
        linf = float(np.max(np.abs(diff)))
        l2 = float(np.linalg.norm(diff))
        amp = int(np.argmax(py[i]))
        amr = int(np.argmax(rs[i]))
        marker = ""
        if linf > args.threshold and div_idx < 0:
            div_idx = i
            marker = " <-- first divergence"
        if i < 32 or div_idx == i or i % 32 == 0:
            print(f"{i:>4} {int(py_tokens[i]):>6} {linf:>12.6e} {l2:>12.6e} {amp:>10} {amr:>10}{marker}")
        if div_idx >= 0 and i > div_idx + 4:
            break

    print()
    if div_idx < 0:
        print(f"OK — no divergence above {args.threshold} in {py.shape[0]} tokens")
        # Still show overall stats.
        all_linf = float(np.max(np.abs(py - rs)))
        print(f"  max L_inf across all tokens: {all_linf:.6e}")
        return 0

    print(f"first divergence at token {div_idx} (input id {int(py_tokens[div_idx])})")
    diff = py[div_idx] - rs[div_idx]
    abs_diff = np.abs(diff)
    top = np.argsort(-abs_diff)[: args.show_top]
    print(f"top-{args.show_top} disagreeing logits at token {div_idx}:")
    print(f"  {'vocab_id':>10} {'python':>14} {'rust':>14} {'diff':>14}")
    for j in top:
        print(f"  {int(j):>10} {py[div_idx, j]:>14.6f} {rs[div_idx, j]:>14.6f} {diff[j]:>14.6e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

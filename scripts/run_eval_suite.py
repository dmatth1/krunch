"""Run the Rust compressor on every file in the eval suite.

Takes a .bin checkpoint + SPM tokenizer and runs `l3tc compress
--verify` on each file in bench/corpora/eval_suite/. Produces a
ratio + speed comparison table.

Usage:
    python scripts/run_eval_suite.py \
        --model l3tc-rust/checkpoints/l3tc_200k.bin \
        --tokenizer vendor/L3TC/dictionary/vocab_enwik8_bpe_16384_0.999/spm_enwik8_bpe_16384_0.999.model

    # Compare two models side by side:
    python scripts/run_eval_suite.py \
        --model l3tc-rust/checkpoints/l3tc_200k.bin \
        --model2 checkpoints-phase11/pile_pass2_epoch5.bin \
        --tokenizer vendor/L3TC/dictionary/vocab_enwik8_bpe_16384_0.999/spm_enwik8_bpe_16384_0.999.model \
        --tokenizer2 tokenizer_pile_32k/spm_pile_bpe_32768.model
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def compress_file(
    l3tc_bin: Path,
    model: Path,
    tokenizer: Path,
    input_file: Path,
) -> dict:
    """Run l3tc compress --verify --time and parse the output."""
    with tempfile.NamedTemporaryFile(suffix=".l3tc", delete=True) as tmp:
        cmd = [
            str(l3tc_bin),
            "compress",
            str(input_file),
            "--model", str(model),
            "--tokenizer", str(tokenizer),
            "-o", tmp.name,
            "--verify",
            "--time",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        output = result.stdout + "\n" + result.stderr
        if result.returncode != 0:
            return {"error": output.strip()[:200]}

        ratio_m = re.search(r"ratio\s+([\d.]+)", output)
        speed_m = re.search(r"compress:\s+[\d.]+s\s+\(([\d.]+)\s+KB/s\)", output)
        if not speed_m:
            speed_m = re.search(r"([\d.]+)\s+KB/s", output)
        verify_m = re.search(r"round-trip:\s+(\w+)", output)

        return {
            "ratio": float(ratio_m.group(1)) if ratio_m else None,
            "compress_kbs": float(speed_m.group(1)) if speed_m else None,
            "round_trip": verify_m.group(1) if verify_m else "unknown",
            "input_bytes": input_file.stat().st_size,
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="Primary .bin checkpoint")
    p.add_argument("--tokenizer", type=Path, required=True, help="Primary SPM .model")
    p.add_argument("--model2", type=Path, default=None, help="Comparison .bin checkpoint")
    p.add_argument("--tokenizer2", type=Path, default=None, help="Comparison SPM .model (defaults to --tokenizer)")
    p.add_argument("--l3tc-bin", type=Path, default=Path("l3tc-rust/target/release/l3tc"))
    p.add_argument("--eval-dir", type=Path, default=Path("bench/corpora/eval_suite"))
    p.add_argument("--skip-large", type=float, default=50.0, help="Skip files larger than N MB")
    args = p.parse_args()

    # Resolve all paths to absolute so we don't depend on cwd
    args.l3tc_bin = args.l3tc_bin.resolve()
    args.model = args.model.resolve()
    args.tokenizer = args.tokenizer.resolve()
    args.eval_dir = args.eval_dir.resolve()
    if args.model2:
        args.model2 = args.model2.resolve()
    if args.tokenizer2:
        args.tokenizer2 = args.tokenizer2.resolve()

    if not args.l3tc_bin.exists():
        print(f"ERROR: l3tc binary not found: {args.l3tc_bin}")
        sys.exit(1)
    if not args.eval_dir.exists():
        print(f"ERROR: eval suite not found: {args.eval_dir}")
        print("Run: python scripts/build_eval_suite.py")
        sys.exit(1)

    tok2 = args.tokenizer2 or args.tokenizer
    files = sorted(args.eval_dir.iterdir())
    skip_bytes = int(args.skip_large * 1e6)

    # Header
    if args.model2:
        print(f"{'file':<25s} {'MB':>6} {'ratio1':>7} {'KB/s1':>7} {'ratio2':>7} {'KB/s2':>7} {'delta':>7}")
        print("-" * 75)
    else:
        print(f"{'file':<25s} {'MB':>6} {'ratio':>7} {'KB/s':>7} {'round-trip':>11}")
        print("-" * 62)

    for f in files:
        if f.stat().st_size > skip_bytes:
            print(f"{f.name:<25s} {'SKIP (too large)':>40}")
            continue

        r1 = compress_file(args.l3tc_bin, args.model, args.tokenizer, f)
        if "error" in r1 or r1.get("ratio") is None:
            err = r1.get("error", "parse failure")
            print(f"{f.name:<25s} ERROR: {err}")
            continue

        mb = r1["input_bytes"] / 1e6

        if args.model2:
            r2 = compress_file(args.l3tc_bin, args.model2, tok2, f)
            if "error" in r2:
                print(f"{f.name:<25s} {mb:>6.2f} {r1['ratio']:>7.4f} {r1['compress_kbs']:>7.1f} {'ERROR':>7} {'':>7} {'':>7}")
            else:
                delta = r2["ratio"] - r1["ratio"]
                sign = "+" if delta > 0 else ""
                print(f"{f.name:<25s} {mb:>6.2f} {r1['ratio']:>7.4f} {r1['compress_kbs']:>7.1f} {r2['ratio']:>7.4f} {r2['compress_kbs']:>7.1f} {sign}{delta:>6.4f}")
        else:
            print(f"{f.name:<25s} {mb:>6.2f} {r1['ratio']:>7.4f} {r1['compress_kbs']:>7.1f} {r1['round_trip']:>11}")


if __name__ == "__main__":
    main()

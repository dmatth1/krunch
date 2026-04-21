"""Cross-validate the 7 SPMs on representative real corpus slices.

The first cross-validation used tiny hand-written snippets. This one
samples ~500 KB of real content from each domain's own corpus and
measures B/T across all 7 tokenizers — a much more reliable specialization
check.

Usage:
    vendor/L3TC/.venv/bin/python scripts/specialist_spm_crossvalidate_real.py
"""
from __future__ import annotations

from pathlib import Path
import random

import sentencepiece as spm

DOMAINS = ["prose", "code", "structured", "logs", "tabular", "markup", "fallback"]
SAMPLE_KB = 500


def read_sample(domain_dir: Path, size_bytes: int, seed: int = 7) -> str:
    """Grab the first ~size_bytes of content from the domain, newline-aligned."""
    rng = random.Random(seed)
    parts = sorted(domain_dir.glob("from_*.txt"))
    if not parts:
        return ""
    # Pick a random part and read a random offset.
    src = rng.choice(parts)
    size = src.stat().st_size
    offset = rng.randint(0, max(size - size_bytes - 1, 0))
    with open(src, "rb") as f:
        f.seek(offset)
        _ = f.readline()  # align
        buf = f.read(size_bytes)
    last = buf.rfind(b"\n")
    if last > 0:
        buf = buf[: last + 1]
    return buf.decode("utf-8", errors="replace")


def main():
    base = Path("data/specialists")
    tokenizers = {}
    for d in DOMAINS:
        sp = spm.SentencePieceProcessor()
        sp.load(str(base / d / "spm.model"))
        tokenizers[d] = sp

    samples = {d: read_sample(base / d, SAMPLE_KB * 1024) for d in DOMAINS}
    for d, s in samples.items():
        print(f"  {d:12s} sample: {len(s.encode('utf-8')) / 1024:.1f} KB")

    print(f"\n=== B/T on real corpus slices (~{SAMPLE_KB} KB each) ===\n")
    header = "content \\ tokenizer".ljust(16) + "".join(d[:8].ljust(10) for d in DOMAINS)
    print(header)
    print("-" * len(header))

    wins = 0
    for content_d in DOMAINS:
        sample = samples[content_d]
        sample_bytes = len(sample.encode("utf-8"))
        row = content_d[:14].ljust(16)
        best_d, best_bt = None, 0.0
        for d in DOMAINS:
            sp = tokenizers[d]
            ids = sp.encode_as_ids(sample)
            bt = sample_bytes / max(len(ids), 1)
            if bt > best_bt:
                best_bt = bt
                best_d = d
            marker = "*" if d == content_d else " "
            row += f"{marker}{bt:.2f}".ljust(10)
        row += f"   best: {best_d}"
        print(row)
        if best_d == content_d:
            wins += 1
        else:
            print(f"   NOTE: {content_d} SPM did not win on {content_d} content "
                  f"({best_bt:.2f} for {best_d} vs "
                  f"{sample_bytes / max(len(tokenizers[content_d].encode_as_ids(sample)), 1):.2f} "
                  f"for {content_d})")

    print(f"\nSpecialization score: {wins}/{len(DOMAINS)} domains won by their own SPM")
    print("(fallback winning its own pile is expected — it's trained on mixed content)")


if __name__ == "__main__":
    main()

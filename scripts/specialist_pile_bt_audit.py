"""SPM-based B/T audit: for each domain's pile, check whether the domain's
own SPM wins vs the other 6 SPMs on real content sampled from THAT pile.

Uses a weighted random sample where each source file contributes a
random slice proportional to its size — so the audit reflects the
pile's actual content distribution, not whichever random offset landed
in a bleed region (the trap I hit before).

Task 29 follow-up. If a domain's own SPM does NOT win on its own pile's
representative sample, that's a flag for contamination or SPM mismatch.

Usage:
    vendor/L3TC/.venv/bin/python scripts/specialist_pile_bt_audit.py
"""
from __future__ import annotations

import random
from pathlib import Path

import sentencepiece as spm

DOMAINS = ["prose", "code", "structured", "logs", "tabular", "markup", "fallback"]
SAMPLE_PER_DOMAIN_KB = 1024  # 1 MB per domain, weighted across its source files
SEED = 7


def load_tokenizers():
    tks = {}
    for d in DOMAINS:
        sp = spm.SentencePieceProcessor()
        sp.load(f"data/specialists/{d}/spm.model")
        tks[d] = sp
    return tks


def weighted_sample(domain_dir: Path, total_bytes: int, seed: int) -> str:
    """Pull `total_bytes` of text from `domain_dir/from_*.txt`, weighted
    by per-file size (so bigger files contribute proportionally more)."""
    rng = random.Random(seed)
    parts = sorted(domain_dir.glob("from_*.txt"))
    if not parts:
        return ""
    sizes = [p.stat().st_size for p in parts]
    grand = sum(sizes)
    pieces = []
    for p, sz in zip(parts, sizes):
        if sz < 4096:
            continue
        chunk = int(total_bytes * sz / grand)
        if chunk < 512:
            continue
        with open(p, "rb") as f:
            offset = rng.randint(0, max(sz - chunk - 1, 0))
            f.seek(offset)
            f.readline()  # align to newline
            buf = f.read(chunk)
        last = buf.rfind(b"\n")
        if last > 0:
            buf = buf[: last + 1]
        pieces.append(buf.decode("utf-8", errors="replace"))
    return "".join(pieces)


def main():
    tks = load_tokenizers()
    base = Path("data/specialists")

    print(f"=== Per-pile SPM B/T audit (sample {SAMPLE_PER_DOMAIN_KB} KB per domain, weighted) ===\n")
    header = "pile            " + "".join(d[:8].ljust(10) for d in DOMAINS) + "  winner"
    print(header)
    print("-" * len(header))

    wins = 0
    for content_d in DOMAINS:
        sample = weighted_sample(base / content_d, SAMPLE_PER_DOMAIN_KB * 1024, seed=SEED)
        if not sample:
            print(f"{content_d:16s}(no content)")
            continue
        sample_bytes = len(sample.encode("utf-8"))
        row = content_d[:14].ljust(16)
        best_d, best_bt = None, 0.0
        for d in DOMAINS:
            ids = tks[d].encode_as_ids(sample)
            bt = sample_bytes / max(len(ids), 1)
            if bt > best_bt:
                best_bt = bt
                best_d = d
            marker = "*" if d == content_d else " "
            row += f"{marker}{bt:.2f}".ljust(10)
        row += f"  {best_d}"
        if best_d != content_d:
            row += " <-- not own"
        else:
            wins += 1
        print(row)

    print(f"\nSpecialization: {wins}/{len(DOMAINS)} piles won by their own SPM")
    print("(fallback is a mixed-content pile, losing to a more specific SPM on its content is fine)")


if __name__ == "__main__":
    main()

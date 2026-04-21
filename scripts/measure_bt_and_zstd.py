"""Cross-domain B/T matrix + zstd-22 baseline for the v0.1.0 release gate.

Produces:
  1. An 8×7 B/T matrix: 7 specialist SPMs + L3TC enwik8 SPM, evaluated on each
     of the 7 domain samples from bench/v010_benchmark/. Diagonal should be
     highest (specialist hits its own domain best); off-diagonal shows how
     much specialization actually helps per domain.
  2. Per-domain zstd-22 ratio: the new single-baseline target (replaces the
     earlier "best of 4 traditional compressors" framing). zstd-22 is the
     ratio users actually compare against.

Exclusions:
  The fallback directory contains deliberately-uncompressible files
  (random_*.bin, highentropy_*.bin, hex_*.txt) that distort aggregate
  ratios. We compute the fallback zstd number twice: with and without
  those binary/high-entropy files, so the "real" bar is visible.
"""

from __future__ import annotations

import subprocess
import statistics
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "bench" / "v010_benchmark"
SPM_DIR = REPO / "data" / "specialists"
ENWIK8_SPM = REPO / "vendor" / "L3TC" / "dictionary" / \
    "vocab_enwik8_bpe_16384_0.999" / "spm_enwik8_bpe_16384_0.999.model"

DOMAINS = ["prose", "code", "structured", "logs", "tabular", "markup", "fallback"]
SPMS = {d: SPM_DIR / d / "spm.model" for d in DOMAINS}
SPMS["enwik8"] = ENWIK8_SPM

# Files we exclude from the fallback "text" measurement — deliberately
# uncompressible content that is not representative of real text.
NONTEXT_PREFIXES = ("highentropy_", "random_", "hex_")


def list_domain_files(domain: str) -> list[Path]:
    """All files in the v010_benchmark domain directory."""
    d = BENCH / domain
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.is_file())


def is_nontext(path: Path) -> bool:
    return any(path.name.startswith(p) for p in NONTEXT_PREFIXES)


def load_domain_sample(domain: str, exclude_nontext: bool = True) -> bytes:
    """Concatenate all domain files into one byte string."""
    parts: list[bytes] = []
    for f in list_domain_files(domain):
        if exclude_nontext and is_nontext(f):
            continue
        parts.append(f.read_bytes())
    return b"".join(parts)


def decode_utf8_best_effort(buf: bytes) -> str:
    return buf.decode("utf-8", errors="replace")


@dataclass
class BTResult:
    tokenizer: str
    domain: str
    bytes: int
    tokens: int
    bt: float  # bytes per token


def bt_matrix() -> list[BTResult]:
    """For each (tokenizer, domain) pair, encode the domain sample with the
    tokenizer and record bytes/tokens ratio. Fallback domain excludes the
    high-entropy binary files (can't meaningfully tokenize random bytes)."""
    results: list[BTResult] = []
    # Pre-load all domain samples once.
    samples: dict[str, tuple[bytes, str]] = {}
    for d in DOMAINS:
        raw = load_domain_sample(d, exclude_nontext=(d == "fallback"))
        samples[d] = (raw, decode_utf8_best_effort(raw))

    for tok_name, tok_path in SPMS.items():
        sp = spm.SentencePieceProcessor()
        sp.Load(str(tok_path))
        for d in DOMAINS:
            raw, text = samples[d]
            ids = sp.EncodeAsIds(text)
            bt = len(raw) / max(1, len(ids))
            results.append(
                BTResult(tokenizer=tok_name, domain=d, bytes=len(raw),
                         tokens=len(ids), bt=bt)
            )
    return results


@dataclass
class ZstdResult:
    domain: str
    files: int
    raw_bytes: int
    compressed_bytes: int
    ratio: float


def zstd22_ratio(domain: str, exclude_nontext: bool = False) -> ZstdResult:
    """Concatenate domain files and zstd --long=27 --ultra -22 them. Use
    --long=27 to let zstd see the whole corpus as one window."""
    parts: list[bytes] = []
    n = 0
    for f in list_domain_files(domain):
        if exclude_nontext and is_nontext(f):
            continue
        parts.append(f.read_bytes())
        n += 1
    raw = b"".join(parts)
    if not raw:
        return ZstdResult(domain=domain, files=0, raw_bytes=0,
                          compressed_bytes=0, ratio=0.0)
    proc = subprocess.run(
        ["zstd", "--long=27", "--ultra", "-22", "--stdout", "--quiet"],
        input=raw, capture_output=True, check=True,
    )
    c = len(proc.stdout)
    return ZstdResult(domain=domain, files=n, raw_bytes=len(raw),
                      compressed_bytes=c, ratio=c / len(raw))


def main() -> None:
    print("=" * 70)
    print("cross-domain B/T matrix")
    print("=" * 70)
    print()
    results = bt_matrix()
    # Pivot into [tokenizer][domain] = bt
    tok_names = list(SPMS.keys())
    header = f"{'tokenizer':<12s} " + "  ".join(f"{d[:9]:>9s}" for d in DOMAINS)
    print(header)
    print("-" * len(header))
    for tok in tok_names:
        row = f"{tok:<12s} "
        for d in DOMAINS:
            r = next(r for r in results if r.tokenizer == tok and r.domain == d)
            row += f"  {r.bt:>8.3f}"
        print(row)
    print()
    print("(B/T = bytes per token; higher = more compact encoding for this domain)")
    print("(diagonal should be the max per column if specialization worked)")
    print()

    # Per-column diagnostic: is the specialist the best on its own domain?
    print("per-domain: specialist vs best-other-tokenizer")
    print("-" * 70)
    for d in DOMAINS:
        col = [(r.tokenizer, r.bt) for r in results if r.domain == d]
        col.sort(key=lambda x: -x[1])
        best_tok, best_bt = col[0]
        own = next(bt for t, bt in col if t == d)
        own_rank = [t for t, _ in col].index(d) + 1
        delta_vs_enwik8 = own - next(bt for t, bt in col if t == "enwik8")
        note = "✓" if best_tok == d else f"✗ beaten by {best_tok}"
        print(f"  {d:<11s} specialist={own:6.3f}  best={best_bt:6.3f} ({best_tok})  "
              f"rank={own_rank}/{len(col)}  Δ-vs-enwik8={delta_vs_enwik8:+.3f}  {note}")
    print()

    print("=" * 70)
    print("zstd -22 --long=27 per-domain ratio  (new v0.1.0 baseline)")
    print("=" * 70)
    print()
    print(f"{'domain':<12s} {'files':>5s}  {'raw MB':>8s}  {'comp MB':>8s}  {'ratio':>7s}")
    print("-" * 52)
    zres: list[ZstdResult] = []
    for d in DOMAINS:
        r = zstd22_ratio(d, exclude_nontext=False)
        zres.append(r)
        raw_mb = r.raw_bytes / 1e6
        c_mb = r.compressed_bytes / 1e6
        print(f"  {d:<10s} {r.files:>5d}  {raw_mb:>8.3f}  {c_mb:>8.3f}  {r.ratio:>7.4f}")
    print()
    # fallback without non-text files, for the "text-only" read.
    r_fb_text = zstd22_ratio("fallback", exclude_nontext=True)
    print(f"  fallback (text-only, excluding random/hex/highentropy): "
          f"{r_fb_text.files} files, ratio {r_fb_text.ratio:.4f}")
    print()


if __name__ == "__main__":
    main()

"""Phase 14 task 7: source a markup corpus (HTML + Markdown + LaTeX).

Task 5 audit showed only ~2 GB of markup bleed from mixed-content
sources — not real HTML/Markdown/LaTeX. The markup specialist needs
dedicated sources:

  - HTML: The Stack v2 HTML files OR Common Crawl WARC samples
  - Markdown: GitHub README.md via lumees/github-code-2025 "markdown" split
  - LaTeX: arXiv public bulk (arXiv ID packages are a few GB)

Usage:
    python scripts/phase14_source_markup.py \\
        --output-dir data/specialists/markup \\
        --target-gb 10
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path


def _dedup_seen() -> set[str]:
    return set()


def _is_new(seen: set[str], text: str) -> bool:
    h = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()
    if h in seen:
        return False
    seen.add(h)
    return True


def stream_stack_v2(out_path: Path, language: str, target_bytes: int,
                    seen: set[str], seed: int = 2026):
    """Stream a language-labeled slice of the Stack v2 (HF-gated)."""
    from datasets import load_dataset
    print(f"\n--- Stack v2 {language} (target: {target_bytes/1e9:.1f} GB) ---")
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        tok_path = Path.home() / ".hf-token"
        if tok_path.exists():
            tok = tok_path.read_text().strip()
    ds_names = [
        ("bigcode/the-stack-v2-dedup", f"data/{language.lower()}"),
        ("bigcode/starcoderdata", language.lower()),
        ("lumees/github-code-2025-language-split", language.lower()),
    ]
    written = 0
    docs = 0
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for name, data_dir in ds_names:
            if written >= target_bytes:
                break
            try:
                if data_dir.startswith("data/"):
                    ds = load_dataset(name, data_dir=data_dir,
                                      split="train", streaming=True, token=tok)
                else:
                    ds = load_dataset(name, data_dir, split="train",
                                      streaming=True, token=tok)
            except Exception as e:
                print(f"  {name} {data_dir}: SKIP ({e})")
                continue
            ds = ds.shuffle(seed=seed, buffer_size=5_000)
            for ex in ds:
                if written >= target_bytes:
                    break
                content = ex.get("content") or ex.get("text") or ""
                if not content or len(content) < 100 or len(content) > 200_000:
                    continue
                if not _is_new(seen, content):
                    continue
                f.write(content)
                f.write("\n\n")  # doc separator
                b = len(content.encode("utf-8", errors="replace")) + 2
                written += b
                docs += 1
                if docs % 5_000 == 0:
                    print(f"  {docs:,} docs, {written/1e9:.2f} GB, {int(time.time()-t0)}s")
            print(f"  {name}: now {written/1e9:.2f} GB, {docs:,} docs")
            break  # first source that worked wins
    print(f"  done: {docs:,} docs, {written/1e9:.2f} GB, {int(time.time()-t0)}s")
    return written


def download_arxiv_bulk(out_path: Path, target_bytes: int, seen: set[str]):
    """arXiv offers a monthly bulk LaTeX snapshot via S3 (requester-pays).
    For a public alternative we pull a few individual paper tarballs from
    arxiv.org/e-print/* — slow, but each tar has the LaTeX source.

    For v1 we use the HF dataset `scientific_papers/arxiv` (abstracts)
    as a LaTeX-flavored fallback since full LaTeX needs requester-pays.
    """
    from datasets import load_dataset
    print(f"\n--- arXiv LaTeX-ish (target: {target_bytes/1e9:.1f} GB) ---")
    written = 0
    docs = 0
    t0 = time.time()
    try:
        ds = load_dataset("scientific_papers", "arxiv",
                          split="train", streaming=True)
    except Exception as e:
        print(f"  scientific_papers/arxiv SKIP ({e}); LaTeX coverage will be thin")
        return 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            if written >= target_bytes:
                break
            text = ex.get("article", "")
            if not text or len(text) < 500:
                continue
            if not _is_new(seen, text):
                continue
            f.write(text)
            f.write("\n\n")
            b = len(text.encode("utf-8", errors="replace")) + 2
            written += b
            docs += 1
            if docs % 1_000 == 0:
                print(f"  {docs:,} papers, {written/1e9:.2f} GB, {int(time.time()-t0)}s")
    print(f"  done: {docs:,} papers, {written/1e9:.2f} GB, {int(time.time()-t0)}s")
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/specialists/markup"))
    p.add_argument("--target-gb", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--skip-latex", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seen = _dedup_seen()
    target_per = int(args.target_gb * 1e9 / 3)
    total = 0

    # HTML
    total += stream_stack_v2(args.output_dir / "html.txt", "html",
                             target_per, seen, args.seed)
    # Markdown
    total += stream_stack_v2(args.output_dir / "markdown.txt", "markdown",
                             target_per, seen, args.seed + 1)
    # LaTeX
    if not args.skip_latex:
        total += download_arxiv_bulk(args.output_dir / "latex.txt",
                                     target_per, seen)

    print(f"\n=== Markup corpus summary ===")
    for f in sorted(args.output_dir.iterdir()):
        if f.is_file() and f.suffix == ".txt":
            print(f"  {f.name:<20s} {f.stat().st_size/1e9:>6.2f} GB")
    print(f"  {'TOTAL':<20s} {total/1e9:>6.2f} GB")


if __name__ == "__main__":
    sys.exit(main())

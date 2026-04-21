"""Phase 14 task 31: investigate markup pile specialization gap.

Training-engineer's B/T audit shows markup SPM barely beats prose/code
(2.22 vs 2.21). Goal: figure out why and recommend a fix option.

Checks:
1. Pile composition: bytes from each source file (HTML, Markdown, bleed).
2. Top-100 SPM pieces — should be structural markers, not prose words.
3. Per-source B/T vs other SPMs — is the gap concentrated in one source?
4. Duplicate-rate in the fresh HTML/Markdown (fuzzy-dedup was not run at
   sourcing time).
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
import time
from collections import Counter
from pathlib import Path


def top_spm_pieces(spm_path: Path, n: int = 100) -> list[tuple[int, str, float]]:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(spm_path))
    pieces = []
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)
        # Skip the reserved control/byte fallback pieces (<pad>, <unk>,
        # <s>, and <0x00>..<0xFF>); all have score 0 and swamp the
        # top-N sort.
        if piece.startswith("<") and piece.endswith(">"):
            continue
        pieces.append((i, piece, score))
    # Unigram SPM scores are log-probabilities: higher = more frequent.
    pieces.sort(key=lambda x: -x[2])
    return pieces[:n]


def pile_composition(markup_dir: Path) -> dict:
    files = sorted(markup_dir.glob("from_*.txt"))
    composition: dict = {"total_bytes": 0, "sources": {}}
    for f in files:
        size = f.stat().st_size
        composition["total_bytes"] += size
        # Categorize by filename
        name = f.name
        if "stackv2_html" in name:
            cat = "html"
        elif "stackv2_markdown" in name:
            cat = "markdown"
        elif "lumees_struct" in name:
            cat = "struct_bleed"
        elif "nick007x_code" in name:
            cat = "code_bleed"
        else:
            cat = "other_bleed"
        composition["sources"].setdefault(cat, {"bytes": 0, "files": 0})
        composition["sources"][cat]["bytes"] += size
        composition["sources"][cat]["files"] += 1
    return composition


def sample_bt_per_source(markup_dir: Path, all_spm_paths: dict,
                        n_samples: int, sample_size: int, seed: int) -> dict:
    """For each markup source file group, compute avg bytes-per-token
    under each SPM. Helps identify which content is weakest."""
    import sentencepiece as spm

    sps = {}
    for d, p in all_spm_paths.items():
        sps[d] = spm.SentencePieceProcessor()
        sps[d].load(str(p))

    rng = random.Random(seed)
    per_source: dict = {}
    cat_files: dict[str, list[Path]] = {}
    for f in sorted(markup_dir.glob("from_*.txt")):
        name = f.name
        if "stackv2_html" in name: cat = "html"
        elif "stackv2_markdown" in name: cat = "markdown"
        elif "lumees_struct" in name: cat = "struct_bleed"
        elif "nick007x_code" in name: cat = "code_bleed"
        else: cat = "other_bleed"
        cat_files.setdefault(cat, []).append(f)

    for cat, files in cat_files.items():
        per_source[cat] = {d: [] for d in all_spm_paths}
        total_sample_bytes = 0
        for f in files:
            sz = f.stat().st_size
            if sz < sample_size * 2:
                continue
            take = max(1, n_samples // max(1, len(files)))
            with open(f, "rb") as fh:
                for _ in range(take):
                    off = rng.randrange(0, sz - sample_size)
                    fh.seek(off)
                    fh.readline()  # skip partial
                    data = fh.read(sample_size)
                    text = data.decode("utf-8", errors="replace")
                    if len(text) < 256:
                        continue
                    total_sample_bytes += len(data)
                    for d, sp in sps.items():
                        ids = sp.encode_as_ids(text)
                        if ids:
                            per_source[cat][d].append(len(data) / len(ids))
        per_source[cat]["_total_sample_bytes"] = total_sample_bytes
    return per_source


def dedup_rate(markup_dir: Path, line_sample: int = 50000) -> dict:
    """Count duplicate lines across first N lines of HTML and Markdown."""
    stats: dict = {}
    for cat, pattern in [("html", "from_stackv2_html*.txt"),
                          ("markdown", "from_stackv2_markdown*.txt")]:
        files = list(markup_dir.glob(pattern))
        if not files:
            continue
        hashes: set[str] = set()
        total = 0
        dup = 0
        with open(files[0], "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= line_sample:
                    break
                line_stripped = line.strip()
                if len(line_stripped) < 20:
                    continue
                h = hashlib.md5(line_stripped.encode("utf-8")).hexdigest()
                if h in hashes:
                    dup += 1
                else:
                    hashes.add(h)
                total += 1
        stats[cat] = {
            "sample_lines": total,
            "unique_lines": len(hashes),
            "dup_lines": dup,
            "dup_rate": dup / total if total else 0,
        }
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path,
                   default=Path("/Users/dmatt/Claude Projects/l3tc-prod"))
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--sample-size", type=int, default=8192)
    p.add_argument("--seed", type=int, default=1204)
    args = p.parse_args()

    root = args.root
    markup_dir = root / "data/specialists/markup"

    print("=== Task 31: markup pile investigation ===\n")

    # 1. Composition
    print("--- 1. Pile composition ---")
    comp = pile_composition(markup_dir)
    total_gb = comp["total_bytes"] / 1e9
    for cat, info in sorted(comp["sources"].items(), key=lambda x: -x[1]["bytes"]):
        pct = 100 * info["bytes"] / comp["total_bytes"]
        print(f"  {cat:18s} {info['bytes']/1e9:>5.2f} GB ({pct:>5.1f}%)  [{info['files']} files]")
    print(f"  {'total':18s} {total_gb:>5.2f} GB")

    # 2. Top SPM pieces
    print("\n--- 2. Top 40 markup SPM pieces (by unigram score) ---")
    spm_path = markup_dir / "spm.model"
    if not spm_path.exists():
        print("  spm.model missing")
        return
    top = top_spm_pieces(spm_path, n=40)
    structural_markers = ("<", ">", "/>", "<!", "</", "<?", "```", "# ", "## ", "### ",
                          "**", "*", "[", "]", "(", ")", "\\")
    struct_count = 0
    prose_count = 0
    for i, piece, score in top:
        # strip sentencepiece meta prefix
        clean = piece.lstrip("▁")
        kind = "structural" if any(m in piece for m in structural_markers) else "other"
        if kind == "structural":
            struct_count += 1
        if clean.replace(" ", "").isalpha() and len(clean) > 2:
            prose_count += 1
        print(f"  [{i:>5}]  score={score:>8.3f}  {kind:10s}  {piece!r}")
    print(f"\n  structural markers in top-40: {struct_count}/40 = {struct_count/40:.0%}")
    print(f"  word-like pieces in top-40:   {prose_count}/40 = {prose_count/40:.0%}")

    # 3. Per-source B/T (critical for diagnosis)
    print("\n--- 3. Per-source bytes-per-token (BT — higher = better compressed) ---")
    all_spms = {
        d: root / f"data/specialists/{d}/spm.model"
        for d in ["prose", "code", "structured", "markup", "fallback"]
    }
    all_spms = {d: p for d, p in all_spms.items() if p.exists()}
    per_src = sample_bt_per_source(markup_dir, all_spms, args.n_samples,
                                    args.sample_size, args.seed)
    header = f"  {'source':18s}" + "".join(f"{d:>10s}" for d in all_spms)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cat, results in per_src.items():
        row = f"  {cat:18s}"
        for d in all_spms:
            vals = results[d]
            avg = sum(vals) / len(vals) if vals else 0
            row += f"{avg:>10.3f}"
        print(row)
    # Identify winner per source
    print(f"\n  winner per source (highest BT):")
    for cat, results in per_src.items():
        winners = [(d, sum(results[d]) / len(results[d])) for d in all_spms
                   if results[d]]
        winners.sort(key=lambda x: -x[1])
        if winners:
            best = winners[0]
            margin = best[1] - winners[1][1] if len(winners) > 1 else 0
            print(f"    {cat:18s} {best[0]} (BT {best[1]:.3f}, +{margin:.3f})")

    # 4. Dedup rate
    print("\n--- 4. Fresh HTML/Markdown dedup rate (line-level, first 50K lines) ---")
    dedup = dedup_rate(markup_dir)
    for cat, info in dedup.items():
        print(f"  {cat}: {info['dup_lines']:,} / {info['sample_lines']:,} dup "
              f"({info['dup_rate']:.1%})")

    print("\n--- Recommendation ---")
    # Basic heuristic
    html_pct = comp["sources"].get("html", {}).get("bytes", 0) / comp["total_bytes"]
    md_pct = comp["sources"].get("markdown", {}).get("bytes", 0) / comp["total_bytes"]
    bleed_pct = sum(comp["sources"].get(k, {}).get("bytes", 0)
                    for k in ("struct_bleed", "code_bleed", "other_bleed")) / comp["total_bytes"]
    print(f"  html={html_pct:.0%}, md={md_pct:.0%}, bleed={bleed_pct:.0%}")
    if bleed_pct > 0.25:
        print(f"  -> bleed from code/struct files is {bleed_pct:.0%} of markup pile.")
        print(f"     Those files aren't really markup; they match detect_markup only")
        print(f"     because they contain Javadoc HTML or markdown comments.")
        print(f"     Consider filtering them out (option b: rebalance + retrain SPM).")


if __name__ == "__main__":
    sys.exit(main())

"""Phase 14 task 5: audit the Phase 11 51 GB corpus by domain.

Approach:
1. Enumerate the raw-text source files that built the Phase 11
   `train_2l_corpus_balanced_32k.txt` blob.
2. For each source, sample 4 KB windows and run the Phase 14 detect()
   Python port (identical rules to l3tc-rust/src/bin/l3tc/specialist.rs).
3. Emit per-source and aggregated per-domain byte breakdown.

The Pile piece is only available as tokenized IDs in
corpus_build/retokenized/pile.part*.txt; we decode a random sample
with the balanced SPM to detect its specialist distribution.

Usage:
    python scripts/phase14_audit_corpus.py \\
        --samples-per-source 2000 \\
        --out docs/phases/PHASE_14_CORPUS_AUDIT.md
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# --- Python port of l3tc-rust/src/bin/l3tc/specialist.rs::detect() ---

PROSE, CODE, STRUCT, LOGS, TAB, MARKUP, FALLBACK = (
    "prose", "code", "structured", "logs", "tabular", "markup", "fallback"
)


def _looks_timestamped(line: str) -> bool:
    """Port of l3tc-rust looks_timestamped (specialist.rs). Covers:
    - ISO-8601 prefix (2024-01-01...)
    - Syslog prefix ("Jan 12 10:00:00...")
    - Bracketed timestamp anywhere in first 160 chars
      ([2024-01-01 ...] or Apache/nginx [02/Jun/2026:22:39:14 +0000])
    - BGL-style prefix ("- <10-digit-epoch> ...")
    """
    b = line.encode("utf-8", errors="replace")
    if len(b) >= 10 and b[:4].isdigit() and b[4:5] == b"-" and b[5:7].isdigit() and b[7:8] == b"-":
        return True
    # Spark-style short-year slash prefix: "YY/MM/DD HH:MM:SS".
    if (len(b) >= 17
            and b[:2].isdigit() and b[2:3] == b"/"
            and b[3:5].isdigit() and b[5:6] == b"/"
            and b[6:8].isdigit() and b[8:9] == b" "
            and b[9:11].isdigit() and b[11:12] == b":"):
        return True
    if len(b) >= 15 and b[:3].isalpha() and b[3:4] == b" ":
        return True
    head = line[:160]
    ob = head.find("[")
    if ob != -1:
        cb_rel = head[ob:].find("]")
        if cb_rel != -1:
            inner = head[ob + 1:ob + cb_rel]
            colons = inner.count(":")
            slashes = inner.count("/")
            has_digit = any(c.isdigit() for c in inner)
            if has_digit and (colons >= 2 or (colons >= 1 and slashes >= 1)):
                return True
    # BGL-style: "- <10-digit-epoch>" OR "<UPPER-TAG> <10-digit-epoch>"
    sp = b.find(b" ")
    if sp != -1:
        rest = b[sp + 1:]
        if len(rest) >= 10 and rest[:10].isdigit():
            prefix = b[:sp]
            if prefix == b"-" or (prefix and prefix.isupper() and prefix.isalpha()):
                return True
    return False


def _detect_yaml(text: str):
    t = text.lstrip()
    if t.startswith("apiVersion:") or t.startswith("kind:") or t.startswith("---\n") or t.startswith("--- \n"):
        return (STRUCT, 0.95)
    lines = text.splitlines()[:100]
    if len(lines) < 3:
        return None
    yaml_like = 0
    for line in lines:
        tt = line.lstrip()
        cp = tt.find(":")
        if cp <= 0:
            continue
        key = tt[:cp]
        if not key:
            continue
        c0 = key[0]
        if not (c0.isalpha() or c0 == "_"):
            continue
        if all(c.isalnum() or c in ("_", "-") for c in key):
            yaml_like += 1
    ratio = yaml_like / len(lines)
    if ratio > 0.6:
        return (STRUCT, 0.80)
    return None


def _detect_json(text: str):
    t = text.lstrip()
    if not t:
        return None
    first = t[0]
    if first not in "{[":
        return None
    opens = sum(1 for c in t if c in "{[")
    closes = sum(1 for c in t if c in "}]")
    quotes = t.count('"')
    balance = (min(opens, closes) / max(opens, closes)) if opens > 0 else 0.0
    if balance > 0.5 and quotes >= 4 and opens >= 2:
        return (STRUCT, 0.90)
    if first == "{" and quotes >= 2:
        return (STRUCT, 0.70)
    return None


def _detect_xml_or_html(text: str):
    t = text.lstrip()
    if t.startswith("<?xml"):
        return (STRUCT, 0.95)
    head = t[:50].lower()
    if head.startswith("<!doctype html") or head.startswith("<html"):
        return (MARKUP, 0.95)
    b = text.encode("utf-8", errors="replace")
    tag_count = 0
    for i in range(len(b) - 2):
        if b[i] == ord("<") and (chr(b[i + 1]).isalpha() or b[i + 1] == ord("/")):
            tag_count += 1
    if tag_count > max(10, len(text) // 100):
        low = text.lower()
        html_tags = ["<div", "<span", "<p>", "<a ", "<img", "<head", "<body", "<script"]
        html_hits = sum(1 for h in html_tags if h in low)
        if html_hits >= 2:
            return (MARKUP, 0.85)
        return (STRUCT, 0.75)
    return None


def _detect_logs(text: str):
    lines = text.splitlines()[:40]
    if len(lines) < 5:
        return None
    # Tightened after task 6 contamination: a line with only a log-level
    # keyword (e.g. "ERROR" as a Python constant) is not a log line.
    # Require timestamp AND (level OR IP) as the strong signal.
    levels = ["INFO", "ERROR", "WARN", "DEBUG", "TRACE", "FATAL", "[INFO]", "[ERROR]"]
    leveled = sum(1 for line in lines if any(lv in line for lv in levels))
    ts = sum(1 for line in lines if _looks_timestamped(line))
    ip_like = sum(1 for line in lines if _has_ip(line))
    total = len(lines)
    lr = leveled / total
    tr = ts / total
    ipr = ip_like / total
    # Strong: majority timestamped AND (some level keyword OR IP pattern).
    if tr > 0.5 and (lr > 0.2 or ipr > 0.2):
        return (LOGS, 0.85)
    # Medium: mixed timestamp + level signal.
    if tr > 0.3 and lr > 0.3:
        return (LOGS, 0.65)
    return None


def _has_ip(line: str) -> bool:
    # Loose IPv4 check: 4 dot-separated 1-3 digit groups in the line.
    parts = 0
    run = 0
    for c in line:
        if c.isdigit():
            run += 1
            if run > 3:
                run = 0
                parts = 0
        elif c == ".":
            if 1 <= run <= 3:
                parts += 1
                run = 0
            else:
                parts = 0
                run = 0
        else:
            if parts == 3 and 1 <= run <= 3:
                return True
            parts = 0
            run = 0
    return parts == 3 and 1 <= run <= 3


def _detect_tabular(text: str):
    lines = [l for l in text.splitlines()[:20] if l]
    if len(lines) < 5:
        return None
    for delim in (",", "\t", ";"):
        counts = [l.count(delim) for l in lines]
        mx = max(counts) if counts else 0
        if mx < 2:
            continue
        mode = sum(1 for c in counts if c == mx)
        if mode / len(lines) > 0.7:
            return (TAB, 0.85)
    return None


def _detect_markdown(text: str):
    lines = text.splitlines()[:60]
    if len(lines) < 3:
        return None
    heading = sum(1 for l in lines if l.startswith("#"))
    fence = sum(1 for l in lines if l.startswith("```"))
    bullet = sum(1 for l in lines if l.startswith(("- ", "* ", "+ ")))
    link_like = text.count("](http") + text.count("](#")
    signals = (heading >= 2) + (fence >= 2) + (bullet >= 3) + (link_like >= 2)
    if signals >= 2:
        return (MARKUP, 0.85)
    if signals >= 1 and (heading + bullet + fence) > 3:
        return (MARKUP, 0.70)
    return None


def _detect_code(text: str):
    keywords = [
        " def ", " class ", " function ", " return ", " import ", "#include",
        " public ", " private ", " static ", " void ", "fn ", " let ", " const ",
        " var ", " async ", " await ", " => ", "::", " struct ", " enum ",
        " impl ", " trait ", " interface ",
    ]
    hits = sum(1 for kw in keywords if kw in text)
    braces = text.count("{") + text.count("}")
    semis = text.count(";")
    punct_ratio = (braces + semis) / max(len(text), 1)
    if hits >= 4 or (hits >= 2 and punct_ratio > 0.02):
        return (CODE, 0.80)
    if hits >= 2:
        return (CODE, 0.60)
    return None


def _detect_prose(text: str):
    letters = 0
    total = 0
    for c in text:
        cc = ord(c) if len(c) == 1 else 0
        if cc < 128:
            total += 1
            if c.isalpha():
                letters += 1
    if total < 100:
        return None
    lr = letters / total
    words = [w for w in text.split() if w]
    punct = text.count(".") + text.count(",")
    pr = punct / max(len(words), 1)
    if lr > 0.72 and pr > 0.03:
        return (PROSE, 0.80)
    if lr > 0.65:
        return (PROSE, 0.55)
    return None


def detect(sample_bytes: bytes):
    n = min(len(sample_bytes), 4096)
    buf = sample_bytes[:n]
    if len(buf) < 256:
        return (FALLBACK, 0.50)
    text = buf.decode("utf-8", errors="replace")
    for fn in (_detect_yaml, _detect_json, _detect_xml_or_html, _detect_logs,
               _detect_tabular, _detect_markdown, _detect_code, _detect_prose):
        d = fn(text)
        if d is not None:
            return d
    return (FALLBACK, 0.40)


# --- Sampling ---

def sample_file(path: Path, samples: int, seed: int, window: int = 4096):
    """Return detect() label counts over `samples` random 4 KB windows."""
    rng = random.Random(seed)
    size = path.stat().st_size
    if size < window:
        return {}, 0
    counts: dict[str, int] = {}
    with open(path, "rb") as f:
        for _ in range(samples):
            off = rng.randrange(0, size - window)
            f.seek(off)
            buf = f.read(window)
            label, _ = detect(buf)
            counts[label] = counts.get(label, 0) + 1
    return counts, size


def sample_pile_decoded(part_files: list[Path], spm_path: Path, samples: int,
                        seed: int, window_tokens: int = 1024):
    """Decode random spans of tokenized pile to text, run detect().

    Uses byte-offset seeks (not line enumeration) to avoid a 10-minute
    pass over every shard. Each sample: pick a random byte offset in
    a random shard, skip the partial line, read `window_tokens` lines,
    decode via SPM.
    """
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(spm_path))
    rng = random.Random(seed)
    counts: dict[str, int] = {}
    total_bytes = sum(p.stat().st_size for p in part_files)

    # Keep handles open to reuse seek
    handles = {p: open(p, "rb") for p in part_files}
    sizes = {p: p.stat().st_size for p in part_files}

    try:
        for _ in range(samples):
            part = rng.choice(part_files)
            f = handles[part]
            sz = sizes[part]
            # Leave ~64 KB tail buffer so we can read window_tokens lines
            off = rng.randrange(0, max(1, sz - 65536))
            f.seek(off)
            f.readline()  # skip partial
            ids: list[int] = []
            for _ in range(window_tokens):
                line = f.readline()
                if not line:
                    break
                try:
                    ids.append(int(line.strip()))
                except ValueError:
                    continue
            if not ids:
                continue
            try:
                text = sp.decode_ids(ids)
            except Exception:
                continue
            bb = text.encode("utf-8", errors="replace")[:4096]
            label, _ = detect(bb)
            counts[label] = counts.get(label, 0) + 1
    finally:
        for f in handles.values():
            f.close()
    return counts, total_bytes


# --- Main audit ---

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path,
                   default=Path("/Users/dmatt/Claude Projects/l3tc-prod"))
    p.add_argument("--samples-per-source", type=int, default=2000)
    p.add_argument("--out", type=Path,
                   default=Path("docs/phases/PHASE_14_CORPUS_AUDIT.md"))
    p.add_argument("--pile-samples", type=int, default=2000,
                   help="Total samples across all pile shards.")
    p.add_argument("--seed", type=int, default=1204)
    args = p.parse_args()

    root = args.root
    corpus_build = root / "corpus_build"
    structured = corpus_build / "structured"
    retokenized = corpus_build / "retokenized"

    # Sources that built the 51 GB balanced corpus. Sizes are the actual
    # raw text bytes that went into tokenization.
    sources = [
        # (label, path, approx-known-domain-hint)
        ("pile_raw_sample_1gb", corpus_build / "pile_raw_1gb.txt", "prose/mixed"),
        ("nick007x_diverse_code", structured / "code_diverse_real.txt", "code"),
        ("lumees_structured_1", structured / "code_real.txt", "structured"),
        ("lumees_structured_2", structured / "code_real_extra.txt", "structured"),
        ("zenodo_loghub", structured / "logs_real.txt", "logs"),
        ("public_csv", structured / "csv_real.txt", "tabular"),
    ]

    per_source: list[dict] = []
    print(f"Auditing {len(sources)} raw-text sources with {args.samples_per_source} samples each...")
    for label, path, hint in sources:
        if not path.exists():
            print(f"  SKIP {label} ({path} missing)")
            continue
        t0 = time.time()
        counts, size = sample_file(path, args.samples_per_source, args.seed)
        total = sum(counts.values())
        fractions = {k: v / total for k, v in counts.items()} if total else {}
        per_source.append({
            "label": label,
            "path": str(path.relative_to(root)),
            "hint": hint,
            "bytes": size,
            "sample_counts": counts,
            "fractions": fractions,
            "elapsed_s": time.time() - t0,
        })
        print(f"  {label}: {size/1e9:.2f} GB, {counts}, {time.time()-t0:.1f}s")

    # Pile decoded: the real 40 GB Pile is only available as tokens.
    pile_parts = sorted(retokenized.glob("pile.part*.txt"))
    balanced_spm = root / "tokenizer_balanced_32k/spm_balanced_unigram_32768.model"
    pile_result = None
    if pile_parts and balanced_spm.exists():
        try:
            import sentencepiece as spm  # noqa
            print(f"\nDecoding {len(pile_parts)} pile shards with {balanced_spm.name}...")
            t0 = time.time()
            counts, tok_size = sample_pile_decoded(
                pile_parts, balanced_spm, args.pile_samples, args.seed)
            total = sum(counts.values())
            fractions = {k: v / total for k, v in counts.items()} if total else {}
            pile_result = {
                "label": "pile_40gb_decoded",
                "path": str((retokenized / "pile.part*.txt").relative_to(root)),
                "hint": "The Pile dedup, decoded from balanced SPM",
                "tokenized_bytes": tok_size,
                "approx_raw_bytes": 40_000_000_000,  # from build logs
                "sample_counts": counts,
                "fractions": fractions,
                "elapsed_s": time.time() - t0,
            }
            print(f"  pile: {counts}, {time.time()-t0:.1f}s")
        except ImportError:
            print("  sentencepiece not available; skipping pile decode sample")

    # --- Aggregate per-domain byte estimate ---
    agg_bytes: dict[str, float] = {
        PROSE: 0, CODE: 0, STRUCT: 0, LOGS: 0, TAB: 0, MARKUP: 0, FALLBACK: 0,
    }
    for s in per_source:
        if s["label"] == "pile_raw_sample_1gb":
            # Scale to full 40 GB Pile
            scale = 40_000_000_000 / max(s["bytes"], 1)
            bytes_to_assign = s["bytes"] * scale
        else:
            bytes_to_assign = s["bytes"]
        for dom, frac in s["fractions"].items():
            agg_bytes[dom] += bytes_to_assign * frac
    # Use decoded pile if available in place of the 1 GB sample-scaled value.
    if pile_result is not None:
        # Subtract the 1 GB sample-scaled contribution first
        pile_sample = next((s for s in per_source if s["label"] == "pile_raw_sample_1gb"), None)
        if pile_sample:
            scale = 40_000_000_000 / max(pile_sample["bytes"], 1)
            for dom, frac in pile_sample["fractions"].items():
                agg_bytes[dom] -= pile_sample["bytes"] * scale * frac
        for dom, frac in pile_result["fractions"].items():
            agg_bytes[dom] += pile_result["approx_raw_bytes"] * frac

    total_bytes = sum(agg_bytes.values())

    # --- Emit Markdown report ---
    out_path = (root / args.out) if not args.out.is_absolute() else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    now = time.strftime("%Y-%m-%d")
    lines = []
    lines.append(f"# Phase 14 — Corpus Audit (task 5)\n")
    lines.append(f"_Generated {now} by `scripts/phase14_audit_corpus.py`._\n")
    lines.append("## Method\n")
    lines.append(
        "Detect() heuristic (exact Python port of "
        "`l3tc-rust/src/bin/l3tc/specialist.rs::detect`) applied to "
        f"random 4 KB windows of each raw-text source that built the Phase 11 "
        f"`train_2l_corpus_balanced_32k.txt` blob. "
        f"{args.samples_per_source} samples per source, seed "
        f"{args.seed}. Pile is tokenized-only on disk, so we decode "
        f"{args.pile_samples} random 1024-token spans via the balanced unigram SPM first.\n"
    )

    lines.append("## Per-source sample breakdown\n")
    lines.append("| source | known-hint | raw bytes | prose | code | structured | logs | tabular | markup | fallback |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    all_rows = list(per_source)
    if pile_result is not None:
        all_rows.append(pile_result)
    for r in all_rows:
        f = r.get("fractions", {})

        def fpct(k):
            v = f.get(k)
            return f"{v*100:.0f}%" if v else "—"

        size = r.get("bytes") or r.get("approx_raw_bytes", 0)
        lines.append(
            f"| `{r['label']}` | {r['hint']} | {size/1e9:.2f} GB | "
            f"{fpct(PROSE)} | {fpct(CODE)} | {fpct(STRUCT)} | "
            f"{fpct(LOGS)} | {fpct(TAB)} | {fpct(MARKUP)} | {fpct(FALLBACK)} |"
        )

    lines.append("\n## Aggregate — extrapolated byte share across the 51 GB corpus\n")
    lines.append("Per-source fractions × each source's raw-text size, summed.\n")
    lines.append("| domain | bytes | share |")
    lines.append("|---|---:|---:|")
    order = [PROSE, CODE, STRUCT, LOGS, TAB, MARKUP, FALLBACK]
    for dom in order:
        b = agg_bytes[dom]
        share = b / total_bytes if total_bytes else 0
        lines.append(f"| {dom} | {b/1e9:.2f} GB | {share*100:.1f}% |")
    lines.append(f"| **total** | **{total_bytes/1e9:.2f} GB** | 100.0% |")

    lines.append("\n## Gaps and recommendations\n")
    # Automatic interpretation of gaps
    targets_gb = {PROSE: 15, CODE: 15, STRUCT: 10, LOGS: 10, TAB: 10, MARKUP: 10}
    for dom, target in targets_gb.items():
        have = agg_bytes[dom] / 1e9
        delta = have - target
        status = "sufficient" if have >= target * 0.5 else "THIN" if have >= target * 0.2 else "MISSING"
        lines.append(f"- **{dom}**: have {have:.1f} GB, phase-14 target ~{target} GB — **{status}**")

    lines.append(
        "\n## Interpretation\n"
        "The detect() heuristic is what the runtime uses to route inputs, so "
        "the same rules must govern the split. Any files that detect() would "
        "label as a domain at compress-time should be in that specialist's "
        "training pile at training-time. Edge cases where detect() fires "
        "differently on short-vs-long samples will show up in both phases; that "
        "is intentional.\n"
    )
    lines.append(
        "The Pile slice is the dominant prose source but bleeds into other "
        "domains (inline code, tables, markdown snippets inside articles). "
        "Detect() picks up those bleeds as their specialist domains; that is "
        "OK — those subspans are genuinely better handled by the matching "
        "specialist at runtime.\n"
    )

    lines.append("\n## Raw counts (for task 6 split decisions)\n")
    lines.append("```json")
    lines.append(json.dumps({
        "per_source": all_rows,
        "aggregate_bytes": agg_bytes,
        "total_bytes": total_bytes,
    }, indent=2))
    lines.append("```")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote audit to {out_path}")


if __name__ == "__main__":
    sys.exit(main())

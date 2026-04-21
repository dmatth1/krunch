"""Phase 14 task 29: clean the logs pile of prose/code contamination.

Re-run detect() (now with the tightened log heuristic: requires timestamp
AND level/IP pattern) over every document in `data/specialists/logs/` and
reassign each doc to its true domain. Documents that still detect as logs
stay put; everything else moves to the correct pile under
`data/specialists/<target>/cleaned_from_logs.part*.txt`.

Also optionally runs a sanity audit on the other 6 piles (100-sample
random probe each, just to confirm no symmetric contamination).

Usage:
    python scripts/phase14_clean_logs_pile.py --dry-run     # report only
    python scripts/phase14_clean_logs_pile.py               # move & rewrite
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from phase14_audit_corpus import detect, PROSE, CODE, STRUCT, LOGS, TAB, MARKUP, FALLBACK  # noqa: E402


DOMAINS = [PROSE, CODE, STRUCT, LOGS, TAB, MARKUP, FALLBACK]


def iter_docs(path: Path):
    """Yield documents separated by blank lines (same format the splitter wrote)."""
    buf: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip() == "":
                if buf:
                    yield "".join(buf)
                    buf = []
            else:
                buf.append(line)
        if buf:
            yield "".join(buf)


def clean_logs_pile(root: Path, dry_run: bool) -> dict:
    logs_dir = root / "data/specialists/logs"
    files = sorted(logs_dir.glob("from_*.txt"))
    if not files:
        print(f"  no logs input files in {logs_dir}")
        return {}

    # New destinations: one output file per (source_file, target_domain)
    # so we don't cross-contaminate when moving.
    new_counts: dict[str, dict[str, int]] = {d: {"docs": 0, "bytes": 0} for d in DOMAINS}

    # Track per-source breakdown
    per_source: dict[str, dict[str, int]] = {}

    t0 = time.time()
    total_docs = 0

    # Collector maps "target_domain -> file_handle" per input file
    for in_file in files:
        source_stem = in_file.stem  # from_zenodo_loghub.part00 etc
        per_source[source_stem] = {d: 0 for d in DOMAINS}

        if dry_run:
            handles: dict[str, object] = {}
        else:
            handles = {}

        # Write cleaned logs back to a temp file; move on success.
        cleaned_logs_tmp = in_file.with_suffix(".cleaned.tmp") if not dry_run else None
        logs_handle = None
        if not dry_run:
            logs_handle = open(cleaned_logs_tmp, "w", encoding="utf-8")

        try:
            for doc in iter_docs(in_file):
                if not doc:
                    continue
                sample = doc.encode("utf-8", errors="replace")[:4096]
                label, _conf = detect(sample)
                dom = label if label in DOMAINS else FALLBACK
                size = len(doc.encode("utf-8", errors="replace"))

                new_counts[dom]["docs"] += 1
                new_counts[dom]["bytes"] += size
                per_source[source_stem][dom] += 1
                total_docs += 1

                if dry_run:
                    continue

                if dom == LOGS:
                    logs_handle.write(doc)
                    if not doc.endswith("\n"):
                        logs_handle.write("\n")
                    logs_handle.write("\n")
                else:
                    out_path = root / f"data/specialists/{dom}/cleaned_from_logs.{source_stem}.txt"
                    if dom not in handles:
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        handles[dom] = open(out_path, "a", encoding="utf-8")
                    fh = handles[dom]
                    fh.write(doc)
                    if not doc.endswith("\n"):
                        fh.write("\n")
                    fh.write("\n")
        finally:
            if logs_handle is not None:
                logs_handle.close()
            for fh in handles.values():
                fh.close()

        # Atomically replace the logs source with the cleaned version.
        if not dry_run and cleaned_logs_tmp is not None:
            cleaned_logs_tmp.replace(in_file)

    elapsed = time.time() - t0
    print(f"  processed {total_docs:,} logs-pile docs in {elapsed:.0f}s")
    for d in DOMAINS:
        n = new_counts[d]["docs"]
        b = new_counts[d]["bytes"]
        if n:
            pct = 100 * n / total_docs
            print(f"    {d:10s} {n:>8,} docs  {b/1e6:>8.1f} MB  ({pct:.1f}%)")

    return {
        "total_docs": total_docs,
        "by_target_domain": new_counts,
        "per_source_file": per_source,
        "elapsed_s": elapsed,
    }


def sanity_other_piles(root: Path, samples: int = 100) -> dict:
    """Spot-check 100 docs per pile to confirm no symmetric contamination."""
    import random
    rng = random.Random(2026)
    report: dict = {}
    for d in [PROSE, CODE, STRUCT, TAB, MARKUP, FALLBACK]:
        pile = root / "data/specialists" / d
        files = sorted(pile.glob("from_*.txt"))
        if not files:
            continue
        counts = {x: 0 for x in DOMAINS}
        probed = 0
        for f in files:
            # Sample per file proportional to samples/len(files)
            per_file = max(1, samples // len(files))
            docs = list(iter_docs(f))
            if not docs:
                continue
            rng.shuffle(docs)
            for doc in docs[:per_file]:
                sample = doc.encode("utf-8", errors="replace")[:4096]
                label, _ = detect(sample)
                counts[label] = counts.get(label, 0) + 1
                probed += 1
        report[d] = {"probed": probed, "by_label": counts}
        correct = counts.get(d, 0)
        pct = 100 * correct / probed if probed else 0
        print(f"  {d:10s} {probed} probed, {pct:.0f}% match own pile: {counts}")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path,
                   default=Path("/Users/dmatt/Claude Projects/l3tc-prod"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-sanity", action="store_true")
    args = p.parse_args()

    print("=== Task 29: clean logs pile ===")
    print("(tightened _detect_logs: timestamp AND (level OR IP))")
    print(f"dry_run={args.dry_run}\n")

    logs_report = clean_logs_pile(args.root, args.dry_run)

    if not args.skip_sanity:
        print(f"\n=== Sanity probe on other 6 piles ===")
        sanity = sanity_other_piles(args.root)
        logs_report["sanity_probe"] = sanity

    out = args.root / "data/specialists/_task29_clean_report.json"
    out.write_text(json.dumps(logs_report, indent=2))
    print(f"\nreport: {out}")


if __name__ == "__main__":
    sys.exit(main())

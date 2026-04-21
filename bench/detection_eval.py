#!/usr/bin/env python3
"""Detection-accuracy evaluation harness (Task 15).

Runs the `l3tc detect` Rust subcommand over every file in
`bench/detection_corpus/`, compares to `labels.tsv`, and emits a
confusion matrix plus per-class precision/recall and a list of
misclassifications.

Success criteria (PHASE_14):
  - Common (clean) domains: ≥95% accuracy
  - Edge cases: ≥80% accuracy

Outputs:
  - bench/detection_eval.md : human-readable confusion matrix + misses
  - bench/detection_eval.json : machine-readable per-file predictions

Usage:
  python3 bench/detection_eval.py                # uses release l3tc
  python3 bench/detection_eval.py --bin /path/to/l3tc  # override
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "bench" / "detection_corpus"
LABELS_TSV = CORPUS_DIR / "labels.tsv"
DEFAULT_BIN = REPO_ROOT / "l3tc-rust" / "target" / "release" / "l3tc"

SPECIALISTS = ["prose", "code", "structured", "logs", "tabular", "markup", "fallback"]


@dataclass
class Prediction:
    filename: str
    expected: str
    predicted: str
    confidence: float
    category: str  # "clean" | "edge"
    notes: str


def load_labels(path: Path) -> list[tuple[str, str, str, str]]:
    rows = []
    with path.open() as f:
        header = f.readline().strip().split("\t")
        assert header == ["filename", "expected_specialist", "category", "notes"], header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            rows.append(tuple(parts))
    return rows


def run_detect(bin_path: Path, file_path: Path) -> tuple[str, float]:
    """Invoke `l3tc detect --json <file>` and return (specialist, confidence)."""
    proc = subprocess.run(
        [str(bin_path), "detect", "--json", str(file_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"detect failed on {file_path}: rc={proc.returncode} stderr={proc.stderr!r}")
    rec = json.loads(proc.stdout.strip())
    return rec["specialist"], float(rec["confidence"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path, default=DEFAULT_BIN, help="Path to l3tc binary")
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR, help="Path to detection corpus")
    parser.add_argument("--labels", type=Path, default=LABELS_TSV, help="Path to labels.tsv")
    parser.add_argument(
        "--limit", type=int, default=0, help="Only eval the first N files (0 = all)"
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=REPO_ROOT / "bench" / "detection_eval.md",
        help="Output markdown report",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "bench" / "detection_eval.json",
        help="Output JSON per-file predictions",
    )
    args = parser.parse_args(argv)

    if not args.bin.exists():
        print(f"error: l3tc binary not found at {args.bin}. Build with `cargo build --release --bin l3tc`.", file=sys.stderr)
        return 2
    if not args.corpus.exists():
        print(f"error: corpus not found at {args.corpus}. Run build_detection_corpus.py first.", file=sys.stderr)
        return 2
    if not args.labels.exists():
        print(f"error: labels.tsv not found at {args.labels}.", file=sys.stderr)
        return 2

    labels = load_labels(args.labels)
    if args.limit > 0:
        labels = labels[: args.limit]
    total = len(labels)
    print(f"evaluating {total} files with {args.bin}...", file=sys.stderr)

    preds: list[Prediction] = []
    for i, (fname, expected, category, notes) in enumerate(labels):
        fpath = args.corpus / fname
        try:
            predicted, conf = run_detect(args.bin, fpath)
        except Exception as e:
            print(f"  skip {fname}: {e}", file=sys.stderr)
            continue
        preds.append(Prediction(fname, expected, predicted, conf, category, notes))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{total}...", file=sys.stderr)

    # Build confusion matrix
    cm: dict[str, dict[str, int]] = {s: {t: 0 for t in SPECIALISTS} for s in SPECIALISTS}
    for p in preds:
        if p.expected in cm and p.predicted in cm[p.expected]:
            cm[p.expected][p.predicted] += 1

    # Per-class accuracy / precision / recall
    class_stats: dict[str, dict[str, float]] = {}
    for s in SPECIALISTS:
        tp = cm[s][s]
        fn = sum(cm[s].values()) - tp
        fp = sum(cm[other][s] for other in SPECIALISTS) - tp
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        accuracy = recall  # same thing for per-class when we talk about class-conditional accuracy
        class_stats[s] = {
            "tp": tp, "fn": fn, "fp": fp, "support": support,
            "precision": precision, "recall": recall, "accuracy": accuracy,
        }

    # Overall accuracy
    n_correct = sum(1 for p in preds if p.expected == p.predicted)
    overall_acc = n_correct / len(preds) if preds else 0.0

    # Per-category accuracy
    clean_total = sum(1 for p in preds if p.category == "clean")
    clean_correct = sum(1 for p in preds if p.category == "clean" and p.expected == p.predicted)
    edge_total = sum(1 for p in preds if p.category == "edge")
    edge_correct = sum(1 for p in preds if p.category == "edge" and p.expected == p.predicted)
    clean_acc = clean_correct / clean_total if clean_total else 0.0
    edge_acc = edge_correct / edge_total if edge_total else 0.0

    # Per-class clean-only accuracy (for the ≥95% gate)
    clean_by_class: dict[str, tuple[int, int]] = {s: (0, 0) for s in SPECIALISTS}
    for p in preds:
        if p.category == "clean":
            c, t = clean_by_class[p.expected]
            clean_by_class[p.expected] = (c + (1 if p.predicted == p.expected else 0), t + 1)

    # Misclassifications
    misses = [p for p in preds if p.expected != p.predicted]

    # Write JSON
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(
            {
                "total": len(preds),
                "overall_accuracy": overall_acc,
                "clean_accuracy": clean_acc,
                "edge_accuracy": edge_acc,
                "confusion_matrix": cm,
                "class_stats": class_stats,
                "clean_per_class": {s: {"correct": c, "total": t, "accuracy": (c / t if t else 0.0)} for s, (c, t) in clean_by_class.items()},
                "predictions": [
                    {"file": p.filename, "expected": p.expected, "predicted": p.predicted,
                     "confidence": p.confidence, "category": p.category, "notes": p.notes}
                    for p in preds
                ],
            },
            f,
            indent=2,
        )

    # Write markdown
    lines = []
    lines.append("# Detection accuracy eval\n")
    lines.append(f"- files evaluated: **{len(preds)}**")
    lines.append(f"- overall accuracy: **{overall_acc:.3f}**  ({n_correct}/{len(preds)})")
    lines.append(f"- clean accuracy: **{clean_acc:.3f}**  ({clean_correct}/{clean_total})  — PHASE_14 target: ≥0.95")
    lines.append(f"- edge accuracy: **{edge_acc:.3f}**  ({edge_correct}/{edge_total})  — PHASE_14 target: ≥0.80")
    gate_common = clean_acc >= 0.95
    gate_edge = edge_acc >= 0.80
    lines.append("")
    if gate_common and gate_edge:
        lines.append("**RELEASE GATE: PASS** — both accuracy thresholds met.")
    else:
        failed = []
        if not gate_common: failed.append(f"clean {clean_acc:.3f} < 0.95")
        if not gate_edge: failed.append(f"edge {edge_acc:.3f} < 0.80")
        lines.append(f"**RELEASE GATE: FAIL** — {'; '.join(failed)}")
    lines.append("")

    lines.append("## Confusion matrix (rows=expected, cols=predicted)\n")
    lines.append("| expected \\ predicted | " + " | ".join(SPECIALISTS) + " | support |")
    lines.append("|" + "---|" * (len(SPECIALISTS) + 2))
    for s in SPECIALISTS:
        row_cells = [str(cm[s][t]) for t in SPECIALISTS]
        support = sum(cm[s].values())
        lines.append(f"| **{s}** | " + " | ".join(row_cells) + f" | {support} |")
    lines.append("")

    lines.append("## Per-class metrics\n")
    lines.append("| class | support | precision | recall | accuracy |")
    lines.append("|---|---:|---:|---:|---:|")
    for s in SPECIALISTS:
        c = class_stats[s]
        lines.append(
            f"| {s} | {c['support']} | {c['precision']:.3f} | {c['recall']:.3f} | {c['accuracy']:.3f} |"
        )
    lines.append("")

    lines.append("## Per-class clean-only accuracy (PHASE_14 common-domain gate ≥0.95)\n")
    lines.append("| class | correct | total | accuracy | gate |")
    lines.append("|---|---:|---:|---:|---:|")
    for s in SPECIALISTS:
        c, t = clean_by_class[s]
        acc = c / t if t else 0.0
        gate = "PASS" if acc >= 0.95 else "FAIL"
        lines.append(f"| {s} | {c} | {t} | {acc:.3f} | {gate} |")
    lines.append("")

    # Misclassifications summary
    miss_by_kind: dict[tuple[str, str], list[Prediction]] = defaultdict(list)
    for p in misses:
        miss_by_kind[(p.expected, p.predicted)].append(p)

    lines.append(f"## Misclassifications ({len(misses)} total)\n")
    if not misses:
        lines.append("None.")
    else:
        lines.append("### By (expected → predicted) pair\n")
        lines.append("| expected | predicted | count | category mix | example file | notes |")
        lines.append("|---|---|---:|---|---|---|")
        for (exp, pred), ps in sorted(miss_by_kind.items(), key=lambda kv: -len(kv[1])):
            cats = {c: sum(1 for p in ps if p.category == c) for c in ("clean", "edge")}
            cat_str = ", ".join(f"{c}={n}" for c, n in cats.items() if n)
            example = ps[0]
            lines.append(
                f"| {exp} | {pred} | {len(ps)} | {cat_str} | `{example.filename}` | {example.notes} |"
            )
        lines.append("")
        lines.append("### Individual misclassifications (first 60)\n")
        lines.append("| file | expected | predicted | conf | category | notes |")
        lines.append("|---|---|---|---:|---|---|")
        for p in misses[:60]:
            lines.append(
                f"| `{p.filename}` | {p.expected} | {p.predicted} | {p.confidence:.2f} | {p.category} | {p.notes} |"
            )
        if len(misses) > 60:
            lines.append(f"\n...and {len(misses) - 60} more (see `detection_eval.json`).")
    lines.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines))

    # Stdout summary
    print(f"\nresults: {args.out_md}")
    print(f"details: {args.out_json}")
    print(f"overall accuracy: {overall_acc:.3f}")
    print(f"clean accuracy:   {clean_acc:.3f}  (gate ≥0.95 {'PASS' if gate_common else 'FAIL'})")
    print(f"edge accuracy:    {edge_acc:.3f}  (gate ≥0.80 {'PASS' if gate_edge else 'FAIL'})")

    # Exit 0 on PASS, 1 on FAIL so CI can gate.
    return 0 if (gate_common and gate_edge) else 1


if __name__ == "__main__":
    sys.exit(main())

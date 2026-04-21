#!/usr/bin/env python3
"""Per-specialist ratio + speed benchmark (Task 16).

BLOCKED on training-engineer task 13 until converted .bin files exist
under `l3tc-rust/models/{specialist}/model.bin` + `tokenizer.model`.
This script is ready to run the moment those artifacts land.

What this measures (per specialist, in-domain + out-of-domain):
  - Uncompressed bytes
  - Compressed bytes (ratio)
  - Compression wall time + throughput (KB/s)
  - Decompression wall time + throughput
  - Round-trip byte-identical

PHASE_14 release gate:
  - Per-specialist in-domain ratio should match Phase 11 specialist
    targets (prose ≤0.17 on enwik6, code ≤0.20 on python_source, etc.)
  - Per-specialist speed should match L3TC-200K (≥120 KB/s on M-series
    multi-thread).

Cross-domain check: each specialist run on every other domain's
corpus. A valid specialization must be NOTICEABLY worse on out-of-
domain content than the right specialist would be. Otherwise the
specialty didn't actually specialize.

Usage:
  # Run full benchmark after training hands off all 7 .bins
  python3 bench/specialist_benchmarks.py

  # Smoke-test one specialist
  python3 bench/specialist_benchmarks.py --only prose

  # Use a different benchmark corpus
  python3 bench/specialist_benchmarks.py --corpus bench/v010_benchmark
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
L3TC_BIN = REPO_ROOT / "l3tc-rust" / "target" / "release" / "l3tc"
MODELS_DIR = REPO_ROOT / "l3tc-rust" / "models"
V010_BENCHMARK = REPO_ROOT / "bench" / "v010_benchmark"
WORK_DIR = REPO_ROOT / "bench" / "work_specialists"

SPECIALISTS = ["prose", "code", "structured", "logs", "tabular", "markup", "fallback"]


def load_manifest_in_domain(corpus: Path) -> dict[str, list[Path]]:
    """Read `manifest.tsv` from the benchmark corpus and group file
    paths by domain. This is the single source of truth for "what
    files does specialist X get tested on?" — kept in
    `bench/v010_benchmark/manifest.tsv`, produced by
    `bench/build_v010_benchmark.py`.
    """
    manifest = corpus / "manifest.tsv"
    if not manifest.exists():
        return {}
    by_domain: dict[str, list[Path]] = {}
    with manifest.open() as f:
        header = f.readline().strip().split("\t")
        assert header[:3] == ["filename", "domain", "bytes"], header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rel, domain = parts[0], parts[1]
            by_domain.setdefault(domain, []).append(corpus / rel)
    return by_domain

# Phase 14 per-domain ratio targets from docs/phases/PHASE_14.md.
# Keep in sync with that doc. Used for the release-gate report.
RATIO_TARGETS: dict[str, float] = {
    "prose": 0.17,
    "code": 0.20,
    "structured": 0.15,
    "logs": 0.10,
    "tabular": 0.10,
    "markup": 0.20,
    "fallback": 0.30,
}

# Speed target — PHASE_14 requires ≥150 KB/s on Mac CPU (parallelized).
SPEED_TARGET_KBS = 150.0


@dataclass
class Measurement:
    specialist: str
    file: str
    file_label: str  # "in-domain" | "cross-domain-<specialist>" | "generalist"
    input_bytes: int
    compressed_bytes: int
    ratio: float
    compress_wall_s: float
    decompress_wall_s: float
    compress_kb_per_s: float
    decompress_kb_per_s: float
    roundtrip_ok: bool
    error: str | None = None


def specialist_available(specialist: str) -> tuple[bool, str]:
    """Check whether a specialist's model.bin + tokenizer.model exist."""
    spec_dir = MODELS_DIR / specialist
    model = spec_dir / "model.bin"
    tok = spec_dir / "tokenizer.model"
    if not model.exists():
        return False, f"missing {model}"
    if not tok.exists():
        return False, f"missing {tok}"
    return True, ""


def measure_one(
    specialist: str,
    input_path: Path,
    file_label: str,
    work_dir: Path,
) -> Measurement:
    """Compress + decompress one file with one specialist; return metrics."""
    spec_dir = MODELS_DIR / specialist
    model = spec_dir / "model.bin"
    tok = spec_dir / "tokenizer.model"
    work_dir.mkdir(parents=True, exist_ok=True)
    compressed = work_dir / f"{input_path.name}.l3tc"
    decompressed = work_dir / f"{input_path.name}.rt"

    for p in (compressed, decompressed):
        if p.exists():
            p.unlink()

    input_bytes = input_path.stat().st_size

    # Compress with explicit specialist override (no auto-detection —
    # we're measuring the specialist itself, not routing accuracy).
    t0 = time.monotonic()
    c_proc = subprocess.run(
        [
            str(L3TC_BIN), "compress", str(input_path),
            "-o", str(compressed),
            "--model", str(model),
            "--tokenizer", str(tok),
            "--specialist", specialist,
        ],
        capture_output=True, text=True, check=False,
    )
    t1 = time.monotonic()
    if c_proc.returncode != 0 or not compressed.exists():
        return Measurement(
            specialist, input_path.name, file_label, input_bytes, 0, 0.0,
            0.0, 0.0, 0.0, 0.0, False,
            error=f"compress failed: {c_proc.stderr[:300]}",
        )

    # Decompress
    t2 = time.monotonic()
    d_proc = subprocess.run(
        [
            str(L3TC_BIN), "decompress", str(compressed),
            "-o", str(decompressed),
            "--model", str(model),
            "--tokenizer", str(tok),
        ],
        capture_output=True, text=True, check=False,
    )
    t3 = time.monotonic()
    if d_proc.returncode != 0 or not decompressed.exists():
        return Measurement(
            specialist, input_path.name, file_label, input_bytes,
            compressed.stat().st_size, compressed.stat().st_size / input_bytes,
            t1 - t0, 0.0, 0.0, 0.0, False,
            error=f"decompress failed: {d_proc.stderr[:300]}",
        )

    # Byte-compare
    roundtrip_ok = _files_equal(input_path, decompressed)
    compress_wall = t1 - t0
    decompress_wall = t3 - t2
    compressed_bytes = compressed.stat().st_size
    ratio = compressed_bytes / input_bytes if input_bytes else 0.0
    compress_kbs = (input_bytes / 1024.0) / compress_wall if compress_wall > 0 else 0.0
    decompress_kbs = (input_bytes / 1024.0) / decompress_wall if decompress_wall > 0 else 0.0

    return Measurement(
        specialist, input_path.name, file_label, input_bytes, compressed_bytes,
        ratio, compress_wall, decompress_wall, compress_kbs, decompress_kbs,
        roundtrip_ok,
    )


def _files_equal(a: Path, b: Path) -> bool:
    if a.stat().st_size != b.stat().st_size:
        return False
    chunk = 1 << 16
    with a.open("rb") as fa, b.open("rb") as fb:
        while True:
            ba = fa.read(chunk)
            bb = fb.read(chunk)
            if ba != bb:
                return False
            if not ba:
                return True


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", default=[], help="Limit to specific specialists")
    parser.add_argument(
        "--corpus", type=Path, default=V010_BENCHMARK,
        help="Benchmark corpus directory (must contain manifest.tsv)",
    )
    parser.add_argument(
        "--out-md", type=Path,
        default=REPO_ROOT / "bench" / "specialist_benchmarks.md",
    )
    parser.add_argument(
        "--out-json", type=Path,
        default=REPO_ROOT / "bench" / "specialist_benchmarks.json",
    )
    parser.add_argument(
        "--cross-domain", action="store_true",
        help="Also measure each specialist on every OTHER domain's file",
    )
    args = parser.parse_args(argv)

    if not L3TC_BIN.exists():
        print(f"error: l3tc binary not built at {L3TC_BIN}", file=sys.stderr)
        return 2

    targets = args.only if args.only else SPECIALISTS
    available = []
    for s in targets:
        ok, reason = specialist_available(s)
        if ok:
            available.append(s)
        else:
            print(f"warn: specialist `{s}` not available: {reason}", file=sys.stderr)
    if not available:
        print("error: no specialists available. Waiting on training-engineer task 13.", file=sys.stderr)
        print(f"Drop .bin + tokenizer.model into {MODELS_DIR}/{{specialist}}/ and re-run.", file=sys.stderr)
        return 2

    in_domain = load_manifest_in_domain(args.corpus)
    if not in_domain:
        print(
            f"error: no manifest at {args.corpus}/manifest.tsv. "
            f"Run bench/build_v010_benchmark.py first.",
            file=sys.stderr,
        )
        return 2

    print(f"measuring {len(available)} specialists: {available}", file=sys.stderr)
    print(
        f"corpus: {args.corpus} "
        f"({sum(len(v) for v in in_domain.values())} files across "
        f"{len(in_domain)} domains)",
        file=sys.stderr,
    )

    results: list[Measurement] = []
    for specialist in available:
        for fpath in in_domain.get(specialist, []):
            print(f"  {specialist} on {fpath.name} (in-domain)...", file=sys.stderr)
            m = measure_one(specialist, fpath, "in-domain", WORK_DIR)
            results.append(m)

        # Cross-domain: run this specialist on every OTHER domain's files.
        if args.cross_domain:
            for other in SPECIALISTS:
                if other == specialist:
                    continue
                for fpath in in_domain.get(other, []):
                    print(f"  {specialist} on {fpath.name} (cross, {other})...", file=sys.stderr)
                    m = measure_one(specialist, fpath, f"cross-{other}", WORK_DIR)
                    results.append(m)

    # Write outputs
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump({
            "results": [asdict(r) for r in results],
            "ratio_targets": RATIO_TARGETS,
            "speed_target_kbs": SPEED_TARGET_KBS,
        }, f, indent=2)

    # Summary table
    lines = ["# Per-specialist ratio + speed benchmark\n"]
    lines.append("## Release-gate summary\n")
    lines.append(f"- Speed target: ≥{SPEED_TARGET_KBS:.0f} KB/s (Mac CPU parallelized)")
    lines.append("- Ratio target (in-domain): per PHASE_14 per-specialist target ±10%")
    lines.append("")
    lines.append("| specialist | file | label | ratio | target | compress KB/s | decompress KB/s | RT |")
    lines.append("|---|---|---|---:|---:|---:|---:|:-:|")
    for r in results:
        if r.error:
            lines.append(f"| {r.specialist} | {r.file} | {r.file_label} | — | — | — | — | ERR |")
            continue
        target = RATIO_TARGETS.get(r.specialist, 0.30) if r.file_label == "in-domain" else "—"
        target_s = f"{target:.2f}" if isinstance(target, float) else target
        rt = "✓" if r.roundtrip_ok else "✗"
        lines.append(
            f"| {r.specialist} | {r.file} | {r.file_label} | {r.ratio:.4f} | {target_s} | "
            f"{r.compress_kb_per_s:.1f} | {r.decompress_kb_per_s:.1f} | {rt} |"
        )
    lines.append("")

    # Release-gate summary
    lines.append("## Release-gate check\n")
    lines.append("| specialist | in-domain ratio | ratio ≤ target ±10% | compress KB/s | speed ≥150 | RT |")
    lines.append("|---|---:|:-:|---:|:-:|:-:|")
    by_spec: dict[str, list[Measurement]] = {}
    for r in results:
        if r.file_label == "in-domain":
            by_spec.setdefault(r.specialist, []).append(r)
    for s in SPECIALISTS:
        ms = by_spec.get(s, [])
        if not ms:
            lines.append(f"| {s} | — | — | — | — | — |")
            continue
        best_ratio = min((m.ratio for m in ms if not m.error), default=0.0)
        median_speed = sorted([m.compress_kb_per_s for m in ms if not m.error])
        median_speed = median_speed[len(median_speed) // 2] if median_speed else 0.0
        target = RATIO_TARGETS.get(s, 0.30)
        ratio_ok = best_ratio <= target * 1.10
        speed_ok = median_speed >= SPEED_TARGET_KBS
        rt_ok = all(m.roundtrip_ok for m in ms if not m.error)
        lines.append(
            f"| {s} | {best_ratio:.4f} | "
            f"{'PASS' if ratio_ok else 'FAIL'} | "
            f"{median_speed:.1f} | {'PASS' if speed_ok else 'FAIL'} | "
            f"{'✓' if rt_ok else '✗'} |"
        )
    lines.append("")

    args.out_md.write_text("\n".join(lines))
    print(f"results: {args.out_md}", file=sys.stderr)
    print(f"details: {args.out_json}", file=sys.stderr)

    # Exit 0 if all gates pass
    failed = any(
        (
            m.error
            or not m.roundtrip_ok
            or (m.file_label == "in-domain" and m.ratio > RATIO_TARGETS.get(m.specialist, 0.30) * 1.10)
            or m.compress_kb_per_s < SPEED_TARGET_KBS
        )
        for m in results
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())

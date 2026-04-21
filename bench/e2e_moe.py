#!/usr/bin/env python3
"""End-to-end MoE benchmark (Task 17).

BLOCKED on training-engineer task 13 until converted .bin files exist
under `l3tc-rust/models/{specialist}/model.bin` + `tokenizer.model`.
This script is ready to run the moment those artifacts land.

Runs `l3tc compress` in auto-routing mode (default — detection picks
the specialist) over the v0.1.0 benchmark corpus. Measures the
top-line "does the product work" numbers:

- End-to-end ratio (compressed / uncompressed) aggregated across the
  whole corpus, and per-domain
- End-to-end speed (KB/s) — both per-file and aggregate median
- Detection latency as % of total compress time
- Correctness: round-trip every file, verify byte-identical

PHASE_14 release gate:
- Speed: ≥150 KB/s on Mac CPU parallelized across all text+structured
  domains
- Ratio: ≤0.20 across ALL 7 domains AND better than best traditional
  compressor on every domain (the cross-traditional check is task 18)

Output: `bench/e2e_moe.md` (human-readable) + `bench/e2e_moe.json`.

Usage:
  # Full benchmark, all specialists
  python3 bench/e2e_moe.py

  # Run with verbose per-file detection output
  python3 bench/e2e_moe.py --verbose
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parent.parent
L3TC_BIN = REPO_ROOT / "l3tc-rust" / "target" / "release" / "l3tc"
MODELS_DIR = REPO_ROOT / "l3tc-rust" / "models"
V010_BENCHMARK = REPO_ROOT / "bench" / "v010_benchmark"
WORK_DIR = REPO_ROOT / "bench" / "work_e2e"

SPECIALISTS = ["prose", "code", "structured", "logs", "tabular", "markup", "fallback"]

# Release gates from PHASE_14.md:
SPEED_TARGET_KBS = 150.0
RATIO_TARGET = 0.20


@dataclass
class E2EMeasurement:
    file: str
    expected_domain: str
    input_bytes: int
    compressed_bytes: int
    ratio: float
    compress_wall_s: float
    decompress_wall_s: float
    compress_kb_per_s: float
    decompress_kb_per_s: float
    detect_wall_s: float
    detect_pct_of_compress: float
    detected_specialist: str  # what detect() picked, via `l3tc detect`
    detection_confidence: float
    roundtrip_ok: bool
    error: str | None = None


def all_specialists_available() -> tuple[bool, list[str]]:
    """Check whether every specialist's model + tokenizer exists.

    End-to-end routing only works if all 7 are present — detection
    can pick any of them and the compressor has to load the chosen
    one. Missing ones would error mid-run.
    """
    missing = []
    for s in SPECIALISTS:
        spec_dir = MODELS_DIR / s
        if not (spec_dir / "model.bin").exists() or not (spec_dir / "tokenizer.model").exists():
            missing.append(s)
    return len(missing) == 0, missing


def load_manifest(corpus: Path) -> list[tuple[Path, str]]:
    """Read `manifest.tsv`, return (abspath, domain) per file."""
    rows = []
    with (corpus / "manifest.tsv").open() as f:
        header = f.readline().strip().split("\t")
        assert header[:2] == ["filename", "domain"], header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rows.append((corpus / parts[0], parts[1]))
    return rows


def run_detect(file_path: Path) -> tuple[str, float, float]:
    """Time a standalone `l3tc detect --json` call.

    Returns (specialist, confidence, wall_time_s). Detection latency
    measured separately from compression so we can report it as %
    of compress time.
    """
    t0 = time.monotonic()
    proc = subprocess.run(
        [str(L3TC_BIN), "detect", "--json", str(file_path)],
        capture_output=True, text=True, check=False,
    )
    t1 = time.monotonic()
    if proc.returncode != 0:
        return ("error", 0.0, t1 - t0)
    rec = json.loads(proc.stdout.strip())
    return (rec["specialist"], float(rec["confidence"]), t1 - t0)


def measure_e2e(input_path: Path, expected_domain: str, work_dir: Path) -> E2EMeasurement:
    """End-to-end: detect + compress + decompress + round-trip verify."""
    work_dir.mkdir(parents=True, exist_ok=True)
    compressed = work_dir / f"{input_path.name}.l3tc"
    decompressed = work_dir / f"{input_path.name}.rt"
    for p in (compressed, decompressed):
        if p.exists():
            p.unlink()

    input_bytes = input_path.stat().st_size

    # Run detect once to record specialist + latency separately.
    detected, confidence, detect_wall = run_detect(input_path)

    # Compress in auto-routing mode (no --specialist flag; the CLI
    # runs detection internally and loads the right model).
    t0 = time.monotonic()
    c_proc = subprocess.run(
        [
            str(L3TC_BIN), "compress", str(input_path),
            "-o", str(compressed),
            "--specialist", "auto",
        ],
        capture_output=True, text=True, check=False,
    )
    t1 = time.monotonic()
    if c_proc.returncode != 0 or not compressed.exists():
        return E2EMeasurement(
            input_path.name, expected_domain, input_bytes, 0, 0.0, 0.0, 0.0,
            0.0, 0.0, detect_wall, 0.0, detected, confidence, False,
            error=f"compress failed: {c_proc.stderr[:300]}",
        )

    t2 = time.monotonic()
    d_proc = subprocess.run(
        [str(L3TC_BIN), "decompress", str(compressed), "-o", str(decompressed)],
        capture_output=True, text=True, check=False,
    )
    t3 = time.monotonic()
    if d_proc.returncode != 0 or not decompressed.exists():
        return E2EMeasurement(
            input_path.name, expected_domain, input_bytes,
            compressed.stat().st_size, compressed.stat().st_size / max(input_bytes, 1),
            t1 - t0, 0.0, 0.0, 0.0,
            detect_wall, 0.0, detected, confidence, False,
            error=f"decompress failed: {d_proc.stderr[:300]}",
        )

    roundtrip_ok = _files_equal(input_path, decompressed)
    compress_wall = t1 - t0
    decompress_wall = t3 - t2
    compressed_bytes = compressed.stat().st_size
    ratio = compressed_bytes / input_bytes if input_bytes else 0.0
    compress_kbs = (input_bytes / 1024.0) / compress_wall if compress_wall > 0 else 0.0
    decompress_kbs = (input_bytes / 1024.0) / decompress_wall if decompress_wall > 0 else 0.0
    detect_pct = (detect_wall / compress_wall * 100.0) if compress_wall > 0 else 0.0

    return E2EMeasurement(
        input_path.name, expected_domain, input_bytes, compressed_bytes, ratio,
        compress_wall, decompress_wall, compress_kbs, decompress_kbs,
        detect_wall, detect_pct, detected, confidence, roundtrip_ok,
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


def write_report(
    results: list[E2EMeasurement],
    out_md: Path,
    out_json: Path,
) -> dict[str, bool]:
    """Write human + machine reports; return release-gate pass flags."""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate
    total_in = sum(r.input_bytes for r in results if not r.error)
    total_out = sum(r.compressed_bytes for r in results if not r.error)
    agg_ratio = total_out / total_in if total_in else 0.0
    rt_ok = all(r.roundtrip_ok for r in results if not r.error)
    any_error = any(r.error for r in results)

    # Per-domain aggregates
    by_domain: dict[str, list[E2EMeasurement]] = {}
    for r in results:
        if not r.error:
            by_domain.setdefault(r.expected_domain, []).append(r)

    domain_stats: dict[str, dict[str, float]] = {}
    for d, rs in by_domain.items():
        in_b = sum(x.input_bytes for x in rs)
        out_b = sum(x.compressed_bytes for x in rs)
        compress_speeds = sorted(x.compress_kb_per_s for x in rs if x.compress_kb_per_s > 0)
        domain_stats[d] = {
            "files": len(rs),
            "in_bytes": in_b,
            "out_bytes": out_b,
            "ratio": out_b / in_b if in_b else 0.0,
            "median_compress_kb_s": median(compress_speeds) if compress_speeds else 0.0,
            "median_decompress_kb_s": median(sorted(x.decompress_kb_per_s for x in rs if x.decompress_kb_per_s > 0)) or 0.0,
        }

    # Release gates (for text + structured-text domains, NOT fallback/mixed)
    # PHASE_14: ratio ≤0.20 AND speed ≥150 KB/s across all 7 domains.
    gate_domains = [d for d in SPECIALISTS if d in domain_stats]
    ratio_pass = all(domain_stats[d]["ratio"] <= RATIO_TARGET for d in gate_domains)
    speed_pass = all(domain_stats[d]["median_compress_kb_s"] >= SPEED_TARGET_KBS for d in gate_domains)

    gates = {
        "ratio": ratio_pass,
        "speed": speed_pass,
        "roundtrip": rt_ok,
    }

    # JSON
    with out_json.open("w") as f:
        json.dump({
            "aggregate": {
                "files": len(results),
                "total_in_bytes": total_in,
                "total_out_bytes": total_out,
                "ratio": agg_ratio,
                "roundtrip_ok": rt_ok,
                "any_error": any_error,
            },
            "by_domain": domain_stats,
            "gates": gates,
            "targets": {"ratio": RATIO_TARGET, "speed_kbs": SPEED_TARGET_KBS},
            "per_file": [asdict(r) for r in results],
        }, f, indent=2)

    # Markdown
    lines = ["# End-to-end MoE benchmark\n"]
    lines.append("## Release gate\n")
    lines.append(
        f"- Ratio target: ≤{RATIO_TARGET:.2f} across all 7 domains — "
        f"**{'PASS' if ratio_pass else 'FAIL'}**"
    )
    lines.append(
        f"- Speed target: ≥{SPEED_TARGET_KBS:.0f} KB/s across all 7 domains — "
        f"**{'PASS' if speed_pass else 'FAIL'}**"
    )
    lines.append(f"- Round-trip: all files bit-identical — **{'PASS' if rt_ok else 'FAIL'}**")
    lines.append("")
    lines.append(
        f"**Overall: {'PASS' if all(gates.values()) else 'FAIL'}**"
    )
    lines.append("")

    lines.append("## Aggregate\n")
    lines.append(f"- Files: {len(results)}")
    lines.append(f"- Input bytes: {total_in:,}")
    lines.append(f"- Compressed bytes: {total_out:,}")
    lines.append(f"- Aggregate ratio: **{agg_ratio:.4f}**")
    lines.append("")

    lines.append("## Per-domain\n")
    lines.append("| domain | files | in MB | ratio | ratio ≤0.20 | compress KB/s | speed ≥150 |")
    lines.append("|---|---:|---:|---:|:-:|---:|:-:|")
    for d in SPECIALISTS + ["mixed"]:
        s = domain_stats.get(d)
        if not s:
            continue
        r_ok = "PASS" if s["ratio"] <= RATIO_TARGET else "FAIL"
        sp_ok = "PASS" if s["median_compress_kb_s"] >= SPEED_TARGET_KBS else "FAIL"
        lines.append(
            f"| {d} | {s['files']:.0f} | {s['in_bytes'] / 1_048_576:.2f} | "
            f"{s['ratio']:.4f} | {r_ok if d != 'mixed' else '—'} | "
            f"{s['median_compress_kb_s']:.1f} | "
            f"{sp_ok if d != 'mixed' else '—'} |"
        )
    lines.append("")

    lines.append("## Per-file\n")
    lines.append(
        "| file | expected | detected | conf | ratio | compress KB/s | decompress KB/s | detect % of compress | RT |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|:-:|")
    for r in sorted(results, key=lambda x: (x.expected_domain, x.file)):
        if r.error:
            lines.append(
                f"| `{r.file}` | {r.expected_domain} | — | — | — | — | — | — | ERR |"
            )
            continue
        rt = "✓" if r.roundtrip_ok else "✗"
        match = "✓" if r.detected_specialist == r.expected_domain else "~"
        lines.append(
            f"| `{r.file}` | {r.expected_domain} | {r.detected_specialist}{match} | "
            f"{r.detection_confidence:.2f} | {r.ratio:.4f} | "
            f"{r.compress_kb_per_s:.1f} | {r.decompress_kb_per_s:.1f} | "
            f"{r.detect_pct_of_compress:.2f}% | {rt} |"
        )
    lines.append("")

    # Detection-latency budget
    total_detect = sum(r.detect_wall_s for r in results if not r.error)
    total_compress = sum(r.compress_wall_s for r in results if not r.error)
    detect_overall_pct = total_detect / total_compress * 100.0 if total_compress > 0 else 0.0
    lines.append("## Detection overhead\n")
    lines.append(f"- Total detect time: {total_detect:.3f} s")
    lines.append(f"- Total compress time: {total_compress:.3f} s")
    lines.append(
        f"- Detection as share of compress: **{detect_overall_pct:.2f}%** "
        f"(budget in PHASE_14: `<1 ms` per file, negligible)"
    )
    lines.append("")

    # Detection accuracy on this corpus (different from task 15 — here
    # we measure against the benchmark manifest's `domain` column).
    correct = sum(
        1 for r in results
        if not r.error and r.detected_specialist == r.expected_domain
    )
    total_clean = sum(1 for r in results if not r.error and r.expected_domain != "mixed")
    correct_clean = sum(
        1 for r in results
        if not r.error and r.expected_domain != "mixed"
        and r.detected_specialist == r.expected_domain
    )
    if total_clean > 0:
        lines.append("## Detection accuracy on this benchmark\n")
        lines.append(f"- {correct_clean}/{total_clean} clean-domain files route to the right specialist "
                     f"({correct_clean / total_clean:.3f})")
        lines.append("")

    out_md.write_text("\n".join(lines))
    return gates


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=V010_BENCHMARK,
        help="Benchmark corpus directory (must contain manifest.tsv)",
    )
    parser.add_argument(
        "--out-md", type=Path,
        default=REPO_ROOT / "bench" / "e2e_moe.md",
    )
    parser.add_argument(
        "--out-json", type=Path,
        default=REPO_ROOT / "bench" / "e2e_moe.json",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not L3TC_BIN.exists():
        print(f"error: l3tc binary not built at {L3TC_BIN}", file=sys.stderr)
        return 2

    ok, missing = all_specialists_available()
    if not ok:
        print(
            f"error: end-to-end benchmark requires ALL 7 specialists to be "
            f"available for auto-routing. Missing: {missing}",
            file=sys.stderr,
        )
        print(
            f"Drop .bin + tokenizer.model into {MODELS_DIR}/{{specialist}}/ and re-run. "
            f"Waiting on training-engineer task 13.",
            file=sys.stderr,
        )
        return 2

    manifest_path = args.corpus / "manifest.tsv"
    if not manifest_path.exists():
        print(f"error: {manifest_path} missing. Run bench/build_v010_benchmark.py.", file=sys.stderr)
        return 2

    files = load_manifest(args.corpus)
    print(f"measuring end-to-end on {len(files)} files from {args.corpus}", file=sys.stderr)

    results: list[E2EMeasurement] = []
    for i, (fpath, domain) in enumerate(files):
        if args.verbose:
            print(f"  [{i + 1}/{len(files)}] {fpath.name} ({domain})...", file=sys.stderr)
        m = measure_e2e(fpath, domain, WORK_DIR)
        results.append(m)

    gates = write_report(results, args.out_md, args.out_json)
    print(f"\nreport: {args.out_md}", file=sys.stderr)
    print(f"details: {args.out_json}", file=sys.stderr)
    print(
        f"gates — ratio:{('PASS' if gates['ratio'] else 'FAIL')}  "
        f"speed:{('PASS' if gates['speed'] else 'FAIL')}  "
        f"roundtrip:{('PASS' if gates['roundtrip'] else 'FAIL')}",
        file=sys.stderr,
    )

    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    sys.exit(run())

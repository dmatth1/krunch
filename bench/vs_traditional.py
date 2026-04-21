#!/usr/bin/env python3
"""vs traditional compressors (Task 18).

BLOCKED on task 17 (which is blocked on task 13). When e2e_moe.py has
run, this harness consumes `bench/e2e_moe.json` for the l3tc numbers
and runs the traditional compressors (zstd-22, xz-9e, brotli-11,
gzip-9) on the same file set, then produces a head-to-head comparison.

Release gate (the strict PHASE_14 requirement):
  **l3tc ratio must be BETTER than the best traditional compressor on
   ALL 7 specialist domains.** If we lose on any domain, flag the
  specialist or tokenizer as a release blocker.

Output:
  - `bench/vs_traditional.md` — per-domain head-to-head table
  - `bench/vs_traditional.json` — full numeric detail
  - stdout: release-gate summary (exit 1 on failure)

Usage:
  # Full comparison (after e2e_moe.py has run)
  python3 bench/vs_traditional.py

  # Skip individual compressors (e.g., if brotli not installed)
  python3 bench/vs_traditional.py --skip brotli
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V010_BENCHMARK = REPO_ROOT / "bench" / "v010_benchmark"
WORK_DIR = REPO_ROOT / "bench" / "work_traditional"
E2E_JSON_DEFAULT = REPO_ROOT / "bench" / "e2e_moe.json"

SPECIALISTS = ["prose", "code", "structured", "logs", "tabular", "markup", "fallback"]

# Traditional compressor definitions — each entry is (name, compress-argv,
# decompress-argv, file-extension). The %INPUT% and %OUTPUT% placeholders
# are substituted by `measure_one()`. We use shell=True with `sh -c` to
# allow stdin/stdout redirection, mirroring bench/compressors.py.
#
# Settings use each tool's MAX-COMPRESSION preset to give traditional
# compressors their best possible showing — PHASE_14's gate says l3tc
# should still win.
TRADITIONAL: dict[str, dict[str, str]] = {
    "gzip-9": {
        "bin": "gzip",
        "ext": "gz",
        "compress": "gzip -9 -c %INPUT% > %OUTPUT%",
        "decompress": "gzip -d -c %INPUT% > %OUTPUT%",
    },
    "brotli-11": {
        "bin": "brotli",
        "ext": "br",
        "compress": "brotli -q 11 --large_window=30 -c %INPUT% > %OUTPUT%",
        "decompress": "brotli -d -c %INPUT% > %OUTPUT%",
    },
    "zstd-22": {
        "bin": "zstd",
        "ext": "zst",
        "compress": "zstd -q --ultra -22 --long=27 -c %INPUT% > %OUTPUT%",
        "decompress": "zstd -q -d -c %INPUT% > %OUTPUT%",
    },
    "xz-9e": {
        "bin": "xz",
        "ext": "xz",
        "compress": "xz -9e -c %INPUT% > %OUTPUT%",
        "decompress": "xz -d -c %INPUT% > %OUTPUT%",
    },
}


@dataclass
class TradMeasurement:
    compressor: str
    file: str
    domain: str
    input_bytes: int
    compressed_bytes: int
    ratio: float
    compress_wall_s: float
    decompress_wall_s: float
    compress_kb_per_s: float
    decompress_kb_per_s: float
    roundtrip_ok: bool
    error: str | None = None


def _sh_quote(p: Path) -> str:
    """Single-quote a path for sh -c."""
    s = str(p)
    return "'" + s.replace("'", "'\"'\"'") + "'"


def measure_one(comp_name: str, spec: dict[str, str], input_path: Path, domain: str) -> TradMeasurement:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    compressed = WORK_DIR / f"{input_path.name}.{spec['ext']}"
    decompressed = WORK_DIR / f"{input_path.name}.rt"
    for p in (compressed, decompressed):
        if p.exists():
            p.unlink()

    input_bytes = input_path.stat().st_size
    if input_bytes == 0:
        # 0-byte files can't meaningfully be measured.
        return TradMeasurement(
            comp_name, input_path.name, domain, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
            True, error="empty input",
        )

    compress_cmd = spec["compress"].replace("%INPUT%", _sh_quote(input_path)).replace(
        "%OUTPUT%", _sh_quote(compressed)
    )
    decompress_cmd = spec["decompress"].replace("%INPUT%", _sh_quote(compressed)).replace(
        "%OUTPUT%", _sh_quote(decompressed)
    )

    t0 = time.monotonic()
    rc = subprocess.run(["sh", "-c", compress_cmd], capture_output=True, text=True, check=False)
    t1 = time.monotonic()
    if rc.returncode != 0 or not compressed.exists():
        return TradMeasurement(
            comp_name, input_path.name, domain, input_bytes, 0, 0.0, 0.0, 0.0, 0.0, 0.0, False,
            error=f"compress rc={rc.returncode}: {rc.stderr[:200]}",
        )

    t2 = time.monotonic()
    rd = subprocess.run(["sh", "-c", decompress_cmd], capture_output=True, text=True, check=False)
    t3 = time.monotonic()
    if rd.returncode != 0 or not decompressed.exists():
        return TradMeasurement(
            comp_name, input_path.name, domain, input_bytes,
            compressed.stat().st_size, compressed.stat().st_size / input_bytes,
            t1 - t0, 0.0, 0.0, 0.0, False,
            error=f"decompress rc={rd.returncode}: {rd.stderr[:200]}",
        )

    rt_ok = _files_equal(input_path, decompressed)
    compress_wall = t1 - t0
    decompress_wall = t3 - t2
    compressed_bytes = compressed.stat().st_size
    ratio = compressed_bytes / input_bytes
    compress_kbs = (input_bytes / 1024.0) / compress_wall if compress_wall > 0 else 0.0
    decompress_kbs = (input_bytes / 1024.0) / decompress_wall if decompress_wall > 0 else 0.0

    return TradMeasurement(
        comp_name, input_path.name, domain, input_bytes, compressed_bytes, ratio,
        compress_wall, decompress_wall, compress_kbs, decompress_kbs, rt_ok,
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


def load_manifest(corpus: Path) -> list[tuple[Path, str]]:
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


def load_l3tc_results(e2e_json: Path) -> dict:
    """Read e2e_moe.json produced by bench/e2e_moe.py."""
    if not e2e_json.exists():
        return {}
    return json.loads(e2e_json.read_text())


def aggregate_by_domain(measurements: list[TradMeasurement]) -> dict[str, dict[str, float]]:
    by_dom: dict[str, list[TradMeasurement]] = {}
    for m in measurements:
        if m.error:
            continue
        by_dom.setdefault(m.domain, []).append(m)
    out = {}
    for d, ms in by_dom.items():
        in_b = sum(m.input_bytes for m in ms)
        out_b = sum(m.compressed_bytes for m in ms)
        compress_speeds = sorted(m.compress_kb_per_s for m in ms if m.compress_kb_per_s > 0)
        out[d] = {
            "files": len(ms),
            "in_bytes": in_b,
            "out_bytes": out_b,
            "ratio": out_b / in_b if in_b else 0.0,
            "median_compress_kb_s": statistics.median(compress_speeds) if compress_speeds else 0.0,
        }
    return out


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=V010_BENCHMARK)
    parser.add_argument("--e2e-json", type=Path, default=E2E_JSON_DEFAULT)
    parser.add_argument(
        "--skip", action="append", default=[],
        help="Skip a traditional compressor by name",
    )
    parser.add_argument(
        "--out-md", type=Path,
        default=REPO_ROOT / "bench" / "vs_traditional.md",
    )
    parser.add_argument(
        "--out-json", type=Path,
        default=REPO_ROOT / "bench" / "vs_traditional.json",
    )
    args = parser.parse_args(argv)

    manifest_path = args.corpus / "manifest.tsv"
    if not manifest_path.exists():
        print(f"error: {manifest_path} missing. Run bench/build_v010_benchmark.py.", file=sys.stderr)
        return 2

    # Availability check for traditional compressors
    wanted = {name: spec for name, spec in TRADITIONAL.items() if name not in args.skip}
    available: dict[str, dict[str, str]] = {}
    for name, spec in wanted.items():
        if shutil.which(spec["bin"]):
            available[name] = spec
        else:
            print(f"warn: {spec['bin']} not on PATH — skipping {name}", file=sys.stderr)
    if not available:
        print("error: no traditional compressors available", file=sys.stderr)
        return 2

    files = load_manifest(args.corpus)
    print(
        f"running {len(available)} traditional compressors on {len(files)} files",
        file=sys.stderr,
    )

    all_results: list[TradMeasurement] = []
    for name, spec in available.items():
        print(f"  {name}...", file=sys.stderr)
        for fpath, domain in files:
            m = measure_one(name, spec, fpath, domain)
            all_results.append(m)

    # Aggregate per-domain per-compressor
    per_comp: dict[str, dict[str, dict[str, float]]] = {}
    for name in available:
        ms = [m for m in all_results if m.compressor == name]
        per_comp[name] = aggregate_by_domain(ms)

    # Load l3tc results
    l3tc = load_l3tc_results(args.e2e_json)
    l3tc_by_domain = l3tc.get("by_domain", {}) if l3tc else {}

    # Head-to-head per domain
    head_to_head: list[dict] = []
    release_gate_fails: list[str] = []

    for d in SPECIALISTS:
        l3tc_ratio = l3tc_by_domain.get(d, {}).get("ratio")
        row = {
            "domain": d,
            "l3tc_ratio": l3tc_ratio,
            "traditional": {},
            "best_traditional_name": None,
            "best_traditional_ratio": None,
            "l3tc_wins": None,
        }
        best_ratio = None
        best_name = None
        for name in available:
            r = per_comp[name].get(d, {}).get("ratio")
            row["traditional"][name] = r
            if r is not None and (best_ratio is None or r < best_ratio):
                best_ratio = r
                best_name = name
        row["best_traditional_ratio"] = best_ratio
        row["best_traditional_name"] = best_name
        if l3tc_ratio is not None and best_ratio is not None:
            # PHASE_14 gate: l3tc must be BETTER (lower ratio).
            l3tc_wins = l3tc_ratio < best_ratio
            row["l3tc_wins"] = l3tc_wins
            if not l3tc_wins:
                release_gate_fails.append(d)
        head_to_head.append(row)

    # JSON output
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump({
            "available_compressors": sorted(available.keys()),
            "per_compressor_per_domain": per_comp,
            "l3tc_per_domain": l3tc_by_domain,
            "head_to_head": head_to_head,
            "release_gate_fails": release_gate_fails,
            "all_measurements": [asdict(r) for r in all_results],
        }, f, indent=2)

    # Markdown report
    lines = ["# l3tc vs traditional compressors\n"]
    lines.append("## Release gate\n")
    lines.append(
        "PHASE_14: l3tc ratio must be BETTER than the best traditional "
        "compressor on ALL 7 specialist domains. If l3tc loses on any "
        "domain, that specialist or its tokenizer is a release blocker."
    )
    lines.append("")
    if not l3tc_by_domain:
        lines.append(
            "**No l3tc numbers available** — run `bench/e2e_moe.py` first "
            "so `bench/e2e_moe.json` exists."
        )
    elif release_gate_fails:
        lines.append(
            f"**RELEASE GATE: FAIL** — l3tc loses on: "
            f"{', '.join(release_gate_fails)}."
        )
    else:
        lines.append("**RELEASE GATE: PASS** — l3tc beats best traditional on all 7 domains.")
    lines.append("")

    lines.append("## Per-domain head-to-head (ratio, lower is better)\n")
    headers = ["domain", "l3tc"] + list(available.keys()) + ["best-trad", "winner"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "---|" * len(headers))
    for row in head_to_head:
        cells = [row["domain"]]
        cells.append(f"**{row['l3tc_ratio']:.4f}**" if row["l3tc_ratio"] is not None else "—")
        for name in available:
            r = row["traditional"].get(name)
            cells.append(f"{r:.4f}" if r is not None else "—")
        cells.append(
            f"{row['best_traditional_ratio']:.4f} ({row['best_traditional_name']})"
            if row["best_traditional_ratio"] is not None else "—"
        )
        if row["l3tc_wins"] is None:
            cells.append("—")
        elif row["l3tc_wins"]:
            cells.append("l3tc")
        else:
            cells.append(f"**{row['best_traditional_name']}**")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Per-compressor per-domain detail\n")
    for name in sorted(available):
        lines.append(f"### {name}\n")
        lines.append("| domain | files | in MB | ratio | compress KB/s |")
        lines.append("|---|---:|---:|---:|---:|")
        for d in SPECIALISTS + ["mixed"]:
            s = per_comp[name].get(d)
            if not s:
                continue
            lines.append(
                f"| {d} | {s['files']:.0f} | {s['in_bytes'] / 1_048_576:.2f} | "
                f"{s['ratio']:.4f} | {s['median_compress_kb_s']:.1f} |"
            )
        lines.append("")

    args.out_md.write_text("\n".join(lines))
    print(f"\nreport: {args.out_md}", file=sys.stderr)
    print(f"details: {args.out_json}", file=sys.stderr)
    if release_gate_fails:
        print(
            f"RELEASE GATE FAIL — l3tc loses on: {', '.join(release_gate_fails)}",
            file=sys.stderr,
        )
        return 1
    if not l3tc_by_domain:
        print("no l3tc numbers (run e2e_moe.py first)", file=sys.stderr)
        return 2
    print("RELEASE GATE PASS — l3tc beats best traditional on all 7 domains", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())

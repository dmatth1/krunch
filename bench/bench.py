#!/usr/bin/env python3
"""l3tc-prod benchmark harness.

Measures compressors on corpora with reproducible, diffable JSON output.
Standard-library only by design (see DECISIONS.md D7).

What we measure, per (compressor, corpus) pair:

  - input_bytes         : uncompressed size in bytes
  - compressed_bytes    : compressed size in bytes
  - ratio               : compressed_bytes / input_bytes (lower = better)
  - compress_wall_s     : median wall time to compress, seconds
  - compress_user_s     : median user CPU time, seconds
  - compress_sys_s      : median system CPU time, seconds
  - compress_peak_kb    : peak resident set size during compression, KB
  - compress_mb_per_s   : throughput during compression (input_bytes / wall)
  - decompress_*        : same for decompression
  - roundtrip_ok        : did decompress(compress(x)) == x exactly
  - samples             : number of repeated runs (we take median)
  - raw_wall_times      : list of individual wall times for variance analysis

Usage:

  # List all known compressors and availability
  python3 bench/bench.py --list

  # Benchmark all classical compressors on a single corpus
  python3 bench/bench.py --corpus bench/corpora/enwik6 --classical-only

  # Benchmark everything on all available corpora
  python3 bench/bench.py --all

  # Run only specific compressors
  python3 bench/bench.py --corpus foo.txt --compressor gzip-9 --compressor zstd-22

  # Specify output file (otherwise auto-generated from date)
  python3 bench/bench.py --all --output bench/results/2026-04-08.json

  # Quick mode: fewer samples for faster iteration during development
  python3 bench/bench.py --all --samples 1
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Make `bench` importable when running as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compressors import (  # noqa: E402
    Compressor,
    all_compressors,
    default_classical,
    default_neural,
)


# -------- Measurement core -------- #


@dataclass
class Measurement:
    """One (compress, decompress) measurement for one (compressor, corpus) pair."""

    compressor: str
    corpus: str
    input_bytes: int
    compressed_bytes: int
    ratio: float
    roundtrip_ok: bool
    compress_wall_s: float
    compress_user_s: float
    compress_sys_s: float
    compress_peak_kb: int
    compress_mb_per_s: float
    decompress_wall_s: float
    decompress_user_s: float
    decompress_sys_s: float
    decompress_peak_kb: int
    decompress_mb_per_s: float
    samples: int
    raw_compress_walls: list[float]
    raw_decompress_walls: list[float]
    error: str | None = None


def _rusage_children() -> tuple[float, float, int]:
    """Return (user_s, sys_s, maxrss_kb) for the children of the current process.

    maxrss is reported in KB on Linux and bytes on macOS (thanks, Apple).
    We normalize both to KB for consistency.
    """
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    user = r.ru_utime
    sys_t = r.ru_stime
    maxrss = r.ru_maxrss
    if sys.platform == "darwin":
        # On macOS, maxrss is in bytes, not kilobytes.
        maxrss = maxrss // 1024
    return user, sys_t, maxrss


def _time_run(cmd: list[str]) -> tuple[int, float, float, float, int, str]:
    """Run a command, return (rc, wall_s, user_s, sys_s, peak_kb, stderr).

    Uses wall-clock time (monotonic) for duration and getrusage for CPU
    time and peak memory. The rusage delta is taken before and after
    the subprocess runs — note that getrusage(RUSAGE_CHILDREN) reports
    cumulative child usage, so we subtract the baseline.
    """
    u0, s0, rss0 = _rusage_children()
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        rc = proc.returncode
        stderr = proc.stderr
    except FileNotFoundError as e:
        return -1, 0.0, 0.0, 0.0, 0, f"binary not found: {e}"
    t1 = time.monotonic()
    u1, s1, rss1 = _rusage_children()

    wall = t1 - t0
    user = u1 - u0
    sys_t = s1 - s0
    # ru_maxrss is a high-water mark for the process tree; we report the
    # post-run value as an approximation. Differential RSS would require
    # polling, which we don't do to keep the harness simple.
    peak_kb = max(rss1, 0)
    return rc, wall, user, sys_t, peak_kb, stderr


def measure_once(
    comp: Compressor,
    corpus: Path,
    work_dir: Path,
    verify_roundtrip: bool = True,
) -> dict[str, Any]:
    """Run compress + decompress once, return a dict of per-run measurements.

    Writes compressed output and decompressed output to `work_dir`.
    If `verify_roundtrip` is True, byte-compares the decompressed output
    to the original corpus. Round-trip failure is returned as an error.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    compressed_path = work_dir / f"{corpus.name}.{comp.extension}"
    decompressed_path = work_dir / f"{corpus.name}.roundtrip"

    # Clean previous attempts so we don't measure stale files.
    for p in (compressed_path, decompressed_path):
        if p.exists():
            p.unlink()

    # Compress
    compress_cmd = comp.compress_cmd(corpus, compressed_path)
    rc, cw, cu, cs, cp, cerr = _time_run(compress_cmd)
    if rc != 0 or not compressed_path.exists():
        return {"error": f"compress failed rc={rc}: {cerr[:500]}"}

    # Decompress
    decompress_cmd = comp.decompress_cmd(compressed_path, decompressed_path)
    rc, dw, du, ds, dp, derr = _time_run(decompress_cmd)
    if rc != 0 or not decompressed_path.exists():
        return {"error": f"decompress failed rc={rc}: {derr[:500]}"}

    # Round-trip verify
    roundtrip_ok = True
    if verify_roundtrip:
        roundtrip_ok = _files_equal(corpus, decompressed_path)

    result = {
        "input_bytes": corpus.stat().st_size,
        "compressed_bytes": compressed_path.stat().st_size,
        "compress_wall_s": cw,
        "compress_user_s": cu,
        "compress_sys_s": cs,
        "compress_peak_kb": cp,
        "decompress_wall_s": dw,
        "decompress_user_s": du,
        "decompress_sys_s": ds,
        "decompress_peak_kb": dp,
        "roundtrip_ok": roundtrip_ok,
    }
    return result


def _files_equal(a: Path, b: Path) -> bool:
    """Byte-compare two files. Returns True if identical."""
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


def measure(
    comp: Compressor,
    corpus: Path,
    work_dir: Path,
    samples: int = 3,
    verify_roundtrip: bool = True,
) -> Measurement:
    """Measure one (compressor, corpus) pair `samples` times, take medians.

    We report median rather than mean because compression timing is often
    right-skewed by OS scheduling jitter; median is a more stable summary
    for small sample counts.
    """
    raw_runs: list[dict[str, Any]] = []
    for _ in range(samples):
        r = measure_once(comp, corpus, work_dir, verify_roundtrip)
        if "error" in r:
            return Measurement(
                compressor=comp.name,
                corpus=corpus.name,
                input_bytes=corpus.stat().st_size if corpus.exists() else 0,
                compressed_bytes=0,
                ratio=0.0,
                roundtrip_ok=False,
                compress_wall_s=0.0,
                compress_user_s=0.0,
                compress_sys_s=0.0,
                compress_peak_kb=0,
                compress_mb_per_s=0.0,
                decompress_wall_s=0.0,
                decompress_user_s=0.0,
                decompress_sys_s=0.0,
                decompress_peak_kb=0,
                decompress_mb_per_s=0.0,
                samples=0,
                raw_compress_walls=[],
                raw_decompress_walls=[],
                error=r["error"],
            )
        raw_runs.append(r)

    def med(key: str) -> float:
        return statistics.median(r[key] for r in raw_runs)

    def med_int(key: str) -> int:
        return int(statistics.median(r[key] for r in raw_runs))

    input_bytes = raw_runs[0]["input_bytes"]
    compressed_bytes = raw_runs[0]["compressed_bytes"]
    compress_wall_s = med("compress_wall_s")
    decompress_wall_s = med("decompress_wall_s")

    compress_mb_per_s = (
        (input_bytes / 1_000_000) / compress_wall_s if compress_wall_s > 0 else 0.0
    )
    decompress_mb_per_s = (
        (input_bytes / 1_000_000) / decompress_wall_s if decompress_wall_s > 0 else 0.0
    )

    return Measurement(
        compressor=comp.name,
        corpus=corpus.name,
        input_bytes=input_bytes,
        compressed_bytes=compressed_bytes,
        ratio=compressed_bytes / input_bytes if input_bytes else 0.0,
        roundtrip_ok=all(r["roundtrip_ok"] for r in raw_runs),
        compress_wall_s=compress_wall_s,
        compress_user_s=med("compress_user_s"),
        compress_sys_s=med("compress_sys_s"),
        compress_peak_kb=med_int("compress_peak_kb"),
        compress_mb_per_s=compress_mb_per_s,
        decompress_wall_s=decompress_wall_s,
        decompress_user_s=med("decompress_user_s"),
        decompress_sys_s=med("decompress_sys_s"),
        decompress_peak_kb=med_int("decompress_peak_kb"),
        decompress_mb_per_s=decompress_mb_per_s,
        samples=len(raw_runs),
        raw_compress_walls=[r["compress_wall_s"] for r in raw_runs],
        raw_decompress_walls=[r["decompress_wall_s"] for r in raw_runs],
    )


# -------- Output formatting -------- #


def format_summary_table(results: list[Measurement]) -> str:
    """Render measurements as a markdown table, grouped by corpus."""
    if not results:
        return "_no results_\n"

    lines = []
    by_corpus: dict[str, list[Measurement]] = {}
    for r in results:
        by_corpus.setdefault(r.corpus, []).append(r)

    for corpus, rs in by_corpus.items():
        input_mb = rs[0].input_bytes / 1_000_000
        lines.append(f"\n## {corpus}  ({input_mb:.2f} MB)\n")
        lines.append("| Compressor | Ratio | Size (MB) | Compress MB/s | Decompress MB/s | RT |")
        lines.append("|---|---:|---:|---:|---:|:-:|")

        # Sort by ratio (best first), errors last
        rs_sorted = sorted(
            rs,
            key=lambda r: (r.error is not None, r.ratio if r.ratio > 0 else 999.0),
        )

        for r in rs_sorted:
            if r.error:
                lines.append(
                    f"| {r.compressor} | — | — | — | — | err |"
                )
                continue
            comp_mb = r.compressed_bytes / 1_000_000
            rt = "✓" if r.roundtrip_ok else "✗"
            lines.append(
                f"| {r.compressor} | {r.ratio:.4f} | "
                f"{comp_mb:.2f} | "
                f"{r.compress_mb_per_s:.2f} | "
                f"{r.decompress_mb_per_s:.2f} | "
                f"{rt} |"
            )

    return "\n".join(lines) + "\n"


def write_json(results: list[Measurement], output_path: Path) -> None:
    """Write measurements as a JSON blob."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "platform": sys.platform,
        "machine": _machine_info(),
        "timestamp": int(time.time()),
        "results": [asdict(r) for r in results],
    }
    output_path.write_text(json.dumps(payload, indent=2))


def _machine_info() -> dict[str, str]:
    """Best-effort machine identifier. Falls back gracefully."""
    info: dict[str, str] = {}
    try:
        info["uname"] = os.uname().machine  # type: ignore[attr-defined]
    except Exception:
        info["uname"] = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except FileNotFoundError:
        # macOS
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=False,
            )
            info["cpu"] = out.stdout.strip() or "unknown"
        except Exception:
            info["cpu"] = "unknown"
    return info


# -------- CLI -------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="l3tc-prod benchmark harness")
    parser.add_argument(
        "--corpus",
        type=Path,
        action="append",
        default=[],
        help="Corpus file to benchmark. Can be specified multiple times.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run on every file in bench/corpora/ that doesn't start with a dot.",
    )
    parser.add_argument(
        "--compressor",
        action="append",
        default=[],
        help="Limit to specific compressor(s) by name. Can be specified multiple times.",
    )
    parser.add_argument(
        "--classical-only",
        action="store_true",
        help="Skip neural compressors (L3TC), run only classical ones.",
    )
    parser.add_argument(
        "--neural-only",
        action="store_true",
        help="Skip classical compressors, run only neural ones.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of runs per (compressor, corpus) pair. Median reported.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write JSON results. Default: bench/results/<date>.json",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("bench/work"),
        help="Scratch directory for intermediate files.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List known compressors and availability, then exit.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip round-trip verification (faster but risky).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print each measurement as it finishes.",
    )
    args = parser.parse_args(argv)

    if args.list:
        print("Classical compressors:")
        for c in default_classical():
            avail = "available" if c.available() else "NOT FOUND"
            print(f"  {c.name:20s} {avail}")
        print()
        print("Neural compressors:")
        for c in default_neural():
            avail = "available" if c.available() else "NOT FOUND (run scripts/setup.sh)"
            print(f"  {c.name:30s} {avail}")
        return 0

    # Resolve compressors
    comps: list[Compressor] = []
    if args.classical_only and args.neural_only:
        print("error: --classical-only and --neural-only are mutually exclusive", file=sys.stderr)
        return 2

    if args.classical_only:
        comps = default_classical()
    elif args.neural_only:
        comps = default_neural()
    else:
        comps = all_compressors()

    if args.compressor:
        wanted = set(args.compressor)
        comps = [c for c in comps if c.name in wanted]
        unknown = wanted - {c.name for c in comps}
        if unknown:
            print(
                f"warning: unknown compressor names: {sorted(unknown)}",
                file=sys.stderr,
            )
        if not comps:
            print("error: no compressors matched --compressor filter", file=sys.stderr)
            return 2

    comps = [c for c in comps if c.available()]
    if not comps:
        print("error: no compressors available. Run scripts/setup.sh?", file=sys.stderr)
        return 2

    # Resolve corpora
    corpora: list[Path] = list(args.corpus)
    if args.all:
        corpora_dir = Path("bench/corpora")
        if corpora_dir.is_dir():
            corpora.extend(
                sorted(
                    p
                    for p in corpora_dir.iterdir()
                    if p.is_file() and not p.name.startswith(".") and p.name != "README.md"
                )
            )
    if not corpora:
        print("error: no corpora specified. Use --corpus PATH or --all", file=sys.stderr)
        return 2

    # Existence check
    missing = [c for c in corpora if not c.exists()]
    if missing:
        print(f"error: corpora not found: {missing}", file=sys.stderr)
        return 2

    # Run measurements
    all_results: list[Measurement] = []
    total = len(comps) * len(corpora)
    done = 0
    for corpus in corpora:
        for comp in comps:
            done += 1
            if args.verbose:
                print(
                    f"[{done}/{total}] {comp.name} on {corpus.name}...",
                    file=sys.stderr,
                    flush=True,
                )
            try:
                m = measure(
                    comp=comp,
                    corpus=corpus,
                    work_dir=args.work_dir,
                    samples=args.samples,
                    verify_roundtrip=not args.no_verify,
                )
                all_results.append(m)
                if args.verbose and m.error is None:
                    print(
                        f"    ratio={m.ratio:.4f}  "
                        f"comp={m.compress_mb_per_s:.1f} MB/s  "
                        f"dec={m.decompress_mb_per_s:.1f} MB/s  "
                        f"rt={'ok' if m.roundtrip_ok else 'FAIL'}",
                        file=sys.stderr,
                        flush=True,
                    )
                elif args.verbose:
                    print(f"    ERROR: {m.error[:200]}", file=sys.stderr, flush=True)
            except KeyboardInterrupt:
                print("\ninterrupted", file=sys.stderr)
                break

    # Output
    if args.output is None:
        args.output = Path(f"bench/results/{time.strftime('%Y-%m-%d-%H%M%S')}.json")

    write_json(all_results, args.output)
    print(f"\nresults written to {args.output}")

    summary_path = args.output.with_suffix(".md")
    summary_path.write_text(format_summary_table(all_results))
    print(f"summary written to {summary_path}")

    # Also print summary to stdout
    print()
    print(format_summary_table(all_results))

    # Clean work dir on success
    if args.work_dir.exists():
        shutil.rmtree(args.work_dir, ignore_errors=True)

    # Exit nonzero if any measurement had an error or round-trip failure
    has_errors = any(r.error or not r.roundtrip_ok for r in all_results)
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())

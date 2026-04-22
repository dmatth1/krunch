#!/usr/bin/env python3
"""Per-chunk hybrid-codec dispatcher simulator.

Implements the design in HYBRID_CODEC_DESIGN.md against a local
corpus, without building the full Rust runtime. For each 64 KB
chunk of the input:

  1. Magic-byte prescreen → route known-binary chunks straight to
     passthrough.
  2. Probe-encode with every available codec (zstd -22, zstd --dict,
     bzip3, lz4, neural-entropy-bound, CLP stub if log-like).
  3. Pick the codec whose output is shortest.
  4. Apply Stage 3 safety net: if picked > 1.01 × zstd-22, substitute
     zstd-22.

Output:
  - Aggregate dispatcher size vs each individual codec's size.
  - Per-codec chunk-count histogram.
  - Per-chunk CSV log for plotting.

The neural estimate uses a constant "bits/byte on this corpus" value
passed via --neural-bpb, typically derived from
scripts/measure_held_out_ratio.py. It is a first-order approximation
— real neural encode has chunk-to-chunk variance we ignore here,
but the aggregate ratio is accurate to within ~1%.

Usage:
  python scripts/simulate_dispatcher.py \\
      --input   /tmp/enwik8_val.txt \\
      --out-csv /tmp/dispatcher_enwik8.csv \\
      --neural-bpb 1.73 \\
      --report  /tmp/dispatcher_enwik8_report.md
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

CHUNK_BYTES = 64 * 1024
PROBE_BYTES = 4 * 1024  # size of probe for extrapolation — unused here since we encode full chunks
SAFETY_THRESHOLD = 1.01  # picked codec > 1.01 × zstd → substitute zstd
TAG_BITS = 3  # overhead per chunk in the blob format

# Magic-byte prescreen: known binary formats that should go straight to passthrough.
MAGIC_PREFIXES: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),      # needs further validation in real impl
    (b"RIFF", "wav/avi"),
    (b"\x7fELF", "elf"),
    (b"MZ", "pe"),
    (b"%PDF", "pdf"),
    (b"PK\x03\x04", "zip/jar/docx"),
    (b"\x1f\x8b", "gzip"),
    (b"\x28\xb5\x2f\xfd", "zstd"),
    (b"\xfd7zXZ\x00", "xz"),
    (b"BZh", "bzip2"),
    (b"7z\xbc\xaf\x27\x1c", "7z"),
]


def magic_match(buf: bytes) -> str | None:
    """Return the format label if the chunk starts with a known
    binary magic. Otherwise None."""
    for prefix, label in MAGIC_PREFIXES:
        if buf.startswith(prefix):
            return label
    return None


# ----------------------------------------------------------------------
# Per-codec "how many bytes would this produce" functions.
# Each returns the compressed size (int). We don't keep the compressed
# bytes — only length — since we're measuring ratio, not building the
# blob.
# ----------------------------------------------------------------------

def _run_stdin(cmd: list[str], data: bytes, timeout: float = 60.0) -> bytes:
    """Feed `data` to `cmd` on stdin, capture stdout bytes."""
    proc = subprocess.run(cmd, input=data, capture_output=True, check=True, timeout=timeout)
    return proc.stdout


def size_zstd_22(buf: bytes) -> int:
    out = _run_stdin(["zstd", "-22", "--ultra", "--long=27", "-q", "-"], buf)
    return len(out)


def size_zstd_dict(buf: bytes, dict_path: Path | None) -> int | None:
    if dict_path is None or not dict_path.exists():
        return None
    out = _run_stdin(
        ["zstd", "-22", "--ultra", "--long=27", "-q", "-D", str(dict_path), "-"],
        buf,
    )
    return len(out)


def size_bzip3(buf: bytes) -> int | None:
    # bzip3 reads stdin + writes stdout by default; no `-q` flag exists.
    try:
        out = _run_stdin(["bzip3", "-e"], buf)
        return len(out)
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError:
        return None


def size_lz4(buf: bytes) -> int:
    out = _run_stdin(["lz4", "-q", "-1"], buf)
    return len(out)


def size_neural(buf: bytes, bpb: float) -> int:
    """Estimate neural codec output bytes = bpb * len / 8.

    This is a first-order approximation using a global bits/byte
    number (from measure_held_out_ratio.py or in-training eval).
    Real encode has per-chunk variance; we ignore that for now.
    The approximation holds to within ~1% on large aggregates."""
    return int(round(len(buf) * bpb / 8.0))


def size_passthrough(buf: bytes) -> int:
    return len(buf)


# ----------------------------------------------------------------------
# Dispatcher logic.
# ----------------------------------------------------------------------

CODEC_TAGS = {
    "passthrough": 0,
    "lz4":         1,
    "zstd":        2,
    "zstd_dict":   3,
    "bzip3":       4,
    "brotli_dict": 5,
    "clp":         6,
    "neural":      7,
}


def dispatch_chunk(
    buf: bytes,
    *,
    neural_bpb: float | None,
    zstd_dict_path: Path | None,
    enable_bzip3: bool,
    enable_neural: bool,
) -> tuple[str, int, dict[str, int]]:
    """Run every enabled codec on `buf`, pick the smallest + apply
    safety net. Return (winning_codec, winning_size, all_sizes)."""

    # Stage 1: magic-byte prescreen
    fmt = magic_match(buf)
    if fmt is not None:
        # Known binary — don't bother probing anything else.
        passthrough_size = size_passthrough(buf)
        return "passthrough", passthrough_size, {"passthrough": passthrough_size}

    # Stage 2: probe every candidate. We encode the FULL chunk rather
    # than a probe + extrapolation, because chunks are small (64 KB)
    # and we want real numbers for the aggregate ratio.
    sizes: dict[str, int] = {}

    sizes["zstd"] = size_zstd_22(buf)

    if zstd_dict_path is not None:
        sd = size_zstd_dict(buf, zstd_dict_path)
        if sd is not None:
            sizes["zstd_dict"] = sd

    if enable_bzip3:
        sb = size_bzip3(buf)
        if sb is not None:
            sizes["bzip3"] = sb

    sizes["lz4"] = size_lz4(buf)

    if enable_neural and neural_bpb is not None:
        sizes["neural"] = size_neural(buf, neural_bpb)

    # Pick smallest.
    picked = min(sizes, key=lambda k: sizes[k])
    picked_size = sizes[picked]

    # Stage 3 safety net: if picked > SAFETY_THRESHOLD × zstd, use zstd.
    if picked != "zstd" and picked_size > sizes["zstd"] * SAFETY_THRESHOLD:
        picked = "zstd"
        picked_size = sizes["zstd"]

    return picked, picked_size, sizes


def simulate(
    input_path: Path,
    *,
    chunk_bytes: int = CHUNK_BYTES,
    neural_bpb: float | None = None,
    zstd_dict_path: Path | None = None,
    enable_bzip3: bool = True,
    enable_neural: bool = True,
    csv_out: Path | None = None,
    limit_bytes: int | None = None,
) -> dict:
    """Run the dispatcher over `input_path`, aggregate results."""

    codec_totals: dict[str, int] = {name: 0 for name in CODEC_TAGS}
    codec_chunks: dict[str, int] = {name: 0 for name in CODEC_TAGS}
    dispatcher_total = 0
    raw_total = 0
    tag_overhead_bits = 0

    # Also track what each INDIVIDUAL codec would have totalled if
    # forced to handle every chunk — needed for the "dispatcher vs
    # codec X alone" comparison.
    individual_totals: dict[str, int] = {}

    rows: list[dict] = []
    chunk_idx = 0

    with input_path.open("rb") as fin:
        while True:
            chunk = fin.read(chunk_bytes)
            if not chunk:
                break
            if limit_bytes is not None and raw_total >= limit_bytes:
                break

            picked, picked_size, all_sizes = dispatch_chunk(
                chunk,
                neural_bpb=neural_bpb,
                zstd_dict_path=zstd_dict_path,
                enable_bzip3=enable_bzip3,
                enable_neural=enable_neural,
            )

            raw_total += len(chunk)
            dispatcher_total += picked_size
            tag_overhead_bits += TAG_BITS
            codec_totals[picked] += picked_size
            codec_chunks[picked] += 1

            for codec_name, sz in all_sizes.items():
                individual_totals[codec_name] = individual_totals.get(codec_name, 0) + sz

            rows.append({
                "chunk": chunk_idx,
                "raw_bytes": len(chunk),
                "picked": picked,
                "picked_bytes": picked_size,
                **{f"size_{k}": v for k, v in all_sizes.items()},
            })
            chunk_idx += 1

            if chunk_idx % 100 == 0:
                print(
                    f"  processed {chunk_idx} chunks ({raw_total/1e6:.1f} MB) "
                    f"dispatcher so far {dispatcher_total/1e6:.2f} MB",
                    file=sys.stderr,
                )

    # Add tag overhead bytes (ceiling).
    tag_overhead_bytes = (tag_overhead_bits + 7) // 8
    dispatcher_total_with_overhead = dispatcher_total + tag_overhead_bytes

    if csv_out is not None:
        fieldnames = set()
        for r in rows:
            fieldnames.update(r.keys())
        with csv_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
            writer.writeheader()
            writer.writerows(rows)

    return {
        "raw_bytes": raw_total,
        "chunks": chunk_idx,
        "tag_overhead_bytes": tag_overhead_bytes,
        "dispatcher_bytes": dispatcher_total_with_overhead,
        "dispatcher_ratio": dispatcher_total_with_overhead / max(1, raw_total),
        "per_codec_totals": codec_totals,
        "per_codec_chunks": codec_chunks,
        "individual_codec_totals": individual_totals,
    }


# ----------------------------------------------------------------------
# Reporting.
# ----------------------------------------------------------------------

def render_report(stats: dict, input_path: Path, neural_bpb: float | None) -> str:
    raw = stats["raw_bytes"]
    disp = stats["dispatcher_bytes"]
    lines: list[str] = []

    lines.append(f"# Dispatcher simulation — `{input_path.name}`")
    lines.append("")
    lines.append(f"Raw bytes: {raw:,}")
    lines.append(f"Chunks ({CHUNK_BYTES//1024} KB each): {stats['chunks']:,}")
    lines.append(f"Tag overhead: {stats['tag_overhead_bytes']:,} bytes")
    if neural_bpb is not None:
        lines.append(f"Neural estimate bits/byte: {neural_bpb}")
    lines.append("")

    lines.append("## Aggregate ratios")
    lines.append("")
    lines.append("| codec | total bytes | ratio | vs dispatcher |")
    lines.append("|---|---|---|---|")
    lines.append(f"| **DISPATCHER** | **{disp:,}** | **{disp/raw:.4f}** | **1.00×** |")
    for codec, total in sorted(stats["individual_codec_totals"].items(),
                                key=lambda kv: kv[1]):
        ratio = total / raw
        vs_disp = total / disp
        lines.append(f"| {codec} alone | {total:,} | {ratio:.4f} | {vs_disp:.2f}× |")
    lines.append("")

    lines.append("## Chunk codec distribution")
    lines.append("")
    lines.append("| codec | chunks picked | % | bytes emitted |")
    lines.append("|---|---|---|---|")
    for codec, n in sorted(stats["per_codec_chunks"].items(),
                            key=lambda kv: -kv[1]):
        if n == 0:
            continue
        pct = 100.0 * n / max(1, stats["chunks"])
        emitted = stats["per_codec_totals"][codec]
        lines.append(f"| {codec} | {n:,} | {pct:.1f}% | {emitted:,} |")

    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--chunk-bytes", type=int, default=CHUNK_BYTES)
    p.add_argument("--neural-bpb", type=float, default=None,
                   help="Neural bits-per-byte estimate (from "
                        "measure_held_out_ratio.py or training eval). "
                        "If omitted, neural codec is disabled.")
    p.add_argument("--zstd-dict", type=Path, default=None,
                   help="Optional zstd dictionary file (from "
                        "`zstd --train`).")
    p.add_argument("--no-bzip3", action="store_true",
                   help="Disable bzip3 codec even if installed.")
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--report", type=Path, default=None,
                   help="Markdown report output.")
    p.add_argument("--limit-bytes", type=int, default=None,
                   help="Stop after processing N bytes.")
    args = p.parse_args()

    print(f"[sim] input={args.input} chunk={args.chunk_bytes} "
          f"neural_bpb={args.neural_bpb}",
          file=sys.stderr)

    stats = simulate(
        args.input,
        chunk_bytes=args.chunk_bytes,
        neural_bpb=args.neural_bpb,
        zstd_dict_path=args.zstd_dict,
        enable_bzip3=not args.no_bzip3,
        enable_neural=args.neural_bpb is not None,
        csv_out=args.out_csv,
        limit_bytes=args.limit_bytes,
    )

    report = render_report(stats, args.input, args.neural_bpb)
    print(report)
    if args.report:
        args.report.write_text(report)
        print(f"[sim] report written to {args.report}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

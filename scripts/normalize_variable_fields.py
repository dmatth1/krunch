"""Phase C4 preprocessor — replace variable log fields with fixed
placeholder tokens before SPM training.

Motivation: SPM's unigram trainer tries to find frequent substrings
as vocabulary pieces. On HDFS, that means it learns
`/user/root/sortrand2/_temporary/_task_200811101024_0003_r_00001`
as a piece — but val has `r_00042` or `r_00089` which SPM can't match,
so val fragments back to byte-fallback and token density drops.

By normalizing varying fields (block IDs, timestamps, IPs, task IDs,
PIDs) to fixed `<BLKID>`, `<TS>`, `<IP>`, `<TASKID>`, `<PID>` strings
BEFORE SPM training, the remaining "skeleton" of each log line is
invariant train-vs-val. SPM learns it cleanly. The actual variable
values become predictable-from-structure in the model.

For Spike 2 we do this AS an input transformation. For production,
the Rust runtime would need to apply the same normalization at
encode time and REVERSE it at decode time using a side channel of
"the actual variable values that were matched" — which itself is
compressible via the model (that's the whole point).

Patterns (HDFS v1 specific for Spike 2; generalize later):
    \\d{6} \\d{6}               -> <TS>       (`081109 203518`)
    blk_-?\\d+                  -> <BLKID>
    \\d+\\.\\d+\\.\\d+\\.\\d+:\\d+  -> <IP>
    _task_\\d+_\\d+_[rm]_\\d+   -> <TASKID>   (Hadoop M/R task ids)
    \\b\\d{2,5}\\b              -> <NUM>      (generic 2-5 digit numbers: pid, port)

Emits BOTH a normalized corpus (written to --out) and a simple
side-channel of "original variable field values per line" (written
to --side-channel), one JSON record per line.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Patterns applied in this order. Order matters — more specific first
# so that e.g. block ID digits aren't caught by the generic <NUM>.
PATTERNS = [
    ("TS",      re.compile(r"\b\d{6} \d{6}\b")),
    ("BLKID",   re.compile(r"blk_-?\d+")),
    ("IP",      re.compile(r"\d+\.\d+\.\d+\.\d+(?::\d+)?")),
    ("TASKID",  re.compile(r"_task_\d+_\d+_[rm]_\d+")),
    ("JOBID",   re.compile(r"_job_\d+_\d+|job_\d+_\d+")),
    ("HEX",     re.compile(r"0x[0-9a-f]{4,}", re.IGNORECASE)),
    ("NUM",     re.compile(r"\b\d{2,}\b")),  # last resort: any digit run >= 2
]


def normalize_line(line: str) -> tuple[str, list[tuple[str, str]]]:
    """Replace variable fields with placeholders. Returns the
    normalized line plus an ordered list of (placeholder_name, original)
    tuples for the side channel so the transform is reversible."""
    captures: list[tuple[str, str]] = []
    normed = line

    for name, pat in PATTERNS:
        placeholder = f"<{name}>"

        def _repl(m, _name=name):
            captures.append((_name, m.group(0)))
            return placeholder

        normed = pat.sub(_repl, normed)

    return normed, captures


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True,
                   help="Raw NDJSON / text corpus to normalize.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output: normalized corpus (one line per input line).")
    p.add_argument("--side-channel", type=Path, default=None,
                   help="Optional: per-line JSON of captured variable "
                        "values, in order.")
    p.add_argument("--limit-lines", type=int, default=0,
                   help="If >0, stop after N lines (for sanity checks).")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.side_channel:
        args.side_channel.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    total_bytes_in = 0
    total_bytes_out = 0
    total_captures = 0

    out_f = args.out.open("w", encoding="utf-8")
    side_f = args.side_channel.open("w", encoding="utf-8") if args.side_channel else None
    try:
        with args.corpus.open("r", encoding="utf-8", errors="replace") as fin:
            for line in fin:
                normed, caps = normalize_line(line)
                out_f.write(normed)
                if side_f is not None:
                    side_f.write(json.dumps(caps, separators=(",", ":")) + "\n")
                total_lines += 1
                total_bytes_in += len(line)
                total_bytes_out += len(normed)
                total_captures += len(caps)
                if args.limit_lines and total_lines >= args.limit_lines:
                    break
    finally:
        out_f.close()
        if side_f is not None:
            side_f.close()

    print(
        f"normalized {total_lines:,} lines "
        f"({total_bytes_in:,} -> {total_bytes_out:,} bytes, "
        f"{total_captures:,} captured fields, "
        f"avg {total_captures/max(1,total_lines):.2f} placeholders/line)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

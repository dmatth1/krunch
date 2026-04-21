"""Reference-corpus zstd-22 baselines for the v0.1.0 release claim.

Our primary bar (bench/v010_bt_and_zstd22.txt) is measured on
bench/v010_benchmark/. That's fair for our methodology but not a
canonical industry reference — nobody has zstd-on-our-corpus published
numbers for cross-validation.

This script captures zstd-22 numbers on two canonical reference corpora:

  1. **Silesia corpus** (2003, the zstd paper's own benchmark). The text /
     structured-text subset only — we skip mozilla/ooffice (binary DLLs),
     sao (star catalog, near-random), x-ray (medical). Keeps: dickens
     (English prose), reymont (Polish prose), webster (dictionary text),
     xml (structured), samba (C source), osdb (tabular/database), mr
     (medical records text), nci (chemical formulas).

  2. **enwik8** (100 MB pure English Wikipedia). The Hutter Prize
     benchmark used by every NN compression paper (NNCP, CMIX, ts_zip,
     Nacrith, L3TC). Lets us publish a number that's directly
     comparable to published neural-compressor literature.

All runs use `zstd --long=27 --ultra -22` for parity with the v010
measurement. Output is written alongside existing bench artifacts.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SILESIA = REPO / "bench" / "corpora" / "silesia"
ENWIK8 = REPO / "bench" / "corpora" / "enwik8"

# Silesia files, grouped. We report text-and-structured separately from
# binary since no neural text model targets binary.
SILESIA_TEXT = ["dickens", "reymont", "webster", "mr", "nci"]
SILESIA_STRUCT = ["xml", "samba", "osdb"]
SILESIA_BINARY = ["mozilla", "ooffice", "sao", "x-ray"]


@dataclass
class Row:
    corpus: str
    raw_bytes: int
    zstd22_bytes: int
    ratio: float


def zstd22(data: bytes) -> int:
    proc = subprocess.run(
        ["zstd", "--long=27", "--ultra", "-22", "--stdout", "--quiet"],
        input=data, capture_output=True, check=True,
    )
    return len(proc.stdout)


def measure_file(path: Path, label: str) -> Row:
    data = path.read_bytes()
    c = zstd22(data)
    return Row(corpus=label, raw_bytes=len(data),
               zstd22_bytes=c, ratio=c / len(data))


def main() -> None:
    print("=" * 70)
    print("reference-corpus zstd-22 baselines")
    print("  all runs: zstd --long=27 --ultra -22")
    print("=" * 70)
    print()

    rows: list[Row] = []

    # --- enwik8 (100 MB prose, canonical NN compressor benchmark) ---
    print("--- enwik8 (100 MB, pure English Wikipedia) ---")
    if not ENWIK8.is_file():
        print(f"  missing: {ENWIK8}")
    else:
        r = measure_file(ENWIK8, "enwik8")
        rows.append(r)
        print(f"  raw: {r.raw_bytes/1e6:.1f} MB   "
              f"zstd-22: {r.zstd22_bytes/1e6:.3f} MB   "
              f"ratio: {r.ratio:.4f}")
    print()

    # --- Silesia by group ---
    print("--- Silesia (zstd's own reference benchmark) ---")
    silesia_text_rows: list[Row] = []
    silesia_struct_rows: list[Row] = []
    silesia_binary_rows: list[Row] = []

    def measure_group(name: str, files: list[str]) -> list[Row]:
        grp: list[Row] = []
        for f in files:
            p = SILESIA / f
            if not p.is_file():
                print(f"  [{name}] missing: {p}")
                continue
            r = measure_file(p, f"silesia/{f}")
            grp.append(r)
            print(f"  [{name}] {f:<10s}  raw {r.raw_bytes/1e6:6.2f} MB   "
                  f"ratio {r.ratio:.4f}")
        return grp

    silesia_text_rows = measure_group("text", SILESIA_TEXT)
    silesia_struct_rows = measure_group("struct", SILESIA_STRUCT)
    silesia_binary_rows = measure_group("binary", SILESIA_BINARY)

    rows.extend(silesia_text_rows + silesia_struct_rows + silesia_binary_rows)

    def agg(label: str, group: list[Row]) -> Row | None:
        if not group:
            return None
        raw = sum(g.raw_bytes for g in group)
        comp = sum(g.zstd22_bytes for g in group)
        return Row(corpus=label, raw_bytes=raw, zstd22_bytes=comp,
                   ratio=comp / raw if raw else 0.0)

    # Aggregate Silesia groups
    print()
    print("--- Silesia aggregates (concatenated per group) ---")
    for label, grp in [
        ("silesia_text (dickens+reymont+webster+mr+nci)", silesia_text_rows),
        ("silesia_struct (xml+samba+osdb)", silesia_struct_rows),
        ("silesia_binary (mozilla+ooffice+sao+x-ray)", silesia_binary_rows),
        ("silesia_text+struct (l3tc target scope)",
         silesia_text_rows + silesia_struct_rows),
        ("silesia_full (all 12 files)",
         silesia_text_rows + silesia_struct_rows + silesia_binary_rows),
    ]:
        a = agg(label, grp)
        if a:
            rows.append(a)
            print(f"  {label:<55s}  raw {a.raw_bytes/1e6:6.1f} MB   "
                  f"ratio {a.ratio:.4f}")

    print()
    print("=" * 70)
    print("takeaways")
    print("=" * 70)
    print()
    print("* enwik8 is where every NN compressor paper publishes.")
    print("* silesia_text+struct is the 'fair' zstd comparison for l3tc's")
    print("  text-and-structured-text product scope (skips binary files).")
    print("* silesia_full is zstd's own published headline aggregate.")
    print()

    # Persist JSON for harness consumption.
    out_json = REPO / "bench" / "zstd_reference.json"
    out_json.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    print(f"saved {out_json.relative_to(REPO)}")


if __name__ == "__main__":
    main()

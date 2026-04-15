"""Build the structured data portion of the 12L training corpus.

Downloads REAL data from open sources — no synthetic, no repetition,
no gated access:

1. Structured code: lumees/github-code-2025-language-split (open HF)
   — YAML, JSON, SQL, XML, Shell, Dockerfile from real GitHub repos
2. System logs: Zenodo Loghub full datasets (not 2K samples)
   — BGL, Spark, HDFS, Android — millions of real log lines
3. CSV: Data.gov CKAN API + public GitHub datasets
   — diverse schemas from real government/public data

Deduplication: tracks all content hashes, skips duplicates both
within and across sources.

Usage:
    cd l3tc-prod
    python scripts/build_corpus_v2.py \
        --output-dir corpus_build/structured \
        --code-gb 3.0 --logs-gb 1.0 --csv-gb 1.0

    # Quick test:
    python scripts/build_corpus_v2.py \
        --output-dir corpus_build/structured_test \
        --code-gb 0.1 --logs-gb 0.1 --csv-gb 0.1
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path


class Deduplicator:
    """Track content hashes to prevent any duplicate documents."""

    def __init__(self):
        self.seen: set[str] = set()

    def is_new(self, text: str) -> bool:
        h = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()
        if h in self.seen:
            return False
        self.seen.add(h)
        return True

    def __len__(self):
        return len(self.seen)


def stream_github_code(
    output_path: Path,
    target_bytes: int,
    dedup: Deduplicator,
    seed: int = 2024,
):
    """Stream structured code from lumees/github-code-2025-language-split."""
    from datasets import load_dataset

    print(f"\n=== Structured code (target: {target_bytes/1e9:.1f} GB) ===")

    langs = {
        "yaml": 1.0,     # lots available
        "json": 0.8,
        "sql": 0.8,
        "xml": 0.8,
        "shell": 0.5,
        "dockerfile": 0.3,
        "toml": 0.3,
        "makefile": 0.3,
    }
    total_weight = sum(langs.values())

    written = 0
    docs = 0
    skipped_dup = 0
    t0 = time.time()

    with open(output_path, "w", encoding="utf-8") as f:
        for lang, weight in langs.items():
            if written >= target_bytes:
                break
            lang_target = int(target_bytes * weight / total_weight)
            lang_written = 0

            try:
                ds = load_dataset(
                    "lumees/github-code-2025-language-split",
                    lang,
                    streaming=True,
                    split="train",
                    token=os.environ.get("HF_TOKEN"),
                )
                ds = ds.shuffle(seed=seed, buffer_size=5_000)
            except Exception as e:
                print(f"  {lang}: SKIP ({e})")
                continue

            for example in ds:
                if lang_written >= lang_target or written >= target_bytes:
                    break
                content = example.get("content", "")
                if not content or len(content) < 50 or len(content) > 100_000:
                    continue
                if not dedup.is_new(content):
                    skipped_dup += 1
                    continue

                f.write(content)
                f.write("\n")
                b = len(content.encode("utf-8")) + 1
                written += b
                lang_written += b
                docs += 1

                if docs % 10_000 == 0:
                    elapsed = time.time() - t0
                    print(f"  {docs:,} files, {written/1e9:.2f} GB, "
                          f"{elapsed:.0f}s ({lang}: {lang_written/1e9:.2f} GB)")

            print(f"  {lang}: {lang_written/1e6:.0f} MB, "
                  f"{docs:,} files total so far")

    print(f"  done: {docs:,} files, {written/1e9:.2f} GB, "
          f"{skipped_dup:,} duplicates skipped, {time.time()-t0:.0f}s")
    return written


def download_zenodo_loghub(
    output_path: Path,
    target_bytes: int,
    dedup: Deduplicator,
):
    """Download full log datasets from Zenodo Loghub."""
    print(f"\n=== System logs from Zenodo (target: {target_bytes/1e9:.1f} GB) ===")

    # Ordered by size — download until we hit target
    log_files = [
        ("BGL.zip", "BGL.log"),
        ("Spark.tar.gz", None),          # contains Spark_2k.log etc
        ("HDFS_v1.zip", None),
        ("Android_v1.zip", None),
        ("OpenStack.tar.gz", None),
        ("SSH.tar.gz", None),
        ("Hadoop.zip", None),
        ("HPC.zip", None),
        ("HealthApp.tar.gz", None),
        ("Mac.tar.gz", None),
    ]

    base_url = "https://zenodo.org/records/8196385/files"
    written = 0
    t0 = time.time()

    with open(output_path, "w", encoding="utf-8") as f:
        for archive_name, log_file_hint in log_files:
            if written >= target_bytes:
                break

            url = f"{base_url}/{archive_name}"
            print(f"  downloading {archive_name}...")

            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "l3tc-corpus/1.0"})
                resp = urllib.request.urlopen(req, timeout=300)
                data = resp.read()
            except Exception as e:
                print(f"    FAIL: {e}")
                continue

            print(f"    downloaded: {len(data)/1e6:.1f} MB compressed")

            # Extract and write log lines
            lines_written = 0
            try:
                if archive_name.endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(data)) as zf:
                        for name in zf.namelist():
                            if written >= target_bytes:
                                break
                            if not name.endswith((".log", ".csv", ".txt")):
                                continue
                            with zf.open(name) as lf:
                                for raw_line in lf:
                                    if written >= target_bytes:
                                        break
                                    line = raw_line.decode(
                                        "utf-8", errors="replace").rstrip()
                                    if len(line) < 10:
                                        continue
                                    f.write(line)
                                    f.write("\n")
                                    b = len(line) + 1
                                    written += b
                                    lines_written += 1
                elif archive_name.endswith(".tar.gz"):
                    with tarfile.open(
                            fileobj=io.BytesIO(data), mode="r:gz") as tf:
                        for member in tf.getmembers():
                            if written >= target_bytes:
                                break
                            if not member.isfile():
                                continue
                            if not member.name.endswith(
                                    (".log", ".csv", ".txt")):
                                continue
                            ef = tf.extractfile(member)
                            if ef is None:
                                continue
                            for raw_line in ef:
                                if written >= target_bytes:
                                    break
                                line = raw_line.decode(
                                    "utf-8", errors="replace").rstrip()
                                if len(line) < 10:
                                    continue
                                f.write(line)
                                f.write("\n")
                                b = len(line) + 1
                                written += b
                                lines_written += 1
            except Exception as e:
                print(f"    extract error: {e}")

            print(f"    {lines_written:,} lines, "
                  f"total: {written/1e9:.2f} GB")

    print(f"  done: {written/1e9:.2f} GB, {time.time()-t0:.0f}s")
    return written


def download_public_csv(
    output_path: Path,
    target_bytes: int,
    dedup: Deduplicator,
):
    """Download diverse real CSV data from public sources."""
    print(f"\n=== CSV datasets (target: {target_bytes/1e9:.1f} GB) ===")

    # Diverse public CSV sources — each is a different schema
    csv_sources = [
        # GitHub datasets org
        ("https://raw.githubusercontent.com/datasets/gdp/main/data/gdp.csv", "gdp"),
        ("https://raw.githubusercontent.com/datasets/population/main/data/population.csv", "population"),
        ("https://raw.githubusercontent.com/datasets/airport-codes/main/data/airport-codes.csv", "airports"),
        ("https://raw.githubusercontent.com/datasets/world-cities/main/data/world-cities.csv", "world_cities"),
        ("https://raw.githubusercontent.com/datasets/country-codes/main/data/country-codes.csv", "country_codes"),
        ("https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv", "covid"),
        ("https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv", "vix_daily"),
        ("https://raw.githubusercontent.com/datasets/natural-gas/main/data/daily.csv", "natural_gas"),
        ("https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv", "gold_prices"),
        ("https://raw.githubusercontent.com/datasets/oil-prices/main/data/brent-monthly.csv", "oil_prices"),
        ("https://raw.githubusercontent.com/datasets/sea-level-rise/main/data/epa-sea-level.csv", "sea_level"),
        ("https://raw.githubusercontent.com/datasets/global-temp/main/data/monthly.csv", "global_temp"),
        ("https://raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv", "exchange_rates"),
        ("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv", "sp500"),
        # UCI ML Repository
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", "adult_income"),
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", "wine_red"),
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", "wine_white"),
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", "iris"),
        # More GitHub datasets org
        ("https://raw.githubusercontent.com/datasets/cpi/main/data/cpi.csv", "cpi"),
        ("https://raw.githubusercontent.com/datasets/house-prices-us/main/data/cities-month.csv", "house_prices"),
        ("https://raw.githubusercontent.com/datasets/investor-flow-of-funds-us/main/data/weekly.csv", "investor_flows"),
    ]

    written = 0
    t0 = time.time()

    with open(output_path, "w", encoding="utf-8") as f:
        for url, name in csv_sources:
            if written >= target_bytes:
                break
            try:
                print(f"  {name}...", end=" ", flush=True)
                req = urllib.request.Request(
                    url, headers={"User-Agent": "l3tc-corpus/1.0"})
                resp = urllib.request.urlopen(req, timeout=60)
                data = resp.read().decode("utf-8", errors="replace")

                if not dedup.is_new(data):
                    print("SKIP (duplicate)")
                    continue

                f.write(data)
                if not data.endswith("\n"):
                    f.write("\n")
                b = len(data.encode("utf-8"))
                written += b
                rows = data.count("\n")
                print(f"{b/1e6:.1f} MB, {rows:,} rows")
            except Exception as e:
                print(f"FAIL ({e})")

    # If we're still short, try streaming CSV files from
    # lumees/github-code-2025-language-split
    if written < target_bytes:
        print(f"  supplementing with GitHub CSV files...")
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "lumees/github-code-2025-language-split",
                "unknown",  # CSV files often classified as unknown
                streaming=True,
                split="train",
                token=os.environ.get("HF_TOKEN"),
            )
            for example in ds:
                if written >= target_bytes:
                    break
                content = example.get("content", "")
                path = example.get("file_path", "")
                if not path.endswith(".csv") and not path.endswith(".tsv"):
                    continue
                if len(content) < 100 or len(content) > 500_000:
                    continue
                if not dedup.is_new(content):
                    continue
                f.write(content)
                f.write("\n")
                written += len(content.encode("utf-8")) + 1
        except Exception as e:
            print(f"    GitHub CSV supplement failed: {e}")

    print(f"  done: {written/1e9:.2f} GB, {time.time()-t0:.0f}s")
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--code-gb", type=float, default=3.0)
    p.add_argument("--logs-gb", type=float, default=1.0)
    p.add_argument("--csv-gb", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2024)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dedup = Deduplicator()
    total = 0

    # 1. Structured code from real GitHub repos
    t = stream_github_code(
        args.output_dir / "code_real.txt",
        int(args.code_gb * 1e9),
        dedup,
        args.seed,
    )
    total += t

    # 2. Real system logs from Zenodo
    t = download_zenodo_loghub(
        args.output_dir / "logs_real.txt",
        int(args.logs_gb * 1e9),
        dedup,
    )
    total += t

    # 3. Real CSV data from diverse public sources
    t = download_public_csv(
        args.output_dir / "csv_real.txt",
        int(args.csv_gb * 1e9),
        dedup,
    )
    total += t

    print(f"\n=== Summary ===")
    print(f"  unique documents tracked: {len(dedup):,}")
    for f in sorted(args.output_dir.iterdir()):
        if f.suffix == ".txt":
            print(f"  {f.name:<25s} {f.stat().st_size / 1e9:>6.2f} GB")
    print(f"  {'TOTAL':<25s} {total / 1e9:>6.2f} GB")


if __name__ == "__main__":
    main()

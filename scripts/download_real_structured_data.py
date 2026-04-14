"""Download real structured text data for compressor training.

Pulls REAL data from public sources — no synthetic, no gated access:
  1. Loghub (GitHub): real system logs from 16 systems
  2. Public CSV/TSV datasets from GitHub datasets org + UCI
  3. Pile GitHub subset: real code files (YAML, SQL, JSON, XML)
     filtered by content patterns from the Pile stream

All downloads use direct URLs (no HF gated access needed).
The Pile subset uses the already-accessible Pile dedup stream.

Usage:
    export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
    python scripts/download_real_structured_data.py \
        --output-dir corpus_build/real \
        --pile-github-gb 3.0

    # Quick test:
    python scripts/download_real_structured_data.py \
        --output-dir corpus_build/real_test \
        --pile-github-gb 0.1 --skip-pile
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path


def download_url(url: str, timeout: int = 30) -> str:
    """Download a URL with proper headers and SSL."""
    req = urllib.request.Request(url, headers={"User-Agent": "l3tc-corpus/1.0"})
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", errors="replace")


def download_loghub(output_path: Path):
    """Download real system logs from logpai/loghub on GitHub."""
    print("=== Loghub system logs ===")

    # All available log types with their 2k-line sample files
    log_types = [
        "Apache", "BGL", "HDFS", "HPC", "Linux", "Mac",
        "OpenSSH", "OpenStack", "Spark", "Thunderbird",
        "Windows", "Zookeeper", "Hadoop", "HealthApp",
        "Android", "Proxifier",
    ]

    base_url = "https://raw.githubusercontent.com/logpai/loghub/master"
    total = 0

    with open(output_path, "w") as f:
        for log_type in log_types:
            url = f"{base_url}/{log_type}/{log_type}_2k.log"
            try:
                data = download_url(url)
                f.write(f"# === {log_type} system logs ===\n")
                f.write(data)
                f.write("\n")
                total += len(data)
                lines = data.count("\n")
                print(f"  {log_type}: {len(data)/1e3:.0f} KB, {lines} lines")
            except Exception as e:
                print(f"  {log_type}: SKIP ({e})")

    print(f"  total: {total/1e6:.1f} MB")
    return total


def download_public_csv(output_path: Path):
    """Download real CSV datasets from public GitHub repos."""
    print("\n=== Public CSV datasets ===")

    csv_sources = [
        # GitHub datasets org (well-maintained public datasets)
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
        # UCI ML Repository (direct CSV)
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", "iris"),
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", "wine_red"),
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", "wine_white"),
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", "adult_income"),
    ]

    total = 0
    with open(output_path, "w") as f:
        for url, name in csv_sources:
            try:
                data = download_url(url)
                f.write(f"# Dataset: {name}\n")
                f.write(data)
                f.write("\n\n")
                total += len(data)
                rows = data.count("\n")
                print(f"  {name}: {len(data)/1e3:.0f} KB, {rows} rows")
            except Exception as e:
                print(f"  {name}: SKIP ({e})")

    print(f"  total: {total/1e6:.1f} MB")
    return total


def filter_pile_github(output_path: Path, target_bytes: int, seed: int = 1204):
    """Stream the Pile and filter for GitHub code files.

    The Pile's GitHub subset contains real YAML, SQL, JSON, XML,
    Dockerfiles, Makefiles, shell scripts, etc. from actual repos.
    We filter by content patterns to get structured text specifically.
    """
    from datasets import load_dataset

    print(f"\n=== Pile GitHub subset (target: {target_bytes/1e9:.1f} GB) ===")

    ds = load_dataset(
        "EleutherAI/the_pile_deduplicated",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    # Content-based filters for structured text types
    def is_structured(text: str) -> str | None:
        """Returns the detected type or None."""
        head = text[:500]
        # YAML
        if (head.startswith("apiVersion:") or head.startswith("kind:")
                or head.startswith("version:") and "services:" in text[:2000]
                or head.startswith("name:") and "on:" in text[:500]
                or "- name:" in head and ":" in head):
            return "yaml"
        # JSON
        if head.lstrip().startswith("{") and '"' in head[:50]:
            return "json"
        # SQL
        if any(kw in head[:300].upper() for kw in
               ["CREATE TABLE", "INSERT INTO", "ALTER TABLE", "DROP TABLE",
                "SELECT ", "CREATE INDEX", "CREATE DATABASE"]):
            return "sql"
        # XML/HTML
        if head.lstrip().startswith("<?xml") or head.lstrip().startswith("<!DOCTYPE"):
            return "xml"
        if head.lstrip().startswith("<") and ">" in head[:100]:
            return "xml"
        # Dockerfile
        if head.startswith("FROM ") and ("RUN " in text[:1000] or "CMD " in text[:1000]):
            return "dockerfile"
        # Makefile
        if head.startswith(".PHONY") or ("\t" in head and ":" in head[:100]):
            return "makefile"
        # Shell script
        if head.startswith("#!/bin/") or head.startswith("#!/usr/bin/env"):
            return "shell"
        # CSV-like (header + rows with consistent delimiters)
        lines = text[:1000].split("\n")
        if len(lines) > 3:
            first = lines[0]
            if first.count(",") > 2 and all(
                abs(l.count(",") - first.count(",")) <= 1
                for l in lines[1:4] if l.strip()
            ):
                return "csv"
        return None

    written = 0
    counts: dict[str, int] = {}
    docs = 0
    t0 = time.time()

    with open(output_path, "w", encoding="utf-8") as f:
        for example in ds:
            if written >= target_bytes:
                break

            meta = example.get("meta", {})
            pile_set = meta.get("pile_set_name", "") if isinstance(meta, dict) else ""

            # Accept GitHub code OR any doc that matches structured patterns
            text = example.get("text", "")
            if not text or len(text) < 100:
                continue

            if pile_set == "Github":
                stype = is_structured(text) or "other_code"
            else:
                stype = is_structured(text)
                if stype is None:
                    continue  # skip non-structured, non-GitHub docs

            # Skip very large files (>50 KB) — often generated/minified
            if len(text) > 50_000:
                continue

            f.write(text)
            f.write("\n")
            b = len(text.encode("utf-8")) + 1
            written += b
            counts[stype] = counts.get(stype, 0) + 1
            docs += 1

            if docs % 5_000 == 0:
                elapsed = time.time() - t0
                print(f"  {docs:,} files, {written/1e9:.2f} GB, {elapsed:.0f}s")
                print(f"    types: {dict(sorted(counts.items(), key=lambda x: -x[1]))}")

    elapsed = time.time() - t0
    print(f"  done: {docs:,} files, {written/1e9:.2f} GB, {elapsed:.0f}s")
    print(f"  type breakdown:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v:,}")
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--pile-github-gb", type=float, default=3.0,
                   help="GB of structured code/config to extract from the Pile.")
    p.add_argument("--skip-pile", action="store_true",
                   help="Skip the slow Pile filtering step (just do logs + CSV).")
    p.add_argument("--seed", type=int, default=1204)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = 0

    # 1. Real system logs
    t = download_loghub(args.output_dir / "logs_real.txt")
    total += t

    # 2. Real CSV datasets
    t = download_public_csv(args.output_dir / "csv_real.txt")
    total += t

    # 3. Real code/configs from the Pile
    if not args.skip_pile:
        t = filter_pile_github(
            args.output_dir / "structured_code_real.txt",
            int(args.pile_github_gb * 1e9),
            args.seed,
        )
        total += t

    print(f"\n=== Summary ===")
    for f in sorted(args.output_dir.iterdir()):
        if f.suffix == ".txt":
            print(f"  {f.name:<30s} {f.stat().st_size / 1e6:>8.1f} MB")
    print(f"  {'TOTAL':<30s} {total / 1e6:>8.1f} MB")


if __name__ == "__main__":
    main()

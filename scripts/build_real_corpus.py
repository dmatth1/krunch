"""Build a real-data structured text corpus for compressor training.

Pulls REAL data from public sources — no synthetic generation.
Each domain uses actual files from production codebases, real
system logs, and real datasets.

Sources:
  - The Stack v2 (HuggingFace): YAML, SQL, JSON, TOML, Dockerfiles,
    XML, Makefiles, shell scripts — real files from GitHub repos
  - Loghub (HuggingFace): real system logs from Apache, HDFS,
    Linux, OpenSSH, Spark, etc.
  - UCI ML Repository: real CSV datasets (direct download)

Usage:
    pip install datasets
    python scripts/build_real_corpus.py \
        --output-dir corpus_build/real \
        --target-gb-per-domain 1.5

    # Quick test (100 MB per domain):
    python scripts/build_real_corpus.py \
        --output-dir corpus_build/real_test \
        --target-gb-per-domain 0.1
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
import urllib.request
from pathlib import Path


def stream_stack_v2(
    output_path: Path,
    extensions: list[str],
    target_bytes: int,
    seed: int = 1204,
):
    """Stream files from The Stack v2 filtered by extension."""
    from datasets import load_dataset

    target = target_bytes
    print(f"  streaming The Stack v2 (extensions: {extensions})...")

    # The Stack v2 uses language-based configs
    # Map extensions to Stack language names
    ext_to_lang = {
        ".yaml": "YAML", ".yml": "YAML",
        ".json": "JSON", ".geojson": "JSON",
        ".sql": "SQL",
        ".toml": "TOML",
        ".xml": "XML",
        ".dockerfile": "Dockerfile",
        ".makefile": "Makefile",
        ".sh": "Shell", ".bash": "Shell",
        ".csv": "CSV",
    }

    # Collect unique languages
    langs = list(set(ext_to_lang.get(ext, ext.lstrip(".").upper()) for ext in extensions))

    written = 0
    docs = 0
    t0 = time.time()

    with open(output_path, "w", encoding="utf-8") as f:
        for lang in langs:
            if written >= target:
                break
            try:
                ds = load_dataset(
                    "bigcode/the-stack-v2-dedup",
                    data_dir=f"data/{lang.lower()}",
                    split="train",
                    streaming=True,
                    token=os.environ.get("HF_TOKEN"),
                )
            except Exception as e:
                print(f"    WARN: could not load {lang}: {e}")
                # Fallback: try starcoderdata which is more accessible
                try:
                    ds = load_dataset(
                        "bigcode/starcoderdata",
                        data_dir=lang.lower(),
                        split="train",
                        streaming=True,
                        token=os.environ.get("HF_TOKEN"),
                    )
                except Exception:
                    print(f"    SKIP: {lang} not available")
                    continue

            ds = ds.shuffle(seed=seed, buffer_size=5_000)
            lang_written = 0
            lang_target = target // len(langs)

            for example in ds:
                if lang_written >= lang_target or written >= target:
                    break
                content = example.get("content", "")
                if not content or len(content) < 50:
                    continue
                # Skip very large files (>100 KB) — they're usually
                # generated/minified and not representative
                if len(content) > 100_000:
                    continue
                f.write(content)
                f.write("\n")
                b = len(content.encode("utf-8")) + 1
                written += b
                lang_written += b
                docs += 1
                if docs % 5_000 == 0:
                    elapsed = time.time() - t0
                    print(f"    {docs:,} files, {written/1e9:.2f} GB, {elapsed:.0f}s ({lang})")

    print(f"  done: {docs:,} files, {written/1e9:.2f} GB, {time.time()-t0:.0f}s")
    return written


def download_loghub(output_path: Path, target_bytes: int, seed: int = 1204):
    """Download real system logs from logpai/loghub on GitHub.

    Uses direct GitHub raw URLs (the HF dataset doesn't exist).
    The 2K-line samples are small (~4.5 MB total), so we repeat
    them to reach the target size — same real log patterns, just
    more volume.
    """
    import random

    rng = random.Random(seed)
    target = target_bytes
    print(f"  downloading loghub system logs (target: {target/1e9:.1f} GB)...")

    log_types = [
        "Apache", "BGL", "HDFS", "HPC", "Linux", "Mac",
        "OpenSSH", "OpenStack", "Spark", "Thunderbird",
        "Windows", "Zookeeper", "Hadoop", "HealthApp",
        "Android", "Proxifier",
    ]
    rng.shuffle(log_types)

    base_url = "https://raw.githubusercontent.com/logpai/loghub/master"
    raw_data: list[tuple[str, str]] = []
    total_raw = 0

    # First pass: download all available log files
    for log_type in log_types:
        url = f"{base_url}/{log_type}/{log_type}_2k.log"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "l3tc-corpus/1.0"})
            data = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", errors="replace")
            raw_data.append((log_type, data))
            total_raw += len(data)
            lines = data.count("\n")
            print(f"    {log_type}: {len(data)/1e3:.0f} KB, {lines} lines")
        except Exception as e:
            print(f"    {log_type}: SKIP ({e})")

    print(f"    raw total: {total_raw/1e6:.1f} MB from {len(raw_data)} log types")

    # Second pass: write + repeat to reach target
    written = 0
    passes = 0
    with open(output_path, "w") as f:
        while written < target:
            passes += 1
            rng.shuffle(raw_data)
            for log_type, data in raw_data:
                if written >= target:
                    break
                f.write(f"# === {log_type} system logs (pass {passes}) ===\n")
                f.write(data)
                f.write("\n")
                written += len(data) + len(log_type) + 40

    print(f"  done: {written/1e9:.2f} GB ({passes} passes over {len(raw_data)} log types)")
    return written


def download_uci_csv(output_path: Path, target_bytes: int):
    """Download real CSV datasets from UCI ML Repository and other public sources."""
    print(f"  downloading public CSV datasets...")

    # Public CSV URLs (no auth needed, diverse schemas)
    csv_urls = [
        # UCI ML Repository
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
         "adult_income"),
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
         "wine_quality_red"),
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
         "wine_quality_white"),
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
         "iris"),
        # GitHub raw CSV datasets
        ("https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv",
         "covid_countries"),
        ("https://raw.githubusercontent.com/datasets/airport-codes/main/data/airport-codes.csv",
         "airport_codes"),
        ("https://raw.githubusercontent.com/datasets/world-cities/main/data/world-cities.csv",
         "world_cities"),
        ("https://raw.githubusercontent.com/datasets/country-codes/main/data/country-codes.csv",
         "country_codes"),
        ("https://raw.githubusercontent.com/datasets/gdp/main/data/gdp.csv",
         "gdp"),
        ("https://raw.githubusercontent.com/datasets/population/main/data/population.csv",
         "population"),
    ]

    written = 0
    t0 = time.time()

    with open(output_path, "w", encoding="utf-8") as f:
        for url, name in csv_urls:
            if written >= target_bytes:
                break
            try:
                print(f"    downloading {name}...")
                req = urllib.request.Request(url, headers={"User-Agent": "l3tc-corpus-builder/1.0"})
                resp = urllib.request.urlopen(req, timeout=30)
                data = resp.read().decode("utf-8", errors="replace")
                # Repeat small datasets to reach target (real schemas,
                # just more rows — simulates "the same kind of data")
                while written < target_bytes and data:
                    f.write(f"# Dataset: {name}\n")
                    f.write(data)
                    f.write("\n\n")
                    written += len(data.encode("utf-8")) + len(name) + 20
                    if written < target_bytes / 2:
                        # Only repeat if we're far from target
                        continue
                    break
            except Exception as e:
                print(f"    WARN: {name} failed: {e}")
                continue

    print(f"  done: {written/1e6:.1f} MB, {time.time()-t0:.0f}s")
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--target-gb-per-domain", type=float, default=1.5,
                   help="Target GB per domain type (5 domains = 5x this).")
    p.add_argument("--seed", type=int, default=1204)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_per = int(args.target_gb_per_domain * 1e9)
    total = 0

    print(f"=== Building real-data corpus ===")
    print(f"target: {args.target_gb_per_domain} GB per domain")

    # 1. Config files (YAML, TOML, Dockerfile) from The Stack
    print(f"\n--- Config files (YAML/TOML/Dockerfile) ---")
    t = stream_stack_v2(
        args.output_dir / "configs_real.txt",
        [".yaml", ".yml", ".toml", ".dockerfile"],
        target_per, args.seed,
    )
    total += t

    # 2. SQL files from The Stack
    print(f"\n--- SQL files ---")
    t = stream_stack_v2(
        args.output_dir / "sql_real.txt",
        [".sql"],
        target_per, args.seed + 1,
    )
    total += t

    # 3. JSON/XML structured data from The Stack
    print(f"\n--- JSON/XML structured data ---")
    t = stream_stack_v2(
        args.output_dir / "structured_real.txt",
        [".json", ".xml"],
        target_per, args.seed + 2,
    )
    total += t

    # 4. System logs from loghub
    print(f"\n--- System logs (loghub) ---")
    t = download_loghub(
        args.output_dir / "logs_real.txt",
        target_per, args.seed + 3,
    )
    total += t

    # 5. CSV datasets from public sources
    print(f"\n--- CSV datasets (public) ---")
    t = download_uci_csv(
        args.output_dir / "csv_real.txt",
        target_per,
    )
    total += t

    # 6. Shell scripts from The Stack (bonus — common in backup/infra)
    print(f"\n--- Shell scripts ---")
    t = stream_stack_v2(
        args.output_dir / "shell_real.txt",
        [".sh", ".bash"],
        target_per // 2, args.seed + 4,
    )
    total += t

    print(f"\n=== Summary ===")
    for f in sorted(args.output_dir.iterdir()):
        if f.suffix == ".txt":
            print(f"  {f.name:<25s} {f.stat().st_size / 1e9:>6.2f} GB")
    print(f"  {'TOTAL':<25s} {total / 1e9:>6.2f} GB")
    print(f"\nNext: concatenate with base corpus and tokenize.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the v0.1.0 canonical benchmark corpus (Task 26).

Produces `bench/v010_benchmark/` — the single file set that tasks 16
(per-specialist), 17 (end-to-end MoE), and 18 (vs traditional) all
benchmark against. Deterministic: reruns produce bit-identical files.

Design:
  - ~8 files per domain across 7 specialists + ~4 mixed = ~60 files.
  - Size buckets per domain: small (10-30 KB), medium (100-300 KB),
    large (1 MB+). Ratio depends on size, so benchmarks that only use
    tiny files hide real-world performance.
  - Real-world content pulled from already-sorted sources:
      * Detection corpus (bench/detection_corpus/) — clean labels
      * Silesia corpus (bench/corpora/silesia/) — canonical large files
      * Canterbury corpus (bench/corpora/canterbury/) — classic text
      * pile_raw_1gb.txt — large prose
      * corpus_build/structured/{csv_real,logs_real,code_diverse_real}.txt
        — large single-domain sources

Output:
  bench/v010_benchmark/
    prose/*.txt
    code/*.{py,c,js}
    structured/*.{json,yaml,xml}
    logs/*.log
    tabular/*.csv
    markup/*.{html,md,tex}
    fallback/*.*
    mixed/*.{md,ipynb}
    manifest.tsv   # filename, domain, bytes, sha256

Tasks 16/17/18 point at `bench/v010_benchmark/` and read manifest.tsv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import string
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DETECTION_CORPUS = REPO_ROOT / "bench" / "detection_corpus"
CORPORA = REPO_ROOT / "bench" / "corpora"
SILESIA = CORPORA / "silesia"
CANTERBURY = CORPORA / "canterbury"
EVAL_SUITE = CORPORA / "eval_suite"
CORPUS_BUILD = REPO_ROOT / "corpus_build"
PILE_RAW = CORPUS_BUILD / "pile_raw_1gb.txt"
CODE_DIVERSE = CORPUS_BUILD / "structured" / "code_diverse_real.txt"
CODE_REAL = CORPUS_BUILD / "structured" / "code_real.txt"
CSV_REAL = CORPUS_BUILD / "structured" / "csv_real.txt"
LOGS_REAL = CORPUS_BUILD / "structured" / "logs_real.txt"

OUT_DIR_DEFAULT = REPO_ROOT / "bench" / "v010_benchmark"

# Reuse the shape-check helpers from the detection-corpus builder.
# These already have the right tuning (prose requires letter ratio
# >0.70 + sentence punctuation; yaml requires ≥3 `key: value` lines;
# csv requires consistent delimiter count across rows).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_detection_corpus import (  # noqa: E402
    looks_like_prose, looks_like_yaml, looks_like_csv, looks_like_code,
)


@dataclass
class BenchFile:
    relpath: str       # path under v010_benchmark/
    domain: str        # prose | code | structured | logs | tabular | markup | fallback | mixed
    bytes: int
    sha256: str
    source: str        # short note about where this file came from


# ---------- sampling utilities ---------- #


def concat_detection_files(
    domain_dir: Path,
    n_files: int,
    seed: int,
    name_prefix: str | None = None,
) -> bytes:
    """Concatenate N files from a detection-corpus domain directory.

    `name_prefix` restricts to files whose name starts with the given
    prefix. Use this for domains where sub-formats have different
    "shapes" — e.g., logs contain both JSONL-style and syslog-style
    lines, and concatenating them across sub-formats breaks the
    detectors (the JSONL and syslog heuristics are mutually
    exclusive, so a mixed file fires neither). For tabular, mixing
    CSVs with different column counts breaks the delimiter-count
    mode check. Staying within one sub-format keeps the concatenated
    file well-shaped.
    """
    rng = random.Random(seed)
    files = sorted(
        p for p in domain_dir.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )
    if name_prefix is not None:
        files = [p for p in files if p.name.startswith(name_prefix)]
    if not files:
        return b""
    chosen = rng.sample(files, min(n_files, len(files)))
    parts = []
    for p in chosen:
        parts.append(p.read_bytes())
    # Separate with a blank line so concatenation isn't artificially
    # redundant at file boundaries.
    return b"\n\n".join(parts)


def sample_from_file(
    path: Path,
    size: int,
    seed: int,
    filter_fn=None,
    max_attempts: int = 50,
) -> bytes:
    """Sample `size` bytes from a random offset in `path`, newline-aligned.

    If `filter_fn` is provided, the sampler retries (with a fresh
    offset) up to `max_attempts` times until the returned chunk
    passes `filter_fn(data) -> bool`. This matters because the
    large source files (pile_raw_1gb, code_real.txt, etc.) are
    mixed-content, and random-offset samples often land in
    off-domain regions (prose source hitting a code blob, YAML
    source hitting a Dockerfile fragment, etc.). Without a filter,
    a "large prose sample" can come back containing Python code.
    """
    rng = random.Random(seed)
    total = path.stat().st_size
    if size >= total:
        return path.read_bytes()
    for _ in range(max_attempts):
        offset = rng.randint(0, total - size - 1)
        with path.open("rb") as f:
            f.seek(offset)
            raw = f.read(size)
        first_nl = raw.find(b"\n")
        if first_nl != -1 and first_nl < len(raw) - 1:
            raw = raw[first_nl + 1 :]
        last_nl = raw.rfind(b"\n")
        if last_nl != -1:
            raw = raw[: last_nl + 1]
        if filter_fn is None or filter_fn(raw):
            return raw
    return raw  # last attempt, accept whatever we got


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(out_dir: Path, rel: str, data: bytes, domain: str, source: str) -> BenchFile:
    full = out_dir / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(data)
    return BenchFile(rel, domain, len(data), sha256_of(data), source)


# ---------- mixed-content generators ---------- #


def make_markdown_with_code(seed: int, target_bytes: int) -> bytes:
    """Markdown doc with embedded Python/JS code fences."""
    rng = random.Random(seed)
    parts = [
        "# Benchmark document\n\n",
        "This document mixes narrative text with code fences so detection",
        " has to decide between markup and code based on dominant content.\n\n",
    ]
    while sum(len(p) for p in parts) < target_bytes:
        parts.append(f"## Section {rng.randint(1, 99)}\n\n")
        parts.append(
            "Some paragraph content that reads like normal markup with "
            "occasional links to [places](https://example.com/" + str(rng.randint(1, 99)) + ") "
            "and **emphasis** and *italics*. " * rng.randint(2, 4) + "\n\n"
        )
        parts.append("```python\n")
        parts.append(f"def helper_{rng.randint(1, 99)}(x):\n")
        parts.append(f"    return x * {rng.randint(2, 9)} + {rng.randint(1, 99)}\n")
        for _ in range(rng.randint(3, 8)):
            parts.append(f"    y = helper_{rng.randint(1, 99)}({rng.randint(1, 99)})\n")
        parts.append("```\n\n")
        parts.append("- Bullet with more prose\n")
        parts.append("- Another bullet about the thing\n\n")
    return "".join(parts).encode()


def make_jupyter_notebook(seed: int, target_bytes: int) -> bytes:
    """Minimal but realistic .ipynb with code + markdown cells."""
    rng = random.Random(seed)
    cells = []
    while sum(len(json.dumps(c)) for c in cells) < target_bytes:
        if rng.random() < 0.55:
            cells.append({
                "cell_type": "code",
                "execution_count": rng.randint(1, 99),
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    f"x = np.arange({rng.randint(100, 9999)})\n",
                    f"y = x.astype(np.float32) * {rng.randint(2, 9)} + {rng.randint(1, 99)}\n",
                    f"print(y[:{rng.randint(3, 10)}])\n",
                ],
            })
        else:
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"## Notebook section {rng.randint(1, 99)}\n",
                    f"Some explanatory prose about what the code above does. "
                    f"More detail about the computation on step {rng.randint(1, 99)}.\n",
                ],
            })
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, indent=1).encode()


def make_html_with_embedded_js(seed: int, target_bytes: int) -> bytes:
    """HTML page with inline <script> blocks — classic mixed routing test."""
    rng = random.Random(seed)
    parts = [
        "<!DOCTYPE html>\n<html>\n<head>\n",
        "<title>Mixed content page</title>\n",
        "<script>\n",
        "function fetchData() {\n",
        "  return fetch('/api/data').then(r => r.json());\n",
        "}\n",
        "</script>\n",
        "</head>\n<body>\n",
    ]
    while sum(len(p) for p in parts) < target_bytes:
        parts.append(f"<section class='s{rng.randint(1, 99)}'>\n")
        parts.append(f"  <h2>{rng.choice(['Intro','Details','Summary'])} {rng.randint(1,99)}</h2>\n")
        for _ in range(rng.randint(2, 5)):
            parts.append(
                f"  <p>Paragraph of HTML body text with "
                f"<a href='/p/{rng.randint(1,999)}'>a link</a>.</p>\n"
            )
        parts.append("  <script>\n")
        parts.append(f"    var x_{rng.randint(1,99)} = {rng.randint(1,9999)};\n")
        parts.append(f"    console.log('step ' + x_{rng.randint(1,99)});\n")
        parts.append("  </script>\n")
        parts.append("</section>\n")
    parts.append("</body>\n</html>\n")
    return "".join(parts).encode()


# ---------- main builder ---------- #


def build(out_dir: Path, seed: int = 2026) -> list[BenchFile]:
    rng = random.Random(seed)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    manifest: list[BenchFile] = []

    # ----- PROSE (8 files) -----
    # Small: 3 files from detection corpus (already curated clean prose)
    for i in range(3):
        src = DETECTION_CORPUS / "prose" / f"prose_{i * 10:03d}.txt"
        if not src.exists():
            continue
        data = src.read_bytes()
        manifest.append(_write(out_dir, f"prose/prose_small_{i}.txt", data, "prose",
                               f"detection_corpus/prose/{src.name}"))
    # Medium: concat of several prose files
    for i in range(2):
        data = concat_detection_files(
            DETECTION_CORPUS / "prose", n_files=15, seed=seed + 100 + i
        )
        manifest.append(_write(out_dir, f"prose/prose_medium_{i}.txt", data, "prose",
                               "concat of detection_corpus/prose files"))
    # Large: 1 MB sample from pile_raw (shape-filtered — the pile
    # contains embedded code/config which can land in the middle of
    # a random-offset chunk), plus canterbury alice29.txt, plus
    # dickens chunk.
    if PILE_RAW.exists():
        data = sample_from_file(
            PILE_RAW, 1_048_576, seed + 201, filter_fn=looks_like_prose
        )
        manifest.append(_write(out_dir, "prose/prose_large_pile.txt", data, "prose",
                               "pile_raw_1gb 1 MB sample (prose-filtered)"))
    if (CANTERBURY / "alice29.txt").exists():
        data = (CANTERBURY / "alice29.txt").read_bytes()
        manifest.append(_write(out_dir, "prose/prose_alice29.txt", data, "prose",
                               "canterbury/alice29.txt"))
    if (SILESIA / "dickens").exists():
        data = (SILESIA / "dickens").read_bytes()[:1_048_576]
        manifest.append(_write(out_dir, "prose/prose_dickens_1mb.txt", data, "prose",
                               "silesia/dickens first 1 MB"))

    # ----- CODE (8 files) -----
    for i in range(3):
        src = DETECTION_CORPUS / "code" / f"code_py_{i * 8:03d}.py"
        if not src.exists():
            continue
        data = src.read_bytes()
        manifest.append(_write(out_dir, f"code/code_small_{i}.py", data, "code",
                               f"detection_corpus/code/{src.name}"))
    # A JS + a Java to get lang diversity
    for fname in ("code_js_000.js", "code_java_000.java", "code_rust_000.rs"):
        src = DETECTION_CORPUS / "code" / fname
        if src.exists():
            data = src.read_bytes()
            manifest.append(_write(out_dir, f"code/{fname}", data, "code",
                                   f"detection_corpus/code/{fname}"))
    # Medium: concat of 15 Python files
    data = concat_detection_files(
        DETECTION_CORPUS / "code", n_files=25, seed=seed + 300
    )
    manifest.append(_write(out_dir, "code/code_medium.py", data, "code",
                           "concat of detection_corpus/code files"))
    # Large: 1 MB sample from code_diverse_real
    if CODE_DIVERSE.exists():
        data = sample_from_file(CODE_DIVERSE, 1_048_576, seed + 301)
        manifest.append(_write(out_dir, "code/code_large.py", data, "code",
                               "code_diverse_real 1 MB sample"))

    # ----- STRUCTURED (8 files) -----
    for fname in ("json_000.json", "yaml_k8s_000.yaml", "toml_000.toml",
                  "xml_000.xml", "env_000.env"):
        src = DETECTION_CORPUS / "structured" / fname
        if src.exists():
            data = src.read_bytes()
            manifest.append(_write(out_dir, f"structured/{fname}", data, "structured",
                                   f"detection_corpus/structured/{fname}"))
    # Medium: concat of JSON files
    data = concat_detection_files(
        DETECTION_CORPUS / "structured", n_files=20, seed=seed + 400
    )
    manifest.append(_write(out_dir, "structured/structured_medium.json", data, "structured",
                           "concat of detection_corpus/structured files"))
    # Large: silesia/xml (5 MB) and a big YAML sample from code_real
    if (SILESIA / "xml").exists():
        data = (SILESIA / "xml").read_bytes()[:1_048_576]
        manifest.append(_write(out_dir, "structured/xml_silesia_1mb.xml", data, "structured",
                               "silesia/xml first 1 MB"))
    if CODE_REAL.exists():
        # code_real.txt contains YAML configs mixed with Dockerfile
        # fragments and embedded numeric blobs. Shape-filter so the
        # sample actually looks YAML-shaped.
        data = sample_from_file(
            CODE_REAL, 512_000, seed + 401, filter_fn=looks_like_yaml
        )
        manifest.append(_write(out_dir, "structured/yaml_large.yaml", data, "structured",
                               "code_real 512 KB sample (yaml-filtered)"))

    # ----- LOGS (8 files) -----
    for fname in ("syslog_000.log", "nginx_000.log", "jsonl_000.log"):
        src = DETECTION_CORPUS / "logs" / fname
        if src.exists():
            data = src.read_bytes()
            manifest.append(_write(out_dir, f"logs/{fname}", data, "logs",
                                   f"detection_corpus/logs/{fname}"))
    # Medium: concat of same-format syslog files. Mixing JSONL logs
    # with syslog-format logs in one file confuses both detectors
    # (the JSONL heuristic needs >70% per-line JSON; syslog heuristic
    # needs timestamp ratio >50%) so the combined file fires neither.
    data = concat_detection_files(
        DETECTION_CORPUS / "logs", n_files=25, seed=seed + 500,
        name_prefix="syslog_",
    )
    manifest.append(_write(out_dir, "logs/logs_medium_syslog.log", data, "logs",
                           "concat of detection_corpus/logs syslog_* files"))
    # Large: samples from logs_real (BGL, 1 GB raw)
    if LOGS_REAL.exists():
        data = sample_from_file(LOGS_REAL, 1_048_576, seed + 501)
        manifest.append(_write(out_dir, "logs/logs_large_bgl.log", data, "logs",
                               "logs_real 1 MB sample"))
    # ~300 KB concat of nginx-style access logs (same format through
    # the whole file — see comment above about why mixing formats
    # breaks detection).
    data = concat_detection_files(
        DETECTION_CORPUS / "logs", n_files=80, seed=seed + 502,
        name_prefix="nginx_",
    )
    manifest.append(_write(out_dir, "logs/logs_medium_nginx.log", data, "logs",
                           "concat of detection_corpus/logs nginx_* files"))
    # Another small sample at different offset
    for i in range(2):
        data = sample_from_file(LOGS_REAL, 20_000, seed + 510 + i)
        manifest.append(_write(out_dir, f"logs/logs_small_{i}.log", data, "logs",
                               f"logs_real 20 KB sample seed {seed + 510 + i}"))

    # ----- TABULAR (8 files) -----
    for fname in ("csv_real_000.csv", "csv_synth_000.csv", "tsv_000.tsv"):
        src = DETECTION_CORPUS / "tabular" / fname
        if src.exists():
            data = src.read_bytes()
            manifest.append(_write(out_dir, f"tabular/{fname}", data, "tabular",
                                   f"detection_corpus/tabular/{fname}"))
    # Medium: concat of CSVs with the same schema. The synthetic
    # CSVs vary column count per file (4-8 cols), so concatenating
    # across schemas breaks the delimiter-count consistency check.
    # tsv_* files have consistent 4-8 column counts per file — use
    # those for medium-size because they're more likely to share
    # schema.
    data = concat_detection_files(
        DETECTION_CORPUS / "tabular", n_files=25, seed=seed + 600,
        name_prefix="tsv_",
    )
    manifest.append(_write(out_dir, "tabular/tabular_medium.tsv", data, "tabular",
                           "concat of detection_corpus/tabular tsv_* files"))
    # Large: 1 MB and 3 MB samples from csv_real — shape-filter so
    # the sample actually has consistent-width CSV rows (csv_real
    # has 50+ columns so mid-file chunks usually pass the filter).
    if CSV_REAL.exists():
        data = sample_from_file(
            CSV_REAL, 1_048_576, seed + 601, filter_fn=looks_like_csv
        )
        manifest.append(_write(out_dir, "tabular/tabular_large_1mb.csv", data, "tabular",
                               "csv_real 1 MB sample (csv-filtered)"))
        data = sample_from_file(
            CSV_REAL, 3 * 1_048_576, seed + 602, filter_fn=looks_like_csv
        )
        manifest.append(_write(out_dir, "tabular/tabular_large_3mb.csv", data, "tabular",
                               "csv_real 3 MB sample (csv-filtered)"))
    # Note: we do NOT include silesia/sao here — `sao` is the SAO
    # star catalog in a fixed-width BINARY format (not text CSV), so
    # detection correctly routes it to fallback. Release-engineer
    # caught this manifest-label mismatch in the task-30 dry-run.
    # Removed rather than relabelled — the canterbury/silesia corpora
    # aren't necessary for tabular (csv_real gives us plenty of text
    # CSV coverage at multiple sizes).

    # ----- MARKUP (8 files) -----
    for fname in ("html_000.html", "md_000.md", "tex_000.tex"):
        src = DETECTION_CORPUS / "markup" / fname
        if src.exists():
            data = src.read_bytes()
            manifest.append(_write(out_dir, f"markup/{fname}", data, "markup",
                                   f"detection_corpus/markup/{fname}"))
    # Medium: concat
    data = concat_detection_files(
        DETECTION_CORPUS / "markup", n_files=25, seed=seed + 700
    )
    manifest.append(_write(out_dir, "markup/markup_medium.html", data, "markup",
                           "concat of detection_corpus/markup files"))
    # Large: 300 KB synthesized HTML (our gen_html in build_detection_corpus)
    data = make_html_with_embedded_js(seed + 701, 500_000)
    manifest.append(_write(out_dir, "markup/html_large_500kb.html", data, "markup",
                           "synthesized HTML+JS 500 KB"))
    # Canterbury cp.html (real HTML)
    if (CANTERBURY / "cp.html").exists():
        data = (CANTERBURY / "cp.html").read_bytes()
        manifest.append(_write(out_dir, "markup/cp.html", data, "markup",
                               "canterbury/cp.html"))
    # eval_suite html.txt
    if (EVAL_SUITE / "html.txt").exists():
        data = (EVAL_SUITE / "html.txt").read_bytes()
        manifest.append(_write(out_dir, "markup/html_eval.html", data, "markup",
                               "eval_suite/html.txt"))

    # ----- FALLBACK (6 files) -----
    # Short files — will short-circuit to Fallback by length.
    for i in range(2):
        src = DETECTION_CORPUS / "fallback" / f"short_{i:03d}.txt"
        if src.exists():
            data = src.read_bytes()
            manifest.append(_write(out_dir, f"fallback/short_{i}.txt", data, "fallback",
                                   f"detection_corpus/fallback/short_{i:03d}.txt"))
    # High-entropy bytes (simulates already-compressed data).
    for i in range(2):
        src = DETECTION_CORPUS / "fallback" / f"highentropy_{i:03d}.bin"
        if src.exists():
            data = src.read_bytes()
            manifest.append(_write(out_dir, f"fallback/highentropy_{i}.bin", data, "fallback",
                                   f"detection_corpus/fallback/highentropy_{i:03d}.bin"))
    # Large pseudo-random blob (1 MB) — stands in for genuinely
    # unknown/binary content that the fallback specialist should
    # gracefully handle.
    rng_blob = random.Random(seed + 800)
    data = bytes(rng_blob.randint(0, 255) for _ in range(256_000))
    manifest.append(_write(out_dir, "fallback/random_256kb.bin", data, "fallback",
                           "256 KB pseudo-random bytes"))
    # Low-signal text (hex blob) to exercise text fallback path.
    rng_blob2 = random.Random(seed + 801)
    hex_chars = "0123456789abcdef"
    data = ("".join(rng_blob2.choice(hex_chars) for _ in range(100_000))).encode()
    manifest.append(_write(out_dir, "fallback/hex_100kb.txt", data, "fallback",
                           "100 KB hex-character blob"))

    # ----- MIXED (4 files) -----
    # Markdown with many code fences — should route to markup (dominant),
    # but exercises fallback policy on small files.
    for i, size in enumerate((50_000, 200_000)):
        data = make_markdown_with_code(seed + 900 + i, size)
        manifest.append(_write(out_dir, f"mixed/markdown_with_code_{i}.md", data, "mixed",
                               f"synthesized markdown+code {size // 1000} KB"))
    # Jupyter notebooks — JSON outer shell, code + markdown inside.
    for i, size in enumerate((30_000, 120_000)):
        data = make_jupyter_notebook(seed + 910 + i, size)
        manifest.append(_write(out_dir, f"mixed/notebook_{i}.ipynb", data, "mixed",
                               f"synthesized .ipynb {size // 1000} KB"))

    return manifest


def write_manifest(manifest: list[BenchFile], out_dir: Path) -> None:
    tsv = out_dir / "manifest.tsv"
    with tsv.open("w") as f:
        f.write("filename\tdomain\tbytes\tsha256\tsource\n")
        for r in manifest:
            f.write(f"{r.relpath}\t{r.domain}\t{r.bytes}\t{r.sha256}\t{r.source}\n")


def summarize(manifest: list[BenchFile]) -> None:
    by_domain: dict[str, list[BenchFile]] = {}
    for r in manifest:
        by_domain.setdefault(r.domain, []).append(r)
    total_bytes = sum(r.bytes for r in manifest)
    print(f"wrote {len(manifest)} files, total {total_bytes / 1_048_576:.1f} MB")
    print()
    print(f"{'domain':<12} {'files':>6} {'min KB':>8} {'median KB':>10} {'max KB':>10} {'total MB':>10}")
    print("-" * 60)
    for domain in ("prose", "code", "structured", "logs", "tabular", "markup", "fallback", "mixed"):
        rs = by_domain.get(domain, [])
        if not rs:
            continue
        sizes = sorted(r.bytes for r in rs)
        mn = sizes[0] / 1024
        md = sizes[len(sizes) // 2] / 1024
        mx = sizes[-1] / 1024
        tot = sum(sizes) / 1_048_576
        print(f"{domain:<12} {len(rs):>6d} {mn:>8.1f} {md:>10.1f} {mx:>10.1f} {tot:>10.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)

    if not DETECTION_CORPUS.exists():
        print(f"error: detection corpus missing at {DETECTION_CORPUS}. "
              f"Run bench/build_detection_corpus.py first.", file=sys.stderr)
        return 2

    manifest = build(args.out, seed=args.seed)
    write_manifest(manifest, args.out)
    summarize(manifest)
    print()
    print(f"manifest: {args.out / 'manifest.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build the labelled detection evaluation corpus (Task 14).

Writes ~1000+ files under `bench/detection_corpus/`, one per specialist
subdirectory, plus a top-level `labels.tsv` that maps every file to its
expected specialist label. Includes clean-domain cases (unambiguous
prose, code, CSV, etc.) and edge cases (minified JS, Jupyter notebooks,
JSON-lines logs, markdown with code fences, non-English prose, YAML
frontmatter, etc.).

Balance target: 100-150 files per domain across the 7 specialists.

Sources used:
  - corpus_build/pile_raw_1gb.txt         -> prose
  - corpus_build/structured/code_diverse_real.txt  -> Python source code
  - corpus_build/structured/code_real.txt  -> YAML/structured configs
  - corpus_build/structured/code_real_extra.txt -> YAML configs
  - corpus_build/structured/csv_real.txt   -> CSV tabular data
  - corpus_build/structured/logs_real.txt  -> BGL logs
  - Synthesized: JSON/TOML/XML/INI/.env, nginx/syslog/JSON logs,
    TSV, HTML/Markdown/LaTeX, edge cases

Labels TSV columns: `filename\texpected_specialist\tcategory\tnotes`
  - `category` is one of: `clean`, `edge`
  - `notes` is a short human-readable tag

Reproducible with `--seed`. Determines sizes via a rough distribution
so the harness sees a mix of small (~500 B), medium (~8 KB) and larger
(~64 KB) files.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_BUILD = REPO_ROOT / "corpus_build"
PILE_RAW = CORPUS_BUILD / "pile_raw_1gb.txt"
CODE_DIVERSE = CORPUS_BUILD / "structured" / "code_diverse_real.txt"
CODE_REAL = CORPUS_BUILD / "structured" / "code_real.txt"
CODE_REAL_EXTRA = CORPUS_BUILD / "structured" / "code_real_extra.txt"
CSV_REAL = CORPUS_BUILD / "structured" / "csv_real.txt"
LOGS_REAL = CORPUS_BUILD / "structured" / "logs_real.txt"

OUT_DIR_DEFAULT = REPO_ROOT / "bench" / "detection_corpus"

# Timestamp patterns for filtering BGL log samples — we want chunks
# that actually look like logs, not bare stack traces.
_TS_RE = re.compile(
    r"(?m)^"
    r"("
    r"\d{4}-\d{2}-\d{2}"  # ISO-8601 prefix
    r"|[A-Z][a-z]{2} +\d{1,2} +\d{2}:\d{2}:\d{2}"  # Syslog "Jan 12 10:00:00"
    r"|\d{10}"  # Unix epoch prefix
    r"|- +\d{10}"  # BGL "- 1117838570"
    r")"
)


def fraction_line_starts(pattern: re.Pattern, data: bytes) -> float:
    """Fraction of lines (of first 40) that match `pattern` at start."""
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return 0.0
    lines = text.splitlines()[:40]
    if not lines:
        return 0.0
    hits = sum(1 for line in lines if pattern.match(line))
    return hits / len(lines)


_CODE_MARKERS = (
    " def ", " class ", " function ", " return ", " import ", "#include",
    " public ", " private ", " static ", " void ", " fn ", " let ", " const ",
    " var ", " async ", " await ", " => ", "::", " struct ", " enum ",
    " impl ", " trait ", " interface ", "jQuery(", "document.", ".getElementById",
    "var ", "function(", "console.log",
)

# Python-specific: require at least one of these to consider a chunk
# "truly Python source" when sampling from code_diverse_real.txt,
# which contains mixed concatenated content including C sources,
# docs, YAML configs, and encrypted blobs.
_PY_HEADER_MARKERS = (
    "def ", "class ", "import ", "from ",
)


def looks_like_python(data: bytes) -> bool:
    """Stricter filter than `looks_like_code` — require Python syntax."""
    try:
        text = data.decode("utf-8", errors="replace")[:4096]
    except Exception:
        return False
    header_hits = sum(1 for m in _PY_HEADER_MARKERS if m in text)
    # Python-specific indent or `self.` / method-call density.
    py_signals = (
        text.count("\n    ")
        + text.count("self.")
        + text.count("def ")
        + text.count("class ")
    )
    return header_hits >= 1 and py_signals >= 3


def looks_like_yaml(data: bytes) -> bool:
    """Cheap check: does this chunk have `key: value` YAML shape?

    Filters out chunks from `code_real.txt` that landed in embedded
    numeric matrices or base64 blobs.
    """
    try:
        text = data.decode("utf-8", errors="replace")[:4096]
    except Exception:
        return False
    lines = text.splitlines()[:60]
    if len(lines) < 5:
        return False
    kv = 0
    for line in lines:
        t = line.lstrip()
        if not t:
            continue
        # `key: value` pattern — letter-started, alphanumeric + _ - .
        if ":" in t:
            key = t.split(":", 1)[0]
            if key and key[0].isalpha() and all(c.isalnum() or c in "_-." for c in key):
                kv += 1
    return kv >= 3


def looks_like_csv(data: bytes, delims=(",", "\t", ";")) -> bool:
    """Cheap check: consistent delimiter counts across lines."""
    try:
        text = data.decode("utf-8", errors="replace")[:4096]
    except Exception:
        return False
    lines = [line for line in text.splitlines() if line][:20]
    if len(lines) < 5:
        return False
    for d in delims:
        counts = [line.count(d) for line in lines]
        mx = max(counts)
        if mx < 2:
            continue
        mode = sum(1 for c in counts if c == mx)
        if mode / len(lines) > 0.7:
            return True
    return False


def looks_like_code(data: bytes) -> bool:
    """Cheap heuristic: does this chunk look like source code?

    Used to filter prose-source samples. `pile_raw_1gb.txt` contains
    forum posts with embedded JS/HTML/Python; we drop chunks where
    code markers fire noticeably.
    """
    try:
        text = data.decode("utf-8", errors="replace")[:4096]
    except Exception:
        return False
    hits = sum(1 for m in _CODE_MARKERS if m in text)
    # Brace / semicolon density — code-heavy text has lots of them,
    # English prose rarely.
    braces = text.count("{") + text.count("}")
    semis = text.count(";")
    punct_ratio = (braces + semis) / max(len(text), 1)
    # Pipe operator (Elixir/functional) and `~s{...}` (Elixir strings).
    elixir = text.count("|>") + text.count("~s{")
    return hits >= 2 or punct_ratio > 0.02 or elixir >= 3


def looks_like_prose(data: bytes) -> bool:
    """Strict prose filter: high letter ratio and sentence punctuation."""
    try:
        text = data.decode("utf-8", errors="replace")[:4096]
    except Exception:
        return False
    ascii_chars = [c for c in text if ord(c) < 128]
    if not ascii_chars:
        return False
    letters = sum(1 for c in ascii_chars if c.isalpha())
    letter_ratio = letters / len(ascii_chars)
    # Sentence structure: periods and commas per word-ish.
    words = len([w for w in text.split() if w])
    puncts = text.count(".") + text.count(",")
    punct_per_word = puncts / max(words, 1)
    # Reject anything code-shaped.
    if looks_like_code(data):
        return False
    return letter_ratio > 0.70 and punct_per_word > 0.05


# ---- Rust detector agreement filter ---- #
#
# For high-precision corpus labels, we can run the actual Rust
# detector on every generated file and keep only those whose detected
# specialist matches the intended label. This creates a training-style
# "self-consistency" corpus: every file has a ground-truth label that
# the detector actually recognises, so detection-accuracy evaluation
# measures *stability* rather than raw correctness.
#
# The downside: this risks circular evaluation. If we filter to keep
# only files the detector recognizes, we can't detect regressions or
# systematic blind spots. To avoid that, we apply this filter only to
# the subset of sources that are known-noisy (pile prose, concatenated
# Python, YAML configs with embedded numeric blobs) — the synthesized
# samples (HTML, Markdown, JSON, ...) get no such filter because they
# are already clean by construction.


def detector_agrees(bin_path: Path, data: bytes, expected: str) -> bool:
    """Return True if the Rust detector classifies `data` as `expected`."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        proc = subprocess.run(
            [str(bin_path), "detect", "--json", tmp_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return False
        rec = json.loads(proc.stdout.strip())
        return rec.get("specialist") == expected
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@dataclass
class FileRec:
    relpath: str
    label: str
    category: str  # "clean" | "edge"
    notes: str


# ---------- Sampling from raw files ---------- #


def sample_chunks(path: Path, n: int, sizes: list[int], seed: int) -> list[bytes]:
    """Extract n random chunks from a large text file.

    For each chunk, pick a random byte offset, seek, read `size` bytes,
    then trim to the first/last newline so chunks are line-aligned.
    Retries on a given slot if the trimmed chunk is too small so we
    always return `n` chunks.
    """
    rng = random.Random(seed)
    if not path.exists():
        raise FileNotFoundError(f"source missing: {path}")
    total = path.stat().st_size
    chunks: list[bytes] = []
    with path.open("rb") as f:
        attempts = 0
        while len(chunks) < n and attempts < n * 20:
            attempts += 1
            size = rng.choice(sizes)
            if size >= total:
                size = total // 2
            offset = rng.randint(0, max(0, total - size - 1))
            f.seek(offset)
            raw = f.read(size)
            first_nl = raw.find(b"\n")
            if first_nl != -1 and first_nl < size - 1:
                raw = raw[first_nl + 1 :]
            last_nl = raw.rfind(b"\n")
            if last_nl != -1:
                raw = raw[: last_nl + 1]
            if len(raw) < 200:
                continue
            chunks.append(raw)
    return chunks


def split_python_files(path: Path, n: int, seed: int, target_size: int = 8000) -> list[bytes]:
    """Split code_diverse_real.txt into plausible per-file chunks.

    Retries on empty slots so we always return `n` chunks.
    """
    rng = random.Random(seed)
    total = path.stat().st_size
    chunks: list[bytes] = []
    with path.open("rb") as f:
        attempts = 0
        while len(chunks) < n and attempts < n * 20:
            attempts += 1
            size = rng.randint(target_size // 2, target_size * 4)
            offset = rng.randint(0, max(0, total - size - 1))
            f.seek(offset)
            raw = f.read(size)
            first_nl = raw.find(b"\n")
            if first_nl != -1:
                raw = raw[first_nl + 1 :]
            last_nl = raw.rfind(b"\n")
            if last_nl != -1:
                raw = raw[: last_nl + 1]
            if len(raw) < 300:
                continue
            chunks.append(raw)
    return chunks


# ---------- Synthesizers ---------- #


def gen_json(rng: random.Random, size_hint: int) -> bytes:
    """Generate JSON-like object of roughly size_hint bytes."""
    def rand_value(depth: int):
        if depth > 3 or rng.random() < 0.3:
            t = rng.choice(["str", "num", "bool"])
            if t == "str":
                return "".join(rng.choice(string.ascii_lowercase + " ") for _ in range(rng.randint(3, 20)))
            if t == "num":
                return rng.randint(0, 100000)
            return rng.choice([True, False, None])
        if rng.random() < 0.5:
            return [rand_value(depth + 1) for _ in range(rng.randint(1, 6))]
        return {f"{rng.choice(['id','name','value','status','data','url','type','count'])}_{i}": rand_value(depth + 1) for i in range(rng.randint(1, 5))}

    root = []
    while len(json.dumps(root)) < size_hint:
        root.append({
            "id": rng.randint(1, 999999),
            "timestamp": f"2026-0{rng.randint(1,9)}-{rng.randint(10,28)}T{rng.randint(10,23)}:{rng.randint(10,59)}:{rng.randint(10,59)}Z",
            "user": f"user_{rng.randint(1, 9999)}",
            "event": rng.choice(["login", "logout", "click", "view", "purchase", "search"]),
            "metadata": rand_value(0),
            "status": rng.choice([200, 201, 204, 400, 404, 500]),
            "path": f"/api/v{rng.randint(1,4)}/{rng.choice(['users','posts','items'])}/{rng.randint(1,9999)}",
        })
    return json.dumps({"events": root}, indent=2).encode()


def gen_yaml(rng: random.Random, size_hint: int) -> bytes:
    lines = []
    kind = rng.choice(["ConfigMap", "Deployment", "Service", "Pod", "Secret"])
    lines.append("apiVersion: v1")
    lines.append(f"kind: {kind}")
    lines.append("metadata:")
    lines.append(f"  name: {rng.choice(['app','service','api','worker'])}-{rng.randint(1, 999)}")
    lines.append(f"  namespace: {rng.choice(['default','kube-system','prod','staging'])}")
    lines.append("  labels:")
    for _ in range(rng.randint(2, 5)):
        lines.append(f"    {rng.choice(['env','tier','team','version'])}: {rng.choice(['prod','backend','platform','v1'])}")
    lines.append("spec:")
    lines.append(f"  replicas: {rng.randint(1, 10)}")
    lines.append("  selector:")
    lines.append("    matchLabels:")
    lines.append(f"      app: app-{rng.randint(1,999)}")
    while sum(len(l) for l in lines) < size_hint:
        lines.append(f"  {rng.choice(['storage','timeout','retries','threads'])}: {rng.randint(1,1000)}")
        lines.append(f"  env:")
        for _ in range(rng.randint(2, 6)):
            lines.append(f"    - name: {rng.choice(['DB_HOST','API_KEY','PORT','LOG_LEVEL'])}_{rng.randint(1,99)}")
            lines.append(f"      value: \"{rng.choice(['prod','dev','test'])}-{rng.randint(100,9999)}\"")
    return "\n".join(lines).encode()


def gen_toml(rng: random.Random, size_hint: int) -> bytes:
    lines = [
        "# Application configuration",
        f"name = \"project-{rng.randint(1,99)}\"",
        f"version = \"{rng.randint(0,9)}.{rng.randint(0,99)}.{rng.randint(0,99)}\"",
        "authors = [\"Alice <alice@example.com>\", \"Bob <bob@example.com>\"]",
        "",
    ]
    sections = ["server", "database", "cache", "logging", "auth", "features"]
    while sum(len(l) for l in lines) < size_hint:
        s = rng.choice(sections)
        lines.append(f"[{s}]")
        for _ in range(rng.randint(3, 8)):
            k = rng.choice(["host","port","timeout","enabled","url","level","max_conn","retries"])
            v_t = rng.choice(["int","str","bool"])
            if v_t == "int":
                lines.append(f"{k} = {rng.randint(1, 65535)}")
            elif v_t == "bool":
                lines.append(f"{k} = {rng.choice(['true','false'])}")
            else:
                lines.append(f"{k} = \"{rng.choice(['localhost','0.0.0.0','/var/log','info'])}\"")
        lines.append("")
    return "\n".join(lines).encode()


def gen_xml(rng: random.Random, size_hint: int) -> bytes:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<catalog>']
    while sum(len(l) for l in lines) < size_hint:
        lines.append('  <book>')
        lines.append(f'    <id>{rng.randint(1000,9999)}</id>')
        lines.append(f'    <title>The {rng.choice(["Great","Secret","Last","Dark","Silent"])} {rng.choice(["Chronicle","Garden","Voyage","Hour","Tower"])}</title>')
        lines.append(f'    <author>{rng.choice(["Alice","Bob","Carol","Dave","Eve"])} {rng.choice(["Smith","Jones","Brown","Lee","Kim"])}</author>')
        lines.append(f'    <year>{rng.randint(1900, 2025)}</year>')
        lines.append(f'    <price currency="USD">{rng.randint(5,100)}.{rng.randint(0,99):02d}</price>')
        lines.append('    <tags>')
        for _ in range(rng.randint(1, 4)):
            lines.append(f'      <tag>{rng.choice(["fiction","mystery","fantasy","thriller","classic"])}</tag>')
        lines.append('    </tags>')
        lines.append('  </book>')
    lines.append('</catalog>')
    return "\n".join(lines).encode()


def gen_ini(rng: random.Random, size_hint: int) -> bytes:
    lines = ["# INI config file", ""]
    sections = ["general", "database", "network", "logging", "paths"]
    while sum(len(l) for l in lines) < size_hint:
        s = rng.choice(sections)
        lines.append(f"[{s}]")
        for _ in range(rng.randint(3, 7)):
            k = rng.choice(["name","host","port","timeout","level","max","min","path"])
            v = rng.choice([str(rng.randint(1, 1000)), rng.choice(["yes","no","on","off"]), f"/var/lib/{rng.choice(['app','data','db'])}"])
            lines.append(f"{k} = {v}")
        lines.append("")
    return "\n".join(lines).encode()


def gen_env(rng: random.Random, size_hint: int) -> bytes:
    keys = ["DATABASE_URL","API_KEY","SECRET_KEY","REDIS_URL","PORT","HOST","LOG_LEVEL","NODE_ENV","DEBUG","AWS_REGION","S3_BUCKET","JWT_SECRET","SMTP_HOST","CACHE_TTL"]
    lines = []
    while sum(len(l) for l in lines) < size_hint:
        k = rng.choice(keys) + f"_{rng.randint(0,99)}"
        v = rng.choice([
            "postgres://user:pass@localhost/db",
            "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(32)),
            str(rng.randint(1, 65535)),
            rng.choice(["production","development","staging"]),
        ])
        lines.append(f"{k}={v}")
    return "\n".join(lines).encode()


def gen_syslog(rng: random.Random, size_hint: int) -> bytes:
    hosts = ["web01","web02","db01","cache-a","lb-edge","worker-1"]
    procs = ["sshd","cron","systemd","kernel","nginx","postfix","snmpd"]
    lines = []
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    while sum(len(l) for l in lines) < size_hint:
        ts = f"{rng.choice(months)} {rng.randint(1,28):2d} {rng.randint(0,23):02d}:{rng.randint(0,59):02d}:{rng.randint(0,59):02d}"
        host = rng.choice(hosts)
        proc = rng.choice(procs)
        pid = rng.randint(100, 99999)
        level = rng.choice(["INFO","WARN","ERROR","DEBUG"])
        msg = rng.choice([
            f"Accepted publickey for root from 10.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}",
            f"Started session {rng.randint(1000,99999)} of user {rng.choice(['alice','bob','carol'])}",
            f"Connection from 192.168.{rng.randint(0,255)}.{rng.randint(0,255)} closed",
            f"Processed {rng.randint(10,9999)} requests in {rng.randint(10,9999)}ms",
            f"Cache miss for key prefix/{rng.randint(1,99999)}",
        ])
        lines.append(f"{ts} {host} {proc}[{pid}]: {level} {msg}")
    return "\n".join(lines).encode()


def gen_nginx_log(rng: random.Random, size_hint: int) -> bytes:
    lines = []
    methods = ["GET","POST","PUT","DELETE","HEAD"]
    paths = ["/","/api/v1/users","/api/v2/posts/42","/static/app.css","/static/app.js","/favicon.ico","/login","/logout"]
    agents = [
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "curl/7.68.0",
        "python-requests/2.28.1",
    ]
    while sum(len(l) for l in lines) < size_hint:
        ip = f"{rng.randint(1,255)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}"
        ts = f"[{rng.randint(1,28):02d}/Jun/2026:{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:{rng.randint(0,59):02d} +0000]"
        method = rng.choice(methods)
        path = rng.choice(paths)
        status = rng.choice([200,200,200,301,302,304,400,404,500])
        size = rng.randint(100, 100000)
        ua = rng.choice(agents)
        lines.append(f'{ip} - - {ts} "{method} {path} HTTP/1.1" {status} {size} "-" "{ua}"')
    return "\n".join(lines).encode()


def gen_jsonlines_log(rng: random.Random, size_hint: int) -> bytes:
    lines = []
    while sum(len(l) for l in lines) < size_hint:
        rec = {
            "ts": f"2026-04-{rng.randint(1,30):02d}T{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:{rng.randint(0,59):02d}Z",
            "level": rng.choice(["INFO","WARN","ERROR","DEBUG"]),
            "svc": rng.choice(["auth","api","worker","db","cache"]),
            "msg": rng.choice(["request completed","request failed","cache miss","retry","connection closed"]),
            "req_id": "".join(rng.choice(string.hexdigits.lower()) for _ in range(16)),
            "duration_ms": rng.randint(1, 5000),
        }
        lines.append(json.dumps(rec))
    return "\n".join(lines).encode()


def gen_tsv(rng: random.Random, size_hint: int) -> bytes:
    cols = rng.randint(4, 9)
    headers = rng.sample(["id","name","age","city","country","score","email","date","status","tag","amount","currency","type","count"], cols)
    lines = ["\t".join(headers)]
    while sum(len(l) for l in lines) < size_hint:
        row = []
        for h in headers:
            if h in ("id","age","score","amount","count"):
                row.append(str(rng.randint(1, 99999)))
            elif h in ("date",):
                row.append(f"2026-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}")
            elif h in ("email",):
                row.append(f"user{rng.randint(1,9999)}@example.com")
            else:
                row.append(rng.choice(["Alice","Bob","Carol","USA","UK","active","premium","foo","bar","baz"]))
        lines.append("\t".join(row))
    return "\n".join(lines).encode()


def gen_csv_small(rng: random.Random, size_hint: int) -> bytes:
    cols = rng.randint(3, 8)
    headers = rng.sample(["id","name","age","city","country","score","email","date","status","tag","amount"], cols)
    lines = [",".join(headers)]
    while sum(len(l) for l in lines) < size_hint:
        row = []
        for h in headers:
            if h in ("id","age","score","amount"):
                row.append(str(rng.randint(1, 99999)))
            elif h in ("date",):
                row.append(f"2026-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}")
            elif h in ("email",):
                row.append(f"u{rng.randint(1,999)}@ex.com")
            else:
                row.append(rng.choice(["Alice","Bob","USA","UK","active","foo","bar"]))
        lines.append(",".join(row))
    return "\n".join(lines).encode()


def gen_html(rng: random.Random, size_hint: int) -> bytes:
    parts = [
        "<!DOCTYPE html>",
        "<html lang=\"en\">",
        "<head>",
        f"  <title>Page {rng.randint(1,999)}</title>",
        "  <meta charset=\"UTF-8\">",
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">",
        "  <link rel=\"stylesheet\" href=\"/static/style.css\">",
        "</head>",
        "<body>",
        "  <header>",
        f"    <h1>{rng.choice(['Welcome','Dashboard','Profile','Settings'])}</h1>",
        "    <nav><ul>",
        "      <li><a href=\"/\">Home</a></li>",
        "      <li><a href=\"/about\">About</a></li>",
        "      <li><a href=\"/contact\">Contact</a></li>",
        "    </ul></nav>",
        "  </header>",
        "  <main>",
    ]
    while sum(len(l) for l in parts) < size_hint:
        parts.append(f"    <section class=\"sec-{rng.randint(1,99)}\">")
        parts.append(f"      <h2>{rng.choice(['Overview','Details','Summary','Items'])}</h2>")
        for _ in range(rng.randint(1, 4)):
            parts.append(f"      <p>This is paragraph {rng.randint(1,999)} with <a href=\"https://example.com/{rng.randint(1,99)}\">a link</a> and <em>emphasis</em>.</p>")
        parts.append("      <ul>")
        for _ in range(rng.randint(2, 6)):
            parts.append(f"        <li>Item {rng.randint(1,99)}: <span>{rng.choice(['foo','bar','baz','qux'])}</span></li>")
        parts.append("      </ul>")
        parts.append("    </section>")
    parts.extend(["  </main>", "  <footer><p>&copy; 2026 Example</p></footer>", "</body>", "</html>"])
    return "\n".join(parts).encode()


def gen_markdown(rng: random.Random, size_hint: int, with_code: bool = True) -> bytes:
    parts = [f"# {rng.choice(['Introduction','Overview','Getting Started','Guide','Tutorial'])}", "",
             "This document walks through the main concepts and gives concrete examples.", ""]
    while sum(len(l) for l in parts) < size_hint:
        parts.append(f"## {rng.choice(['Setup','Usage','Configuration','Examples','Troubleshooting'])}")
        parts.append("")
        parts.append(f"Here is some text about the {rng.choice(['project','library','tool','module'])}. " * rng.randint(2, 6))
        parts.append("")
        parts.append(f"- {rng.choice(['First','Second','Third'])} bullet about {rng.choice(['installation','usage','API'])}")
        parts.append(f"- Another bullet with [a link]({rng.choice(['https://example.com','https://github.com/x/y','#section'])})")
        parts.append(f"- Third bullet with **bold** and *italic* text")
        parts.append("")
        if with_code and rng.random() < 0.6:
            parts.append(f"```{rng.choice(['python','bash','javascript','rust'])}")
            parts.append(f"def foo(x):")
            parts.append(f"    return x + {rng.randint(1,99)}")
            parts.append(f"")
            parts.append(f"print(foo({rng.randint(1,99)}))")
            parts.append("```")
            parts.append("")
    return "\n".join(parts).encode()


def gen_latex(rng: random.Random, size_hint: int) -> bytes:
    parts = [
        "\\documentclass{article}",
        "\\usepackage{amsmath}",
        "\\usepackage{graphicx}",
        f"\\title{{Paper {rng.randint(1,999)}}}",
        f"\\author{{{rng.choice(['Alice','Bob','Carol'])} {rng.choice(['Smith','Jones','Brown'])}}}",
        "\\begin{document}",
        "\\maketitle",
        "\\begin{abstract}",
        "We present a study of various phenomena with significant implications for the field.",
        "\\end{abstract}",
    ]
    while sum(len(l) for l in parts) < size_hint:
        parts.append(f"\\section{{{rng.choice(['Introduction','Related Work','Methods','Results','Discussion'])}}}")
        parts.append(f"In this section we discuss {rng.choice(['compression','modeling','evaluation'])} techniques.")
        parts.append(f"The equation is \\begin{{equation}} E = mc^{{{rng.randint(2,4)}}} + \\sum_{{i=0}}^{{{rng.randint(5,20)}}} x_i \\end{{equation}}")
        parts.append("\\begin{itemize}")
        for _ in range(rng.randint(2, 5)):
            parts.append(f"  \\item {rng.choice(['First','Second','Third','Fourth'])} observation about the data")
        parts.append("\\end{itemize}")
    parts.append("\\end{document}")
    return "\n".join(parts).encode()


# ---- Synthetic code in non-Python languages ---- #


def gen_js(rng: random.Random, size_hint: int) -> bytes:
    parts = [
        "const express = require('express');",
        "const app = express();",
        "",
        f"const PORT = {rng.randint(3000, 9999)};",
        "",
    ]
    while sum(len(l) for l in parts) < size_hint:
        fn = rng.choice(["fetchUser","createPost","deleteItem","updateRecord","listAll"])
        parts.append(f"async function {fn}(id) {{")
        parts.append(f"  const res = await fetch(`/api/v{rng.randint(1,3)}/${{id}}`);")
        parts.append("  if (!res.ok) throw new Error('request failed');")
        parts.append("  const data = await res.json();")
        parts.append(f"  return {{ id, status: {rng.randint(200,500)}, data }};")
        parts.append("}")
        parts.append("")
        parts.append(f"app.get('/{fn}/:id', async (req, res) => {{")
        parts.append(f"  const item = await {fn}(req.params.id);")
        parts.append(f"  res.json(item);")
        parts.append("});")
        parts.append("")
    parts.append("app.listen(PORT, () => console.log(`listening on ${PORT}`));")
    return "\n".join(parts).encode()


def gen_minified_js(rng: random.Random, size_hint: int) -> bytes:
    # Classic minified JS: no whitespace, dense braces, single letters.
    out = []
    while sum(len(s) for s in out) < size_hint:
        # A long minified expression
        pieces = []
        for _ in range(rng.randint(5, 15)):
            op = rng.choice([
                f"function {chr(97+rng.randint(0,25))}({chr(97+rng.randint(0,25))},{chr(97+rng.randint(0,25))}){{return {chr(97+rng.randint(0,25))}+{rng.randint(1,99)};}}",
                f"var {chr(97+rng.randint(0,25))}={{a:{rng.randint(1,99)},b:{rng.randint(1,99)},c:'{rng.choice(['x','y','z'])}'}};",
                f"if({chr(97+rng.randint(0,25))}>{rng.randint(1,99)}){{return!0;}}else{{return!1;}}",
                f"for(var {chr(97+rng.randint(0,25))}=0;{chr(97+rng.randint(0,25))}<{rng.randint(10,99)};++{chr(97+rng.randint(0,25))}){{console.log({chr(97+rng.randint(0,25))});}}",
            ])
            pieces.append(op)
        out.append("".join(pieces))
    return "".join(out).encode()


def gen_java(rng: random.Random, size_hint: int) -> bytes:
    parts = [
        "package com.example.service;",
        "",
        "import java.util.List;",
        "import java.util.Optional;",
        "import java.util.stream.Collectors;",
        "",
        f"public class Service{rng.randint(1,99)} {{",
        f"    private final int limit = {rng.randint(10,1000)};",
        "",
    ]
    while sum(len(l) for l in parts) < size_hint:
        m = rng.choice(["findAll","findById","update","delete","create"])
        parts.append(f"    public Optional<String> {m}(Long id) {{")
        parts.append(f"        if (id == null || id < 0) return Optional.empty();")
        parts.append(f"        return Optional.of(\"result-\" + id);")
        parts.append(f"    }}")
        parts.append("")
        parts.append(f"    private static final String CONST_{rng.randint(1,99)} = \"value-{rng.randint(100,999)}\";")
        parts.append("")
    parts.append("}")
    return "\n".join(parts).encode()


def gen_c(rng: random.Random, size_hint: int) -> bytes:
    parts = [
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <string.h>",
        "",
        f"#define MAX_BUF {rng.randint(64, 4096)}",
        "",
    ]
    while sum(len(l) for l in parts) < size_hint:
        fname = rng.choice(["process","compute","handle","transform","calculate"])
        parts.append(f"static int {fname}(const char *input, size_t len) {{")
        parts.append(f"    if (!input || len == 0) return -1;")
        parts.append(f"    char buf[MAX_BUF];")
        parts.append(f"    size_t n = len < MAX_BUF ? len : MAX_BUF - 1;")
        parts.append(f"    memcpy(buf, input, n);")
        parts.append(f"    buf[n] = '\\0';")
        parts.append(f"    for (size_t i = 0; i < n; ++i) {{")
        parts.append(f"        if (buf[i] == '\\n') buf[i] = ' ';")
        parts.append(f"    }}")
        parts.append(f"    return (int)n;")
        parts.append(f"}}")
        parts.append("")
    parts.append("int main(int argc, char **argv) { return 0; }")
    return "\n".join(parts).encode()


def gen_go(rng: random.Random, size_hint: int) -> bytes:
    parts = [
        "package main",
        "",
        "import (",
        "    \"fmt\"",
        "    \"log\"",
        "    \"net/http\"",
        ")",
        "",
    ]
    while sum(len(l) for l in parts) < size_hint:
        parts.append(f"func handle{rng.randint(1,99)}(w http.ResponseWriter, r *http.Request) {{")
        parts.append(f"    if r.Method != \"GET\" {{")
        parts.append(f"        http.Error(w, \"method not allowed\", http.StatusMethodNotAllowed)")
        parts.append(f"        return")
        parts.append(f"    }}")
        parts.append(f"    fmt.Fprintf(w, \"hello %s\", r.URL.Path)")
        parts.append(f"}}")
        parts.append("")
    parts.append("func main() { log.Fatal(http.ListenAndServe(\":8080\", nil)) }")
    return "\n".join(parts).encode()


def gen_rust(rng: random.Random, size_hint: int) -> bytes:
    parts = [
        "use std::collections::HashMap;",
        "use std::io::{self, Read, Write};",
        "",
    ]
    while sum(len(l) for l in parts) < size_hint:
        s = rng.choice(["Config","Handler","Session","Request","Response"])
        parts.append(f"pub struct {s}{rng.randint(1,99)} {{")
        parts.append(f"    pub id: u64,")
        parts.append(f"    pub name: String,")
        parts.append(f"    pub data: Vec<u8>,")
        parts.append(f"}}")
        parts.append("")
        parts.append(f"impl {s}{rng.randint(1,99)} {{")
        parts.append(f"    pub fn new(id: u64) -> Self {{ Self {{ id, name: String::new(), data: Vec::new() }} }}")
        parts.append(f"    pub fn len(&self) -> usize {{ self.data.len() }}")
        parts.append(f"}}")
        parts.append("")
    parts.append("fn main() { println!(\"hello\"); }")
    return "\n".join(parts).encode()


def gen_ipynb(rng: random.Random, size_hint: int) -> bytes:
    """Jupyter notebook — structurally JSON, but user content is code/markdown."""
    cells = []
    while sum(len(str(c)) for c in cells) < size_hint:
        if rng.random() < 0.6:
            # code cell
            cells.append({
                "cell_type": "code",
                "execution_count": rng.randint(1, 99),
                "metadata": {},
                "outputs": [],
                "source": [
                    f"import numpy as np\n",
                    f"x = np.arange({rng.randint(1,100)})\n",
                    f"y = x ** {rng.randint(2,4)}\n",
                    f"print(y.sum())\n",
                ],
            })
        else:
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# Section {rng.randint(1,99)}\n", f"Some explanation about the code above.\n"],
            })
    nb = {
        "cells": cells,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, indent=1).encode()


def gen_non_english_prose(rng: random.Random, size_hint: int) -> bytes:
    # Spanish / French / German-ish plausible sentences.
    chunks = [
        "La compresión de datos es un tema fundamental en la informática moderna. Desde los primeros algoritmos hasta los modelos neuronales actuales, el objetivo ha sido siempre reducir el tamaño sin perder información. ",
        "Le chat dort sur le canapé pendant que la pluie tombe doucement sur les toits de la ville. Les voisins préparent le dîner et parlent des événements récents dans le quartier. ",
        "Die Datenkompression ist ein wichtiges Thema in der Informatik. Moderne Verfahren kombinieren statistische Modelle mit neuronalen Netzen, um bessere Ergebnisse zu erzielen als traditionelle Algorithmen. ",
    ]
    out = ""
    while len(out.encode()) < size_hint:
        out += rng.choice(chunks)
    return out.encode()


def gen_yaml_frontmatter_md(rng: random.Random, size_hint: int) -> bytes:
    # Markdown file with YAML frontmatter header.
    head = [
        "---",
        f"title: Post {rng.randint(1,999)}",
        f"date: 2026-0{rng.randint(1,9)}-{rng.randint(10,28)}",
        f"tags: [{', '.join(rng.sample(['python','rust','go','tech','ml','web','note'], 3))}]",
        f"author: {rng.choice(['alice','bob','carol'])}",
        "---",
        "",
    ]
    body = []
    while sum(len(l) for l in head) + sum(len(l) for l in body) < size_hint:
        body.append(f"# {rng.choice(['Heading','Chapter','Section'])} {rng.randint(1,99)}")
        body.append("")
        body.append(f"Normal prose paragraph with some detail about {rng.choice(['topic','subject','idea'])}. " * 3)
        body.append("")
        if rng.random() < 0.5:
            body.append("```python")
            body.append("print('hello')")
            body.append("```")
            body.append("")
    return "\n".join(head + body).encode()


def gen_makefile(rng: random.Random, size_hint: int) -> bytes:
    parts = [
        "CC = gcc",
        "CFLAGS = -O2 -Wall -Wextra",
        "LDFLAGS = -lm",
        "",
        "TARGET = app",
        "SRCS = main.c util.c io.c",
        "OBJS = $(SRCS:.c=.o)",
        "",
        ".PHONY: all clean install",
        "",
        "all: $(TARGET)",
        "",
        "$(TARGET): $(OBJS)",
        "\t$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)",
        "",
        "%.o: %.c",
        "\t$(CC) $(CFLAGS) -c $< -o $@",
        "",
        "clean:",
        "\trm -f $(OBJS) $(TARGET)",
        "",
        "install: $(TARGET)",
        "\tcp $(TARGET) /usr/local/bin/",
    ]
    while sum(len(l) for l in parts) < size_hint:
        tgt = rng.choice(["test","bench","docs","lint"])
        parts.append("")
        parts.append(f"{tgt}: $(TARGET)")
        parts.append(f"\t./$(TARGET) --{tgt}")
    return "\n".join(parts).encode()


def gen_shell_script(rng: random.Random, size_hint: int) -> bytes:
    parts = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"readonly SCRIPT_DIR=\"$(cd \"$(dirname \"${{BASH_SOURCE[0]}}\")\" && pwd)\"",
        f"readonly VERSION=\"{rng.randint(1,9)}.{rng.randint(0,99)}.{rng.randint(0,99)}\"",
        "",
    ]
    while sum(len(l) for l in parts) < size_hint:
        f = rng.choice(["check_deps","run_tests","build","deploy","clean"])
        parts.append(f"{f}() {{")
        parts.append(f"    local arg=\"${{1:-}}\"")
        parts.append(f"    if [[ -z \"$arg\" ]]; then")
        parts.append(f"        echo \"usage: $0 {f} <arg>\" >&2")
        parts.append(f"        return 1")
        parts.append(f"    fi")
        parts.append(f"    echo \"running {f} on $arg\"")
        parts.append(f"    return 0")
        parts.append(f"}}")
        parts.append("")
    parts.append('"$@"')
    return "\n".join(parts).encode()


# ---------- Main corpus builder ---------- #


def _write(out_dir: Path, relpath: str, data: bytes) -> None:
    full = out_dir / relpath
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(data)


def build_corpus(out_dir: Path, seed: int = 17) -> list[FileRec]:
    rng = random.Random(seed)
    records: list[FileRec] = []

    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- PROSE (clean, ~145 files) -------- #
    # pile_raw_1gb.txt is a *balanced* pile — it contains forum posts
    # with embedded code, Elixir/SQL snippets, configs. Apply a strict
    # prose filter (high letter ratio + sentence punctuation + not
    # code-shaped) and over-sample to compensate.
    prose_raw = sample_chunks(
        PILE_RAW, n=600, sizes=[1500, 4000, 12000, 32000], seed=seed
    )
    prose_chunks = [c for c in prose_raw if looks_like_prose(c)][:145]
    for i, chunk in enumerate(prose_chunks):
        rel = f"prose/prose_{i:03d}.txt"
        _write(out_dir, rel, chunk)
        records.append(FileRec(rel, "prose", "clean", "pile_raw_1gb sample (strict-filtered)"))

    # -------- CODE (clean, ~140 files + edge non-Python) -------- #
    # code_diverse_real.txt is a concatenation of many files with
    # embedded C code, docs, YAML configs, and binary blobs — it is
    # NOT pure Python. Use a strict Python-syntax filter so labelled
    # code files actually contain Python.
    py_raw = split_python_files(CODE_DIVERSE, n=600, seed=seed + 1, target_size=8000)
    py_chunks = [c for c in py_raw if looks_like_python(c)][:90]
    for i, chunk in enumerate(py_chunks):
        rel = f"code/code_py_{i:03d}.py"
        _write(out_dir, rel, chunk)
        records.append(FileRec(rel, "code", "clean", "code_diverse_real Python (strict-filtered)"))

    # Synthetic code samples in other langs (clean)
    for i in range(15):
        _write(out_dir, f"code/code_js_{i:03d}.js", gen_js(rng, rng.choice([2000, 8000])))
        records.append(FileRec(f"code/code_js_{i:03d}.js", "code", "clean", "synthetic JS"))
    for i in range(12):
        _write(out_dir, f"code/code_java_{i:03d}.java", gen_java(rng, rng.choice([2000, 8000])))
        records.append(FileRec(f"code/code_java_{i:03d}.java", "code", "clean", "synthetic Java"))
    for i in range(10):
        _write(out_dir, f"code/code_c_{i:03d}.c", gen_c(rng, rng.choice([2000, 6000])))
        records.append(FileRec(f"code/code_c_{i:03d}.c", "code", "clean", "synthetic C"))
    for i in range(10):
        _write(out_dir, f"code/code_go_{i:03d}.go", gen_go(rng, rng.choice([2000, 6000])))
        records.append(FileRec(f"code/code_go_{i:03d}.go", "code", "clean", "synthetic Go"))
    for i in range(10):
        _write(out_dir, f"code/code_rust_{i:03d}.rs", gen_rust(rng, rng.choice([2000, 6000])))
        records.append(FileRec(f"code/code_rust_{i:03d}.rs", "code", "clean", "synthetic Rust"))
    # Shell scripts + makefiles are code-ish
    for i in range(8):
        _write(out_dir, f"code/code_sh_{i:03d}.sh", gen_shell_script(rng, rng.choice([1500, 4000])))
        records.append(FileRec(f"code/code_sh_{i:03d}.sh", "code", "clean", "shell script"))
    for i in range(5):
        _write(out_dir, f"code/code_mk_{i:03d}.mk", gen_makefile(rng, rng.choice([1500, 3000])))
        records.append(FileRec(f"code/code_mk_{i:03d}.mk", "code", "clean", "makefile"))

    # -------- STRUCTURED (clean, ~140 files) -------- #
    # Real YAML configs from code_real — filter chunks that land
    # inside embedded numeric blobs (the source file contains
    # matrix dumps / checkpoint tensors in YAML values).
    yaml_raw = sample_chunks(CODE_REAL, n=120, sizes=[1500, 4000, 8000], seed=seed + 2)
    yaml_chunks = [c for c in yaml_raw if looks_like_yaml(c)][:35]
    for i, chunk in enumerate(yaml_chunks):
        rel = f"structured/yaml_real_{i:03d}.yaml"
        _write(out_dir, rel, chunk)
        records.append(FileRec(rel, "structured", "clean", "code_real YAML (shape-filtered)"))
    yaml_raw2 = sample_chunks(CODE_REAL_EXTRA, n=80, sizes=[1500, 4000, 8000], seed=seed + 3)
    yaml_chunks2 = [c for c in yaml_raw2 if looks_like_yaml(c)][:20]
    for i, chunk in enumerate(yaml_chunks2):
        rel = f"structured/yaml_extra_{i:03d}.yaml"
        _write(out_dir, rel, chunk)
        records.append(FileRec(rel, "structured", "clean", "code_real_extra YAML (shape-filtered)"))
    # Synthetic JSON
    for i in range(25):
        _write(out_dir, f"structured/json_{i:03d}.json", gen_json(rng, rng.choice([800, 4000, 16000])))
        records.append(FileRec(f"structured/json_{i:03d}.json", "structured", "clean", "synthetic JSON"))
    # Synthetic YAML (k8s)
    for i in range(15):
        _write(out_dir, f"structured/yaml_k8s_{i:03d}.yaml", gen_yaml(rng, rng.choice([600, 2000, 5000])))
        records.append(FileRec(f"structured/yaml_k8s_{i:03d}.yaml", "structured", "clean", "synthetic Kubernetes YAML"))
    # TOML
    for i in range(15):
        _write(out_dir, f"structured/toml_{i:03d}.toml", gen_toml(rng, rng.choice([800, 2500])))
        records.append(FileRec(f"structured/toml_{i:03d}.toml", "structured", "clean", "synthetic TOML"))
    # XML
    for i in range(15):
        _write(out_dir, f"structured/xml_{i:03d}.xml", gen_xml(rng, rng.choice([1500, 6000])))
        records.append(FileRec(f"structured/xml_{i:03d}.xml", "structured", "clean", "synthetic XML"))
    # INI
    for i in range(8):
        _write(out_dir, f"structured/ini_{i:03d}.ini", gen_ini(rng, rng.choice([600, 2000])))
        records.append(FileRec(f"structured/ini_{i:03d}.ini", "structured", "clean", "synthetic INI"))
    # .env
    for i in range(8):
        _write(out_dir, f"structured/env_{i:03d}.env", gen_env(rng, rng.choice([400, 1500])))
        records.append(FileRec(f"structured/env_{i:03d}.env", "structured", "clean", "synthetic .env"))

    # -------- LOGS (clean, ~140 files) -------- #
    # Real BGL logs. Filter chunks to require a majority of lines
    # with timestamp-shaped prefixes — mid-file BGL chunks sometimes
    # land in bare stack traces / HDFS paths with no log structure.
    log_raw = sample_chunks(LOGS_REAL, n=160, sizes=[2000, 8000, 32000], seed=seed + 4)
    log_chunks = [c for c in log_raw if fraction_line_starts(_TS_RE, c) > 0.5][:55]
    for i, chunk in enumerate(log_chunks):
        rel = f"logs/bgl_{i:03d}.log"
        _write(out_dir, rel, chunk)
        records.append(FileRec(rel, "logs", "clean", "BGL real logs (timestamp-filtered)"))
    # Synthetic syslog
    for i in range(35):
        _write(out_dir, f"logs/syslog_{i:03d}.log", gen_syslog(rng, rng.choice([1500, 6000, 20000])))
        records.append(FileRec(f"logs/syslog_{i:03d}.log", "logs", "clean", "synthetic syslog"))
    # Synthetic nginx
    for i in range(30):
        _write(out_dir, f"logs/nginx_{i:03d}.log", gen_nginx_log(rng, rng.choice([1500, 5000, 20000])))
        records.append(FileRec(f"logs/nginx_{i:03d}.log", "logs", "clean", "synthetic nginx access log"))
    # JSON-lines logs (clean, clearly log-shaped)
    for i in range(20):
        _write(out_dir, f"logs/jsonl_{i:03d}.log", gen_jsonlines_log(rng, rng.choice([1500, 6000])))
        records.append(FileRec(f"logs/jsonl_{i:03d}.log", "logs", "clean", "JSON-lines log"))

    # -------- TABULAR (clean, ~140 files) -------- #
    # Real CSVs — filter chunks that don't look CSV-shaped (the
    # source file occasionally has header-only regions or multi-line
    # quoted strings that break the per-row delimiter count).
    csv_raw = sample_chunks(CSV_REAL, n=150, sizes=[1500, 6000, 20000], seed=seed + 5)
    csv_chunks = [c for c in csv_raw if looks_like_csv(c)][:55]
    for i, chunk in enumerate(csv_chunks):
        rel = f"tabular/csv_real_{i:03d}.csv"
        _write(out_dir, rel, chunk)
        records.append(FileRec(rel, "tabular", "clean", "csv_real sample (shape-filtered)"))
    for i in range(45):
        _write(out_dir, f"tabular/csv_synth_{i:03d}.csv", gen_csv_small(rng, rng.choice([800, 3000, 10000])))
        records.append(FileRec(f"tabular/csv_synth_{i:03d}.csv", "tabular", "clean", "synthetic CSV"))
    for i in range(35):
        _write(out_dir, f"tabular/tsv_{i:03d}.tsv", gen_tsv(rng, rng.choice([800, 3000, 10000])))
        records.append(FileRec(f"tabular/tsv_{i:03d}.tsv", "tabular", "clean", "synthetic TSV"))

    # -------- MARKUP (clean, ~140 files) -------- #
    for i in range(55):
        _write(out_dir, f"markup/html_{i:03d}.html", gen_html(rng, rng.choice([1500, 6000, 24000])))
        records.append(FileRec(f"markup/html_{i:03d}.html", "markup", "clean", "synthetic HTML"))
    for i in range(50):
        _write(out_dir, f"markup/md_{i:03d}.md", gen_markdown(rng, rng.choice([1500, 5000, 15000]), with_code=rng.random() < 0.5))
        records.append(FileRec(f"markup/md_{i:03d}.md", "markup", "clean", "synthetic Markdown"))
    for i in range(30):
        _write(out_dir, f"markup/tex_{i:03d}.tex", gen_latex(rng, rng.choice([1500, 5000])))
        records.append(FileRec(f"markup/tex_{i:03d}.tex", "markup", "clean", "synthetic LaTeX"))

    # -------- FALLBACK (clean, ~120 files) -------- #
    # Per PHASE_14, fallback is for "mixed/unknown content" where no
    # heuristic fires confidently. We do NOT ship mixed (prose+code)
    # files as "expected fallback" — the spec says detection picks
    # the dominant type, so a prose+code mix correctly routes to
    # prose or code, not fallback. Fallback's real domain is short
    # inputs, high-entropy/binary data, and content with no strong
    # domain signals.

    # Short files (<256 B — detect() short-circuits to Fallback).
    for i in range(45):
        short = b"x = 1\ny = 2\nprint(x + y)\n"
        _write(out_dir, f"fallback/short_{i:03d}.txt", short)
        records.append(FileRec(f"fallback/short_{i:03d}.txt", "fallback", "clean", "too short (<256 B)"))
    # High-entropy random-like (e.g., already-compressed / binary-ish).
    for i in range(45):
        data = bytes(rng.randint(0, 255) for _ in range(rng.choice([1000, 3000])))
        _write(out_dir, f"fallback/highentropy_{i:03d}.bin", data)
        records.append(FileRec(f"fallback/highentropy_{i:03d}.bin", "fallback", "clean", "random high-entropy bytes"))
    # Low-signal text that genuinely has no strong specialist signal:
    # mostly digits/punct, hex blobs, base64-like. These should land
    # in fallback because no specialist heuristic fires above
    # threshold (prose's letter ratio and code's keyword counts both
    # fail).
    for i in range(30):
        kind = i % 3
        if kind == 0:
            # Hex/base64-ish blob (real-world log IDs, hashes).
            chars = "0123456789abcdef"
            data = ("".join(rng.choice(chars) for _ in range(rng.randint(800, 3000))) + "\n").encode()
        elif kind == 1:
            # Numbers-heavy free text with no code or CSV structure
            # (e.g., a scraped page of phone numbers or IDs).
            parts = [f"{rng.randint(1, 99999)} " for _ in range(rng.randint(100, 300))]
            data = "".join(parts).encode()
        else:
            # Punctuation-heavy noise (formulas, math notation out of
            # context, random symbol runs) that isn't code or markup.
            syms = "+-*/=<>%&|^~!?@$#"
            pieces = []
            for _ in range(rng.randint(50, 150)):
                pieces.append(
                    "".join(rng.choice(syms + string.digits) for _ in range(rng.randint(3, 8)))
                )
            data = (" ".join(pieces)).encode()
        _write(out_dir, f"fallback/lowsignal_{i:03d}.txt", data)
        records.append(FileRec(
            f"fallback/lowsignal_{i:03d}.txt", "fallback", "clean",
            "low-signal text (hex/digits/symbols), no domain fires",
        ))

    # -------- EDGE CASES -------- #
    # Minified JS: expected label depends on use case. detect.rs likely
    # won't match code keywords (no spaces around them). We label as
    # `code` (the right specialist) and mark it edge so the analyst sees
    # it's a known hard case.
    for i in range(10):
        _write(out_dir, f"code/minified_js_{i:03d}.min.js", gen_minified_js(rng, rng.choice([1500, 6000])))
        records.append(FileRec(f"code/minified_js_{i:03d}.min.js", "code", "edge", "minified JS — low whitespace"))

    # Jupyter notebooks: structurally JSON (.ipynb), but body is code+md.
    # We label as `structured` (JSON outer shell) — this is the dominant
    # structural signal; revisit if detection accuracy suffers.
    for i in range(8):
        _write(out_dir, f"structured/notebook_{i:03d}.ipynb", gen_ipynb(rng, rng.choice([3000, 10000])))
        records.append(FileRec(f"structured/notebook_{i:03d}.ipynb", "structured", "edge", "Jupyter notebook (JSON shell, code+md inside)"))

    # Markdown with heavy YAML frontmatter — still Markdown.
    for i in range(8):
        _write(out_dir, f"markup/md_frontmatter_{i:03d}.md", gen_yaml_frontmatter_md(rng, rng.choice([1500, 6000])))
        records.append(FileRec(f"markup/md_frontmatter_{i:03d}.md", "markup", "edge", "Markdown with YAML frontmatter"))

    # Non-English prose — still prose specialist.
    for i in range(8):
        _write(out_dir, f"prose/noneng_{i:03d}.txt", gen_non_english_prose(rng, rng.choice([1500, 6000])))
        records.append(FileRec(f"prose/noneng_{i:03d}.txt", "prose", "edge", "non-English prose (es/fr/de mix)"))

    # CSV with very few columns (might look like other things).
    for i in range(5):
        # 2 columns only — the `tabular` detector needs >= 2 delimiters to fire,
        # so this is exactly on the boundary.
        rows = ["name,value"]
        for _ in range(rng.randint(20, 60)):
            rows.append(f"item_{rng.randint(1,999)},{rng.randint(1,99999)}")
        _write(out_dir, f"tabular/csv_thin_{i:03d}.csv", ("\n".join(rows)).encode())
        records.append(FileRec(f"tabular/csv_thin_{i:03d}.csv", "tabular", "edge", "CSV with only 2 columns"))

    # Prose with a short YAML-like header (book metadata followed by
    # narrative). The first 4 KB is dominated by the narrative body,
    # so expected routing is `prose`. We make the header SHORT so the
    # narrative (not the YAML head) dominates the 4 KB detection
    # window.
    for i in range(3):
        data = b"Title: The Great Book\nAuthor: Some Person\nWritten: 2020\n\n"
        data += b" ".join([b"The dog ran quickly across the field." for _ in range(80)])
        _write(out_dir, f"prose/book_header_{i:03d}.txt", data)
        records.append(FileRec(
            f"prose/book_header_{i:03d}.txt", "prose", "edge",
            "narrative with short YAML-like book header",
        ))

    # Logs without timestamps (pure stack traces) — should still be logs.
    for i in range(5):
        parts = [
            "Traceback (most recent call last):",
            "  File \"/app/main.py\", line 42, in <module>",
            "    main()",
            "  File \"/app/main.py\", line 15, in main",
            "    result = process(data)",
            "  File \"/app/handler.py\", line 88, in process",
            "    raise ValueError(\"bad input: %r\" % data)",
            "ValueError: bad input: {'foo': 'bar'}",
            "",
        ]
        out = "\n".join(parts * rng.randint(2, 5))
        _write(out_dir, f"logs/stacktrace_{i:03d}.log", out.encode())
        records.append(FileRec(f"logs/stacktrace_{i:03d}.log", "logs", "edge", "bare stack trace, no timestamps"))

    return records


def write_labels(records: list[FileRec], out_dir: Path) -> None:
    tsv = out_dir / "labels.tsv"
    with tsv.open("w") as f:
        f.write("filename\texpected_specialist\tcategory\tnotes\n")
        for r in records:
            f.write(f"{r.relpath}\t{r.label}\t{r.category}\t{r.notes}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR_DEFAULT, help="Output directory")
    parser.add_argument("--seed", type=int, default=17, help="RNG seed for reproducibility")
    args = parser.parse_args(argv)

    records = build_corpus(args.out, seed=args.seed)
    write_labels(records, args.out)

    # Summary
    by_label: dict[str, dict[str, int]] = {}
    for r in records:
        by_label.setdefault(r.label, {"clean": 0, "edge": 0})
        by_label[r.label][r.category] += 1
    print(f"wrote {len(records)} files to {args.out}")
    print(f"labels.tsv: {args.out / 'labels.tsv'}")
    print()
    print(f"{'specialist':<14} {'clean':>6} {'edge':>6} {'total':>7}")
    print("-" * 36)
    for label in ["prose", "code", "structured", "logs", "tabular", "markup", "fallback"]:
        c = by_label.get(label, {"clean": 0, "edge": 0})
        tot = c["clean"] + c["edge"]
        print(f"{label:<14} {c['clean']:>6} {c['edge']:>6} {tot:>7}")
    print("-" * 36)
    total_clean = sum(v["clean"] for v in by_label.values())
    total_edge = sum(v["edge"] for v in by_label.values())
    print(f"{'TOTAL':<14} {total_clean:>6} {total_edge:>6} {total_clean + total_edge:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

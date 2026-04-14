"""Build the custom compressor-oriented training corpus.

Assembles ~50 GB of text from multiple sources:
- Base: RedPajama/Dolma or Pile (~40 GB, streamed from HuggingFace)
- Augmentation: synthetic logs, CSV, JSON logs, YAML configs, SQL (~7 GB)

The augmentation data is what makes this a compressor corpus instead
of an LLM corpus. The base provides broad text coverage; the
augmentation provides the structured-text patterns that real users
compress (logs, tabular data, configs, database dumps).

Output: a single text file (one document per line, newline-separated)
ready for tokenization with build_pile_corpus.py.

Usage:
    # Full corpus (~50 GB, takes ~2 hours for HF streaming + generation):
    python scripts/build_training_corpus.py --target-gb 50 --output corpus_50gb.txt

    # Quick test (~1 GB):
    python scripts/build_training_corpus.py --target-gb 1 --output corpus_1gb.txt

    # Just the synthetic augmentation (~7 GB):
    python scripts/build_training_corpus.py --synth-only --output synth_7gb.txt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


# ============================================================
# Synthetic generators
# ============================================================

def gen_nginx_log(f, target_bytes: int, seed: int = 42):
    """Generate realistic nginx access log entries."""
    rng = random.Random(seed)
    ips = [f"{rng.randint(10,192)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}" for _ in range(200)]
    methods = ["GET"] * 6 + ["POST"] * 2 + ["PUT", "DELETE", "PATCH", "HEAD"]
    paths = ["/", "/api/v2/users", "/api/v2/posts", "/api/v2/comments",
             "/api/v2/auth/login", "/api/v2/auth/logout", "/api/v2/auth/refresh",
             "/static/js/bundle.min.js", "/static/css/main.css", "/static/img/logo.png",
             "/api/v2/search", "/api/v2/upload", "/api/v2/download",
             "/health", "/metrics", "/api/v2/settings", "/api/v2/notifications",
             "/api/v2/billing", "/api/v2/webhooks", "/favicon.ico",
             "/api/v1/legacy/users", "/graphql", "/ws/connect"]
    agents = [
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15',
        'curl/7.88.1', 'Python-urllib/3.11', 'Go-http-client/2.0',
        'internal-monitor/1.0', 'Prometheus/2.45', 'Datadog-Agent/7.50',
    ]
    statuses = [200] * 10 + [201, 204, 301, 302, 304, 400, 401, 403, 404, 404, 500, 502, 503]
    referrers = ["-"] * 4 + ["https://app.example.com/dashboard", "https://app.example.com/",
                 "https://www.google.com/search?q=example", "https://github.com/example/repo"]

    written = 0
    base = datetime(2024, 6, 15, 0, 0, 0)
    while written < target_bytes:
        ts = base + timedelta(seconds=written // 50)
        ip = rng.choice(ips)
        method = rng.choice(methods)
        path = rng.choice(paths)
        if rng.random() < 0.3:
            path += f"?page={rng.randint(1,100)}&limit={rng.choice([10,20,50,100])}"
        if rng.random() < 0.1:
            path += f"&sort={rng.choice(['created_at','updated_at','name'])}&order={rng.choice(['asc','desc'])}"
        status = rng.choice(statuses)
        size = rng.randint(100, 100000) if status in (200, 201) else rng.randint(0, 1000)
        dur = f"{rng.uniform(0.001, 5.0):.3f}" if rng.random() < 0.5 else "-"
        line = f'{ip} - - [{ts.strftime("%d/%b/%Y:%H:%M:%S")} +0000] "{method} {path} HTTP/1.1" {status} {size} "{rng.choice(referrers)}" "{rng.choice(agents)}" rt={dur}\n'
        f.write(line)
        written += len(line)
    return written


def gen_json_logs(f, target_bytes: int, seed: int = 43):
    """Generate structured JSON application logs (ELK/Datadog style)."""
    rng = random.Random(seed)
    levels = ["INFO"] * 6 + ["WARN"] * 2 + ["ERROR", "DEBUG"]
    services = ["api-gateway", "user-service", "payment-service", "notification-service",
                "auth-service", "search-service", "analytics-worker", "email-worker"]
    messages = {
        "INFO": ["Request completed", "Cache hit", "Connection established", "Task completed",
                 "Health check passed", "Config reloaded", "Batch processed", "Session created"],
        "WARN": ["Slow query detected", "Rate limit approaching", "Retry attempt",
                 "Deprecated API called", "High memory usage", "Connection pool near capacity"],
        "ERROR": ["Internal server error", "Database connection failed", "Timeout exceeded",
                  "Authentication failed", "Invalid input", "Service unavailable"],
        "DEBUG": ["Query executed", "Cache miss", "Token validated", "Middleware applied"],
    }

    written = 0
    base = datetime(2024, 8, 1, 0, 0, 0)
    while written < target_bytes:
        ts = base + timedelta(milliseconds=written // 10)
        level = rng.choice(levels)
        service = rng.choice(services)
        msg = rng.choice(messages[level])
        entry = {
            "timestamp": ts.isoformat() + "Z",
            "level": level,
            "service": service,
            "message": msg,
            "request_id": f"req_{rng.randint(100000, 999999):06d}",
            "duration_ms": rng.randint(1, 5000) if level != "DEBUG" else rng.randint(1, 50),
            "host": f"{service}-{rng.randint(1,10):02d}.prod.internal",
        }
        if level == "ERROR":
            entry["error"] = {"code": rng.choice(["ECONNREFUSED", "ETIMEDOUT", "EPERM", "ENOENT"]),
                              "stack": f"at {service}/src/handlers/{rng.choice(['user','payment','auth'])}.ts:{rng.randint(10,500)}"}
        if rng.random() < 0.3:
            entry["metadata"] = {"user_id": f"usr_{rng.randint(1,50000):05d}",
                                 "ip": f"{rng.randint(10,192)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}"}
        line = json.dumps(entry) + "\n"
        f.write(line)
        written += len(line)
    return written


def gen_csv_data(f, target_bytes: int, seed: int = 44):
    """Generate realistic CSV datasets (mixed schemas)."""
    rng = random.Random(seed)

    schemas = [
        # Web analytics
        ("timestamp,session_id,user_id,event_type,page,duration_ms,status_code,browser,country,revenue_usd",
         lambda: f"{(datetime(2024,1,1)+timedelta(seconds=rng.randint(0,86400*365))).isoformat()}Z,"
                 f"sess_{rng.randint(1,999999):06d},usr_{rng.randint(1,50000):05d},"
                 f"{rng.choice(['page_view','click','scroll','form_submit','search','download','purchase'])},"
                 f"{rng.choice(['/home','/products','/pricing','/about','/blog','/docs','/api','/settings'])},"
                 f"{rng.randint(10,30000)},{rng.choice([200,200,200,201,301,400,500])},"
                 f"{rng.choice(['Chrome/120','Firefox/121','Safari/17','Edge/120','Mobile-Safari/17'])},"
                 f"{rng.choice(['US','GB','DE','FR','JP','BR','IN','CA','AU','KR'])},"
                 f"{rng.uniform(0,500):.2f}"),
        # Server metrics
        ("timestamp,host,cpu_percent,memory_mb,disk_io_read_mb,disk_io_write_mb,network_rx_mb,network_tx_mb,load_avg_1m",
         lambda: f"{(datetime(2024,6,1)+timedelta(seconds=rng.randint(0,86400*30))).isoformat()}Z,"
                 f"srv-{rng.randint(1,100):03d}.prod,"
                 f"{rng.uniform(0,100):.1f},{rng.randint(1000,32000)},"
                 f"{rng.uniform(0,500):.2f},{rng.uniform(0,200):.2f},"
                 f"{rng.uniform(0,1000):.2f},{rng.uniform(0,500):.2f},"
                 f"{rng.uniform(0,16):.2f}"),
        # Financial transactions
        ("transaction_id,timestamp,account_from,account_to,amount_usd,currency,type,status,fee_usd",
         lambda: f"txn_{rng.randint(1000000,9999999)},"
                 f"{(datetime(2024,1,1)+timedelta(seconds=rng.randint(0,86400*180))).isoformat()}Z,"
                 f"acct_{rng.randint(10000,99999)},acct_{rng.randint(10000,99999)},"
                 f"{rng.uniform(1,50000):.2f},{rng.choice(['USD','EUR','GBP','JPY','CAD'])},"
                 f"{rng.choice(['transfer','payment','refund','deposit','withdrawal'])},"
                 f"{rng.choice(['completed','pending','failed','reversed'])},"
                 f"{rng.uniform(0,50):.2f}"),
    ]

    written = 0
    while written < target_bytes:
        header, row_gen = rng.choice(schemas)
        # Write a batch with the same schema (realistic: CSV files have one schema)
        batch_size = rng.randint(100, 5000)
        line = header + "\n"
        f.write(line)
        written += len(line)
        for _ in range(batch_size):
            if written >= target_bytes:
                break
            line = row_gen() + "\n"
            f.write(line)
            written += len(line)
        f.write("\n")  # blank line between batches
        written += 1
    return written


def gen_yaml_configs(f, target_bytes: int, seed: int = 45):
    """Generate realistic YAML/TOML config files."""
    rng = random.Random(seed)

    templates = [
        # Kubernetes deployment
        lambda: f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {rng.choice(['api','web','worker','scheduler','cache'])}-{rng.choice(['prod','staging','dev'])}
  namespace: {rng.choice(['default','monitoring','logging','platform'])}
  labels:
    app: {rng.choice(['frontend','backend','worker'])}
    version: v{rng.randint(1,5)}.{rng.randint(0,20)}.{rng.randint(0,100)}
spec:
  replicas: {rng.choice([1,2,3,5,10])}
  selector:
    matchLabels:
      app: {rng.choice(['frontend','backend','worker'])}
  template:
    spec:
      containers:
      - name: app
        image: registry.example.com/{rng.choice(['api','web','worker'])}:{rng.choice(['latest','v1.2.3','sha-abc123'])}
        ports:
        - containerPort: {rng.choice([3000,8080,8443,9090])}
        resources:
          requests:
            memory: "{rng.choice([128,256,512,1024])}Mi"
            cpu: "{rng.choice([100,250,500,1000])}m"
          limits:
            memory: "{rng.choice([256,512,1024,2048])}Mi"
            cpu: "{rng.choice([250,500,1000,2000])}m"
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        - name: REDIS_HOST
          value: "redis-{rng.choice(['prod','staging'])}.internal"
        livenessProbe:
          httpGet:
            path: /health
            port: {rng.choice([3000,8080])}
          initialDelaySeconds: {rng.choice([10,15,30])}
          periodSeconds: {rng.choice([10,15,30])}
---
""",
        # Docker Compose
        lambda: f"""version: '3.8'
services:
  {rng.choice(['app','api','web'])}:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "{rng.randint(3000,9000)}:{rng.choice([3000,8080])}"
    environment:
      - NODE_ENV={rng.choice(['production','development','test'])}
      - DATABASE_URL=postgres://user:pass@db:{rng.choice([5432,5433])}/{rng.choice(['myapp','testdb'])}
      - REDIS_URL=redis://cache:{rng.choice([6379,6380])}
    depends_on:
      - db
      - cache
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  db:
    image: postgres:{rng.choice([14,15,16])}
    environment:
      POSTGRES_DB: {rng.choice(['myapp','testdb','analytics'])}
      POSTGRES_USER: {rng.choice(['app','admin','root'])}
      POSTGRES_PASSWORD: ${{DB_PASSWORD}}
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "{rng.randint(5432,5440)}:5432"

  cache:
    image: redis:{rng.choice([6,7])}
    ports:
      - "{rng.randint(6379,6385)}:6379"

volumes:
  pgdata:

""",
        # GitHub Actions CI
        lambda: f"""name: {rng.choice(['CI','Build and Test','Deploy','Release'])}
on:
  push:
    branches: [{rng.choice(['main','master','develop'])}]
  pull_request:
    branches: [{rng.choice(['main','master'])}]

jobs:
  {rng.choice(['build','test','lint'])}:
    runs-on: {rng.choice(['ubuntu-latest','ubuntu-22.04','macos-latest'])}
    strategy:
      matrix:
        {rng.choice(['node-version','python-version','go-version'])}: [{rng.choice(["'18'","'20'","'3.11'","'3.12'","'1.21'"])}]
    steps:
    - uses: actions/checkout@v4
    - name: Setup
      uses: actions/setup-{rng.choice(['node','python','go'])}@v4
      with:
        {rng.choice(['node-version','python-version','go-version'])}: ${{{{ matrix.{rng.choice(['node-version','python-version','go-version'])} }}}}
    - name: Install dependencies
      run: {rng.choice(['npm ci','pip install -r requirements.txt','go mod download'])}
    - name: Run tests
      run: {rng.choice(['npm test','pytest','go test ./...'])}
    - name: Build
      run: {rng.choice(['npm run build','python setup.py bdist_wheel','go build -o bin/app'])}

""",
    ]

    written = 0
    while written < target_bytes:
        doc = rng.choice(templates)()
        f.write(doc)
        written += len(doc)
    return written


def gen_sql_dumps(f, target_bytes: int, seed: int = 46):
    """Generate realistic SQL dump fragments."""
    rng = random.Random(seed)

    written = 0
    while written < target_bytes:
        table = rng.choice(["users", "orders", "products", "sessions", "events", "payments"])
        # CREATE TABLE
        block = f"-- Table: {table}\n"
        block += f"DROP TABLE IF EXISTS {table};\n"
        block += f"CREATE TABLE {table} (\n"
        block += f"    id SERIAL PRIMARY KEY,\n"
        if table == "users":
            block += f"    email VARCHAR(255) NOT NULL UNIQUE,\n"
            block += f"    name VARCHAR(100) NOT NULL,\n"
            block += f"    password_hash VARCHAR(60) NOT NULL,\n"
            block += f"    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),\n"
            block += f"    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),\n"
            block += f"    is_active BOOLEAN DEFAULT TRUE\n"
        elif table == "orders":
            block += f"    user_id INTEGER REFERENCES users(id),\n"
            block += f"    total_amount DECIMAL(10,2) NOT NULL,\n"
            block += f"    currency VARCHAR(3) DEFAULT 'USD',\n"
            block += f"    status VARCHAR(20) DEFAULT 'pending',\n"
            block += f"    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()\n"
        else:
            block += f"    name VARCHAR(255) NOT NULL,\n"
            block += f"    data JSONB,\n"
            block += f"    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()\n"
        block += f");\n\n"

        # INSERT statements
        batch = rng.randint(50, 500)
        block += f"INSERT INTO {table} VALUES\n"
        for i in range(batch):
            if table == "users":
                name = f"{rng.choice(['Alice','Bob','Charlie','Diana','Eve','Frank'])}{rng.choice(['','son',' Jr.','ovich'])}"
                block += f"    ({rng.randint(1,100000)}, '{name.lower().replace(' ','')}@{rng.choice(['gmail.com','example.com','corp.io'])}', "
                block += f"'{name}', '$2b$12${''.join(rng.choices(string.ascii_letters+string.digits, k=53))}', "
                block += f"'{(datetime(2024,1,1)+timedelta(days=rng.randint(0,365))).isoformat()}', "
                block += f"'{(datetime(2024,6,1)+timedelta(days=rng.randint(0,180))).isoformat()}', "
                block += f"{'true' if rng.random() > 0.1 else 'false'})"
            elif table == "orders":
                block += f"    ({rng.randint(1,100000)}, {rng.randint(1,50000)}, "
                block += f"{rng.uniform(5,5000):.2f}, '{rng.choice(['USD','EUR','GBP'])}', "
                block += f"'{rng.choice(['pending','completed','shipped','cancelled','refunded'])}', "
                block += f"'{(datetime(2024,1,1)+timedelta(days=rng.randint(0,365))).isoformat()}')"
            else:
                block += f"    ({rng.randint(1,100000)}, '{rng.choice(['Item','Widget','Service','Plan'])} {rng.randint(1,9999)}', "
                block += f"'{{\"type\": \"{rng.choice(['basic','premium','enterprise'])}\", \"count\": {rng.randint(1,1000)}}}', "
                block += f"'{(datetime(2024,1,1)+timedelta(days=rng.randint(0,365))).isoformat()}')"
            block += ",\n" if i < batch - 1 else ";\n\n"

        f.write(block)
        written += len(block)
    return written


# ============================================================
# Main
# ============================================================

def build_synthetic(output_dir: Path, target_gb: float = 7.0):
    """Generate all synthetic augmentation data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target_each = int(target_gb * 1e9 / 5)  # ~5 generators, equal share

    generators = [
        ("nginx_logs.txt", gen_nginx_log, target_each),
        ("json_app_logs.txt", gen_json_logs, target_each),
        ("csv_datasets.txt", gen_csv_data, target_each),
        ("yaml_configs.txt", gen_yaml_configs, target_each),
        ("sql_dumps.txt", gen_sql_dumps, target_each),
    ]

    total = 0
    for fname, gen_fn, target in generators:
        path = output_dir / fname
        print(f"  generating {fname} ({target / 1e9:.1f} GB)...")
        t0 = time.time()
        with open(path, "w") as f:
            written = gen_fn(f, target)
        elapsed = time.time() - t0
        total += written
        print(f"    done: {written / 1e9:.2f} GB in {elapsed:.0f}s")

    print(f"  total synthetic: {total / 1e9:.2f} GB")
    return total


def build_base_corpus(output_path: Path, target_gb: float = 40.0, seed: int = 1204):
    """Stream base corpus from HuggingFace (RedPajama or Pile)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    target_bytes = int(target_gb * 1e9)
    print(f"  streaming {target_gb} GB from HuggingFace...")

    # Try RedPajama first (newer, better curated), fall back to Pile
    try:
        ds = load_dataset("togethercomputer/RedPajama-Data-V2",
                          name="default", split="train", streaming=True,
                          trust_remote_code=True)
        source = "RedPajama-V2"
    except Exception:
        try:
            ds = load_dataset("EleutherAI/the_pile_deduplicated",
                              split="train", streaming=True)
            source = "Pile-dedup"
        except Exception as e:
            print(f"ERROR: could not load any base corpus: {e}")
            sys.exit(1)

    print(f"  source: {source}")
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    bytes_written = 0
    docs = 0
    t0 = time.time()

    with open(output_path, "w", encoding="utf-8") as f:
        for example in ds:
            if bytes_written >= target_bytes:
                break
            text = example.get("text", example.get("raw_content", ""))
            if not text or len(text) < 100:
                continue
            f.write(text)
            f.write("\n")
            bytes_written += len(text.encode("utf-8")) + 1
            docs += 1
            if docs % 50_000 == 0:
                elapsed = time.time() - t0
                gb = bytes_written / 1e9
                print(f"    {docs:,} docs, {gb:.2f}/{target_gb} GB, {elapsed:.0f}s")

    print(f"  done: {docs:,} docs, {bytes_written / 1e9:.2f} GB, {time.time() - t0:.0f}s")
    return bytes_written


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-gb", type=float, default=50.0,
                   help="Total corpus size in GB (base + synthetic).")
    p.add_argument("--synth-gb", type=float, default=7.0,
                   help="Synthetic augmentation size in GB.")
    p.add_argument("--output-dir", type=Path, default=Path("corpus_build"),
                   help="Directory for intermediate files.")
    p.add_argument("--output", type=Path, required=True,
                   help="Final concatenated corpus file.")
    p.add_argument("--synth-only", action="store_true",
                   help="Only generate synthetic data, skip base corpus.")
    p.add_argument("--seed", type=int, default=1204)
    p.add_argument("--s3-upload", type=str, default=None,
                   help="Upload final corpus to S3.")
    args = p.parse_args()

    print(f"=== Building training corpus ===")
    print(f"target: {args.target_gb} GB total ({args.target_gb - args.synth_gb:.0f} GB base + {args.synth_gb:.0f} GB synthetic)")

    # Step 1: synthetic augmentation
    print(f"\n--- Synthetic augmentation ({args.synth_gb} GB) ---")
    synth_total = build_synthetic(args.output_dir, args.synth_gb)

    if args.synth_only:
        # Concatenate synthetic files into output
        print(f"\n--- Concatenating synthetic files ---")
        with open(args.output, "w") as out:
            for f in sorted(args.output_dir.iterdir()):
                if f.suffix == ".txt":
                    print(f"  appending {f.name}")
                    out.write(f.read_text())
        print(f"done: {args.output} ({args.output.stat().st_size / 1e9:.2f} GB)")
        return

    # Step 2: base corpus from HuggingFace
    base_gb = args.target_gb - args.synth_gb
    base_path = args.output_dir / "base_corpus.txt"
    print(f"\n--- Base corpus ({base_gb:.0f} GB from HuggingFace) ---")
    build_base_corpus(base_path, base_gb, args.seed)

    # Step 3: concatenate base + synthetic → final output
    print(f"\n--- Concatenating into {args.output} ---")
    with open(args.output, "w") as out:
        # Base first (larger)
        print(f"  appending base corpus...")
        with open(base_path, "r") as f:
            while True:
                chunk = f.read(10_000_000)
                if not chunk:
                    break
                out.write(chunk)
        # Synthetic augmentation
        for f in sorted(args.output_dir.iterdir()):
            if f.suffix == ".txt" and f.name != "base_corpus.txt":
                print(f"  appending {f.name}")
                out.write(f.read_text())

    final_size = args.output.stat().st_size
    print(f"\ndone: {args.output} ({final_size / 1e9:.2f} GB)")

    if args.s3_upload:
        import subprocess
        print(f"uploading to {args.s3_upload}...")
        subprocess.run(["aws", "s3", "cp", str(args.output), args.s3_upload], check=True)
        print("uploaded")


if __name__ == "__main__":
    main()

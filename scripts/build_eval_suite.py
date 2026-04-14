"""Build the evaluation suite for Phase 11.

Assembles ~5 MB test files for each target domain. Uses existing
corpora where possible, generates synthetic data where needed.
Outputs to bench/corpora/eval_suite/.
"""
import json
import os
import random
import sys
from pathlib import Path
from datetime import datetime, timedelta

EVAL_DIR = Path("bench/corpora/eval_suite")


def build_json_sample(out_path: Path, target_mb: float = 5.0):
    """Generate realistic JSON API responses."""
    random.seed(42)
    target_bytes = int(target_mb * 1e6)
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace",
             "Henry", "Iris", "Jack", "Kate", "Leo", "Mia", "Noah", "Olive"]
    domains = ["example.com", "corp.io", "startup.dev", "bigco.org", "app.net"]
    roles = ["admin", "editor", "viewer", "moderator", "analyst", "developer"]
    themes = ["dark", "light", "auto", "system"]
    langs = ["en", "es", "fr", "de", "ja", "zh", "pt", "ko"]
    events = ["page_view", "click", "api_call", "form_submit", "search", "download"]
    statuses = [200, 200, 200, 200, 201, 204, 301, 400, 403, 404, 500]

    written = 0
    with open(out_path, "w") as f:
        while written < target_bytes:
            # Alternate between user records and event logs
            if random.random() < 0.4:
                batch = {"users": [], "total": 0, "page": random.randint(1, 100)}
                for _ in range(random.randint(5, 20)):
                    user = {
                        "id": random.randint(1, 100000),
                        "name": f"{random.choice(names)} {random.choice(names)}son",
                        "email": f"{random.choice(names).lower()}{random.randint(1,999)}@{random.choice(domains)}",
                        "roles": random.sample(roles, random.randint(1, 3)),
                        "created_at": (datetime(2024, 1, 1) + timedelta(days=random.randint(0, 500))).isoformat() + "Z",
                        "settings": {
                            "theme": random.choice(themes),
                            "notifications": random.choice([True, False]),
                            "language": random.choice(langs),
                        },
                        "metadata": {
                            "last_login": (datetime(2024, 6, 1) + timedelta(hours=random.randint(0, 5000))).isoformat() + "Z",
                            "login_count": random.randint(1, 500),
                        }
                    }
                    batch["users"].append(user)
                batch["total"] = len(batch["users"])
                line = json.dumps(batch) + "\n"
            else:
                events_batch = []
                for _ in range(random.randint(10, 50)):
                    evt = {
                        "timestamp": (datetime(2024, 6, 1) + timedelta(seconds=random.randint(0, 86400*30))).isoformat() + "Z",
                        "user_id": f"usr_{random.randint(1, 10000):05d}",
                        "event": random.choice(events),
                        "duration_ms": random.randint(10, 5000),
                        "status": random.choice(statuses),
                        "path": f"/api/v{random.randint(1,3)}/{random.choice(['users','posts','comments','settings'])}/{random.randint(1,9999)}",
                        "ip": f"{random.randint(10,192)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
                    }
                    events_batch.append(evt)
                line = json.dumps({"events": events_batch}) + "\n"
            f.write(line)
            written += len(line.encode("utf-8"))
    print(f"  json: {out_path} ({written / 1e6:.1f} MB)")


def build_nginx_log(out_path: Path, target_mb: float = 5.0):
    """Generate realistic nginx access log entries."""
    random.seed(43)
    target_bytes = int(target_mb * 1e6)
    ips = [f"{random.randint(10,192)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}" for _ in range(50)]
    methods = ["GET", "GET", "GET", "GET", "POST", "POST", "PUT", "DELETE", "PATCH"]
    paths = ["/", "/api/v2/users", "/api/v2/posts", "/api/v2/comments",
             "/api/v2/auth/login", "/api/v2/auth/logout", "/static/js/bundle.min.js",
             "/static/css/main.css", "/api/v2/search", "/api/v2/upload",
             "/health", "/metrics", "/api/v2/settings", "/favicon.ico"]
    agents = [
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
        'curl/7.88.1', 'Python-urllib/3.11', 'internal-monitor/1.0',
    ]
    statuses = [200, 200, 200, 200, 200, 201, 204, 301, 304, 400, 403, 404, 500, 502]
    referrers = ["-", "https://app.example.com/dashboard", "https://app.example.com/",
                 "https://www.google.com/", "-", "-"]

    written = 0
    base = datetime(2024, 6, 15, 0, 0, 0)
    with open(out_path, "w") as f:
        while written < target_bytes:
            ts = base + timedelta(seconds=written // 100)
            ip = random.choice(ips)
            method = random.choice(methods)
            path = random.choice(paths)
            if random.random() < 0.3:
                path += f"?page={random.randint(1,50)}&limit={random.choice([10,20,50,100])}"
            status = random.choice(statuses)
            size = random.randint(0, 50000) if status == 200 else random.randint(0, 500)
            agent = random.choice(agents)
            ref = random.choice(referrers)
            line = f'{ip} - - [{ts.strftime("%d/%b/%Y:%H:%M:%S")} +0000] "{method} {path} HTTP/1.1" {status} {size} "{ref}" "{agent}"\n'
            f.write(line)
            written += len(line.encode("utf-8"))
    print(f"  nginx: {out_path} ({written / 1e6:.1f} MB)")


def build_csv_sample(out_path: Path, target_mb: float = 5.0):
    """Generate realistic CSV data (event analytics)."""
    random.seed(44)
    target_bytes = int(target_mb * 1e6)
    events = ["page_view", "click", "scroll", "form_submit", "search", "download", "error", "purchase"]
    browsers = ["Chrome/120", "Firefox/121", "Safari/17", "Edge/120", "Mobile-Safari/17", "Chrome-Mobile/120"]
    countries = ["US", "GB", "DE", "FR", "JP", "BR", "IN", "CA", "AU", "KR"]
    pages = ["/home", "/products", "/pricing", "/about", "/blog", "/docs", "/api", "/login", "/signup", "/settings"]

    written = 0
    with open(out_path, "w") as f:
        header = "timestamp,session_id,user_id,event_type,page,duration_ms,status_code,browser,country,revenue_usd\n"
        f.write(header)
        written += len(header.encode("utf-8"))
        base = datetime(2024, 6, 1)
        while written < target_bytes:
            ts = base + timedelta(seconds=random.randint(0, 86400 * 30))
            sess = f"sess_{random.randint(1, 999999):06d}"
            uid = f"usr_{random.randint(1, 50000):05d}"
            evt = random.choice(events)
            page = random.choice(pages)
            dur = random.randint(10, 30000)
            status = random.choice([200, 200, 200, 200, 301, 400, 500])
            browser = random.choice(browsers)
            country = random.choice(countries)
            rev = f"{random.uniform(0, 200):.2f}" if evt == "purchase" else "0.00"
            line = f"{ts.isoformat()}Z,{sess},{uid},{evt},{page},{dur},{status},{browser},{country},{rev}\n"
            f.write(line)
            written += len(line.encode("utf-8"))
    print(f"  csv: {out_path} ({written / 1e6:.1f} MB)")


def copy_existing(src: Path, dst: Path, label: str):
    """Copy an existing corpus file."""
    if src.exists():
        import shutil
        shutil.copy2(src, dst)
        print(f"  {label}: {dst} ({src.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"  {label}: MISSING {src}")


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    base = Path("bench/corpora")

    print("building eval suite...")

    # Existing corpora
    copy_existing(base / "enwik6", EVAL_DIR / "enwik6.txt", "wikipedia")
    copy_existing(base / "silesia/webster", EVAL_DIR / "webster.txt", "dictionary")
    copy_existing(base / "silesia/xml", EVAL_DIR / "xml_silesia.txt", "xml")
    copy_existing(base / "canterbury/fields.c", EVAL_DIR / "c_source.txt", "c_code")
    copy_existing(base / "canterbury/cp.html", EVAL_DIR / "html.txt", "html")
    copy_existing(base / "canterbury/alice29.txt", EVAL_DIR / "fiction.txt", "fiction")

    # Generated corpora
    build_json_sample(EVAL_DIR / "json_api.txt", target_mb=5.0)
    build_nginx_log(EVAL_DIR / "nginx_log.txt", target_mb=5.0)
    build_csv_sample(EVAL_DIR / "csv_data.txt", target_mb=5.0)

    print("\neval suite ready:")
    total = 0
    for f in sorted(EVAL_DIR.iterdir()):
        sz = f.stat().st_size
        total += sz
        print(f"  {f.name:<25s} {sz / 1e6:>8.2f} MB")
    print(f"  {'TOTAL':<25s} {total / 1e6:>8.2f} MB")


if __name__ == "__main__":
    os.chdir(sys.argv[1] if len(sys.argv) > 1 else ".")
    main()

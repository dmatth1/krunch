#!/usr/bin/env python3
"""
Local roundtrip test: compress then decompress a corpus, verify byte-exactness.

Usage:
  python scripts/roundtrip_test.py [--file data/spike6/wildchat_en_content.content.bin]
  python scripts/roundtrip_test.py --url http://localhost:8080 --file <path>
"""

import sys
import time
import hashlib
import argparse
import urllib.request

DEFAULT_ENDPOINT = "http://localhost:8080"


def roundtrip(endpoint: str, data: bytes) -> tuple[bytes, float, float]:
    def post(path: str, body: bytes) -> bytes:
        req = urllib.request.Request(
            f"{endpoint}{path}", data=body,
            headers={"Content-Type": "application/octet-stream"}, method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            return resp.read()

    t0 = time.perf_counter()
    compressed = post("/compress", data)
    compress_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    decompressed = post("/decompress", compressed)
    decompress_time = time.perf_counter() - t1

    return compressed, decompressed, compress_time, decompress_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/spike6/wildchat_en_content.content.bin")
    ap.add_argument("--url", default=DEFAULT_ENDPOINT)
    ap.add_argument("--limit-mb", type=float, default=10.0,
                    help="Limit input to this many MB (default 10)")
    args = ap.parse_args()

    with open(args.file, "rb") as f:
        raw = f.read()

    limit = int(args.limit_mb * 1024 * 1024)
    raw = raw[:limit]
    print(f"Input: {len(raw):,} bytes ({len(raw)/1024/1024:.1f} MB) from {args.file}")

    # Check server is up
    try:
        with urllib.request.urlopen(f"{args.url}/readyz") as r:
            print(f"Server: {r.read().decode()}")
    except Exception as e:
        print(f"ERROR: server not ready at {args.url}: {e}", file=sys.stderr)
        sys.exit(1)

    compressed, decompressed, ct, dt = roundtrip(args.url, raw)

    ratio = len(compressed) / len(raw)
    ok = decompressed == raw
    compress_kb_s = len(raw) / 1024 / ct
    decompress_kb_s = len(raw) / 1024 / dt

    print(f"\nResults:")
    print(f"  Compressed:   {len(compressed):,} bytes")
    print(f"  Ratio:        {ratio:.4f}  ({ratio*100:.1f}% of original)")
    print(f"  Compress:     {ct:.2f}s  ({compress_kb_s:.0f} KB/s)")
    print(f"  Decompress:   {dt:.2f}s  ({decompress_kb_s:.0f} KB/s)")
    print(f"  Byte-exact:   {'PASS ✓' if ok else 'FAIL ✗'}")

    if not ok:
        orig_hash = hashlib.sha256(raw).hexdigest()[:16]
        dec_hash = hashlib.sha256(decompressed).hexdigest()[:16]
        print(f"  orig sha256:  {orig_hash}...")
        print(f"  dec  sha256:  {dec_hash}...")
        sys.exit(1)

    # Gate checks (from V1_PLAN.md)
    gates = [
        (ratio <= 0.165, f"ratio {ratio:.4f} ≤ 0.165"),
        (compress_kb_s >= 300, f"compress {compress_kb_s:.0f} KB/s ≥ 300 KB/s"),
    ]
    print("\nGate checks:")
    all_pass = True
    for passed, label in gates:
        print(f"  {'PASS' if passed else 'FAIL'} {label}")
        all_pass = all_pass and passed

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

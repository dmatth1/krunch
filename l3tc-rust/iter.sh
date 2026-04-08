#!/usr/bin/env bash
# Quick iteration loop: rebuild, compress, decompress, verify.
#
# Usage:
#     ./iter.sh                     # 50 KB corpus (fast)
#     ./iter.sh /tmp/e6_50k.txt     # explicit corpus
#     INPUT=bench/corpora/enwik6 ./iter.sh   # full 1 MB enwik6
#
# Prints per-phase timing and the end-to-end round trip status.

set -euo pipefail

INPUT="${1:-${INPUT:-/tmp/e6_50k.txt}}"
OUT_L3TC="/tmp/iter.l3tc"
OUT_TXT="/tmp/iter.out"

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== build ==="
cargo build --release 2>&1 | grep -E "^(warning|error|Compiling|Finished)" || true

echo
echo "=== compress ${INPUT} ==="
./target/release/l3tc compress "$INPUT" -o "$OUT_L3TC" --time

echo
echo "=== decompress ==="
./target/release/l3tc decompress "$OUT_L3TC" -o "$OUT_TXT" --time

echo
echo "=== verify ==="
if diff -q "$INPUT" "$OUT_TXT" > /dev/null; then
    echo "OK — byte-identical round trip"
else
    echo "FAIL — output differs from input"
    exit 1
fi

orig=$(wc -c < "$INPUT" | tr -d ' ')
comp=$(wc -c < "$OUT_L3TC" | tr -d ' ')
ratio=$(awk "BEGIN { printf \"%.4f\", $comp / $orig }")
echo "  ratio: $ratio  ($comp / $orig bytes)"

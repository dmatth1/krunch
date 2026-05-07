#!/bin/bash
# Reproduce the README's ratio-comparison numbers on a local corpus.
#
# Runs zstd at max level (`-22 --long=27` = 128 MB window — the de-facto
# strong-text baseline) on the input, then if KRUNCH_RATIO is provided
# (or you've run `krunch compress < input > input.krunch` separately
# and pass --krunch-out=input.krunch) prints both ratios + the
# percentage advantage.
#
# Usage:
#   scripts/bench_ratio.sh INPUT [--krunch-out KRUNCH_FILE]
#   KRUNCH_RATIO=0.114 scripts/bench_ratio.sh INPUT
#
# Requires: zstd (apt install zstd / brew install zstd).
#
# Corpora used in the README ratio table — fetch into data/<name>/
# (gitignored) before running. README says ~100 MB per corpus where
# the upstream supports it.
#
#   data/wildchat-en/wildchat_en_content.content.bin
#       — Allen AI WildChat-1M (English subset). Repo-internal at
#         data/spike6/wildchat_en_content.content.bin.
#
#   data/enwik8/enwik8.bin    (100 MB exactly)
#       curl -L https://mattmahoney.net/dc/enwik8.zip -o /tmp/enwik8.zip
#       unzip /tmp/enwik8.zip -d data/enwik8/
#       mv data/enwik8/enwik8 data/enwik8/enwik8.bin
#
#   data/http_logs/nasa_apache_100mb.bin    (NASA Apache logs, 1995)
#       curl -L https://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz -o /tmp/jul.gz
#       curl -L https://ita.ee.lbl.gov/traces/NASA_access_log_Aug95.gz -o /tmp/aug.gz
#       gunzip /tmp/jul.gz /tmp/aug.gz
#       cat /tmp/NASA_access_log_Jul95 /tmp/NASA_access_log_Aug95 \
#           | head -c 104857600 > data/http_logs/nasa_apache_100mb.bin
#
#   data/support_tickets/bitext_support_tickets.csv    (~19 MB)
#       python -c "from huggingface_hub import hf_hub_download; \
#         hf_hub_download(repo_id='bitext/Bitext-customer-support-llm-chatbot-training-dataset', \
#           filename='Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv', \
#           repo_type='dataset', local_dir='data/support_tickets/')"
#
#   data/code_python/python_code_100mb.bin    (CodeParrot Python, 100 MB)
#       python -c "from huggingface_hub import hf_hub_download; \
#         hf_hub_download(repo_id='codeparrot/codeparrot-clean', \
#           filename='file-000000000001.json.gz', \
#           repo_type='dataset', local_dir='/tmp/')"
#       python -c "import gzip,json; \
#         out=open('data/code_python/python_code_100mb.bin','wb'); t=0; \
#         f=gzip.open('/tmp/file-000000000001.json.gz','rt'); \
#         [out.write((r['content']+chr(10)).encode('utf-8')) for r in (json.loads(L) for L in f) \
#          for _ in [0] if (t := t + len(r['content'])+1) <= 100_000_000]"

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 INPUT [--krunch-out KRUNCH_FILE]" >&2
  exit 1
fi

INPUT=$1
shift
KRUNCH_OUT=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --krunch-out) KRUNCH_OUT=$2; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -f $INPUT ]] || { echo "FAIL: input not found: $INPUT" >&2; exit 1; }
command -v zstd >/dev/null || { echo "FAIL: zstd not on PATH" >&2; exit 1; }

ORIG_BYTES=$(wc -c < "$INPUT")
echo "input:    $INPUT ($ORIG_BYTES bytes)"

# zstd reference run. -k keeps the input; -q suppresses the progress bar.
ZSTD_OUT="${INPUT}.zst"
rm -f "$ZSTD_OUT"
echo "running:  zstd -22 --long=27 -k -q $INPUT"
TIME_START=$(date +%s)
zstd -22 --long=27 -k -q "$INPUT"
ZSTD_WALL=$(( $(date +%s) - TIME_START ))
ZSTD_BYTES=$(wc -c < "$ZSTD_OUT")
ZSTD_RATIO=$(awk "BEGIN{printf \"%.4f\", $ZSTD_BYTES/$ORIG_BYTES}")
echo "zstd:     $ZSTD_BYTES bytes (ratio $ZSTD_RATIO, ${ZSTD_WALL}s wall)"

# krunch ratio comes from one of: --krunch-out (size of an existing
# .krunch file), $KRUNCH_RATIO env var (precomputed), or skipped.
KRUNCH_RATIO_OUT=""
if [[ -n $KRUNCH_OUT ]]; then
  [[ -f $KRUNCH_OUT ]] || { echo "FAIL: --krunch-out not found: $KRUNCH_OUT" >&2; exit 1; }
  KRUNCH_BYTES=$(wc -c < "$KRUNCH_OUT")
  KRUNCH_RATIO_OUT=$(awk "BEGIN{printf \"%.4f\", $KRUNCH_BYTES/$ORIG_BYTES}")
  echo "krunch:   $KRUNCH_BYTES bytes (ratio $KRUNCH_RATIO_OUT) ← from $KRUNCH_OUT"
elif [[ -n ${KRUNCH_RATIO:-} ]]; then
  KRUNCH_RATIO_OUT=$KRUNCH_RATIO
  echo "krunch:   ratio $KRUNCH_RATIO_OUT  ← from \$KRUNCH_RATIO"
fi

if [[ -n $KRUNCH_RATIO_OUT ]]; then
  ADVANTAGE=$(awk "BEGIN{printf \"%.1f\", (1 - $KRUNCH_RATIO_OUT/$ZSTD_RATIO) * 100}")
  echo
  echo "result:   krunch is ${ADVANTAGE}% better than zstd -22 --long=27"
  echo "          ($KRUNCH_RATIO_OUT vs $ZSTD_RATIO)"
fi

# Cleanup: leave the zstd output for inspection but make obvious where
# it came from.
echo
echo "(zstd output kept at $ZSTD_OUT — delete with: rm $ZSTD_OUT)"

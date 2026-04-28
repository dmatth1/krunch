#!/bin/bash
# Free, runs in seconds. Validates everything that doesn't need a GPU.
#
#   1. unit tests (header, AC codec, chunking, CRC, tokenizer)
#   2. CDK TypeScript type-check
#   3. CDK CloudFormation synth (proves the stack generates valid templates)
#   4. krunch submit --dry-run (proves the CLI builds correct Batch payloads)

set -euo pipefail
cd "$(dirname "$0")/.."

PASS=0
FAIL=0
SKIP=0
report() {
  if [[ $1 -eq 0 ]]; then echo "PASS $2"; PASS=$((PASS+1))
  elif [[ $1 -eq 99 ]]; then echo "SKIP $2"; SKIP=$((SKIP+1))
  else echo "FAIL $2"; FAIL=$((FAIL+1)); fi
}

echo "=== Krunch quick checks ==="

# 1. Unit tests
echo "[1/4] Unit tests"
if [[ -x /tmp/krunch-venv/bin/python ]]; then
  /tmp/krunch-venv/bin/python tests/test_blob.py >/dev/null
  report $? "unit tests (5/5)"
else
  report 99 "smoke test — /tmp/krunch-venv missing (run: python3 -m venv /tmp/krunch-venv && /tmp/krunch-venv/bin/pip install constriction tokenizers zstandard numpy httpx boto3)"
fi

# 2. CDK type-check
echo "[2/4] CDK TypeScript type-check"
( cd deploy/aws-cdk && npx --no-install tsc --noEmit ) 2>/dev/null
report $? "tsc --noEmit clean"

# 3. CDK synth
echo "[3/4] CDK CloudFormation synth"
( cd deploy/aws-cdk && CDK_DEFAULT_ACCOUNT=000000000000 CDK_DEFAULT_REGION=us-east-1 \
    npx --no-install cdk synth --quiet 2>&1 | tail -5 ) >/tmp/cdk-synth.log 2>&1
synth_rc=$?
if [[ $synth_rc -ne 0 ]]; then
  echo "--- cdk synth output:"
  cat /tmp/cdk-synth.log
fi
report $synth_rc "cdk synth"

# 4. krunch submit --dry-run
echo "[4/4] krunch submit --dry-run"
output=$(scripts/krunch submit --source s3://test/in --dest s3://test/out --workers 4 --dry-run 2>&1)
if [[ $? -eq 0 ]] && echo "$output" | grep -q '"workers": 4' && echo "$output" | grep -q "compress_job"; then
  report 0 "dry-run prints valid Batch payloads"
else
  echo "$output" | head -10
  report 1 "dry-run output malformed"
fi

echo
echo "=== $PASS passed, $FAIL failed, $SKIP skipped ==="
exit $FAIL

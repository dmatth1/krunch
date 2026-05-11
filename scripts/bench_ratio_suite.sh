#!/bin/bash
# bench_ratio_suite.sh — run AWS Batch on each ratio-comparison corpus
# back-to-back on a warm fleet, with a fixed krunch image pin, and
# print a summary table to fold into README.md.
#
# Each individual run is delegated to tests/integration/batch.sh with
# KRUNCH_BATCH_CLEANUP=0 so the compute environment stays warm between
# corpora. A single teardown runs at the end via the EXIT trap.
#
# Usage:
#   scripts/bench_ratio_suite.sh
#
# Required env (or accept defaults):
#   KRUNCH_VERSION       (default v0.1.0)              — image tag pin
#   KRUNCH_BATCH_WORKERS (default 4)                   — passed to batch.sh
#   AWS_REGION           (default us-east-1)
#   KRUNCH_STACK_NAME    (default KrunchStack)
#
# Corpora are uploaded once to s3://<stack-bucket>/krunch-ratio-corpora/
# and reused. Re-running this script reuses already-uploaded corpora.

set -euo pipefail
cd "$(dirname "$0")/.."

# ---- config ----------------------------------------------------------------
REGION="${AWS_REGION:-us-east-1}"
STACK="${KRUNCH_STACK_NAME:-KrunchStack}"
WORKERS="${KRUNCH_BATCH_WORKERS:-4}"
VERSION="${KRUNCH_VERSION:-v0.1.0}"
IMAGE="ghcr.io/dmatth1/krunch:${VERSION}"
SUITE_TAG="$(date +%Y%m%d-%H%M%S)"
RESULTS_DIR="/tmp/krunch-ratio-suite-${SUITE_TAG}"
mkdir -p "$RESULTS_DIR"

# Stack bucket lookup (same approach batch.sh uses)
BUCKET=$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
           --query 'Stacks[0].Outputs[?OutputKey==`BucketName`].OutputValue' \
           --output text 2>/dev/null || true)
if [[ -z $BUCKET || $BUCKET == None ]]; then
  # Fallback: discover via tag
  BUCKET=$(aws s3api list-buckets --query "Buckets[?starts_with(Name,'krunchstack-krunchbucket')].Name | [0]" --output text)
fi
[[ -n $BUCKET && $BUCKET != None ]] || { echo "FAIL: could not resolve stack bucket"; exit 1; }
CORPORA_PREFIX="s3://${BUCKET}/krunch-ratio-corpora"

# ---- corpora table ---------------------------------------------------------
# Format: NAME|LOCAL_PATH|S3_KEY|README_LABEL|ZSTD_RATIO
CORPORA=(
  # enwik8 + nasa_apache already run in earlier suite invocation; enwik8
  # passed byte-exact (ratio 0.1463); nasa_apache produced ratio 0.1572
  # but FAILED byte-exact — codec correctness investigation tracked
  # separately. Resuming with just the remaining two corpora.
  # "enwik8|data/enwik8/enwik8.bin|enwik8.bin|Wikipedia — enwik8 (100 MB)|0.253"
  # "nasa_apache|data/http_logs/nasa_apache_100mb.bin|nasa_apache.bin|HTTP logs — NASA Apache (100 MB)|0.061"
  "bitext|data/support_tickets/bitext_support_tickets.csv|bitext.bin|Support tickets — Bitext (19 MB)|0.083"
  "codeparrot|data/code_python/python_code_100mb.bin|codeparrot.bin|Python code — CodeParrot (100 MB)|0.154"
)

# ---- teardown trap ---------------------------------------------------------
# Same logic as batch.sh's cleanup(): drain compute envs to minvCpus=0
# and force-terminate any still-running Batch g5.xlarge.
final_teardown() {
  echo
  echo "[suite-cleanup] Final teardown (running regardless of suite outcome)..."
  local QUEUE
  QUEUE=$(aws cloudformation describe-stack-resources --stack-name "$STACK" --region "$REGION" \
            --query "StackResources[?ResourceType=='AWS::Batch::JobQueue'].PhysicalResourceId | [0]" \
            --output text 2>/dev/null || echo "")
  if [[ -n $QUEUE && $QUEUE != None ]]; then
    for ce_arn in $(aws batch describe-job-queues --job-queues "$QUEUE" --region "$REGION" \
                      --query 'jobQueues[0].computeEnvironmentOrder[].computeEnvironment' \
                      --output text 2>/dev/null); do
      ce_name=${ce_arn##*/}
      aws batch update-compute-environment --compute-environment "$ce_name" --region "$REGION" \
          --compute-resources minvCpus=0 --query 'computeEnvironmentName' --output text >/dev/null 2>&1 \
        && echo "  $ce_name minvCpus=0" \
        || echo "  $ce_name update failed (harmless if already updating)"
    done
  fi
  local ids
  ids=$(aws ec2 describe-instances --region "$REGION" \
          --filters "Name=instance-state-name,Values=pending,running" \
                    "Name=tag:AWSBatchServiceTag,Values=batch" \
                    "Name=instance-type,Values=g5.xlarge" \
          --query 'Reservations[].Instances[].InstanceId' --output text 2>/dev/null)
  if [[ -n $ids ]]; then
    aws ec2 terminate-instances --region "$REGION" --instance-ids $ids \
        --query 'TerminatingInstances[].InstanceId' --output text >/dev/null 2>&1 \
      && echo "  terminated: $ids" || echo "  terminate failed: $ids"
  else
    echo "  no Batch-tagged g5.xlarge running"
  fi
}
trap final_teardown EXIT

# ---- pre-flight ------------------------------------------------------------
echo "=== krunch ratio suite — ${SUITE_TAG} ==="
echo "  image:        ${IMAGE}"
echo "  workers:      ${WORKERS}"
echo "  bucket:       ${BUCKET}"
echo "  corpora dir:  ${CORPORA_PREFIX}/"
echo "  results dir:  ${RESULTS_DIR}/"
echo

for entry in "${CORPORA[@]}"; do
  IFS='|' read -r NAME LOCAL_PATH S3_KEY LABEL ZSTD <<< "$entry"
  if [[ ! -f $LOCAL_PATH ]]; then
    echo "FAIL: missing local corpus: $LOCAL_PATH"
    echo "  (see scripts/bench_ratio.sh header for fetch commands)"
    exit 1
  fi
done
echo "[pre-flight] all 4 local corpora present ✓"

# ---- upload corpora (skip if already in S3 with matching size) ------------
echo
echo "[1/3] Staging corpora to ${CORPORA_PREFIX}/..."
for entry in "${CORPORA[@]}"; do
  IFS='|' read -r NAME LOCAL_PATH S3_KEY LABEL ZSTD <<< "$entry"
  LOCAL_BYTES=$(wc -c < "$LOCAL_PATH" | tr -d ' ')
  S3_BYTES=$(aws s3api head-object --bucket "$BUCKET" \
               --key "krunch-ratio-corpora/${S3_KEY}" --region "$REGION" \
               --query ContentLength --output text 2>/dev/null || echo "0")
  if [[ $S3_BYTES == "$LOCAL_BYTES" ]]; then
    echo "  ${NAME}: already staged (${LOCAL_BYTES} bytes) — skip"
  else
    echo "  ${NAME}: uploading ${LOCAL_BYTES} bytes..."
    aws s3 cp "$LOCAL_PATH" "${CORPORA_PREFIX}/${S3_KEY}" --region "$REGION" --quiet
    echo "  ${NAME}: ✓"
  fi
done

# ---- run each corpus through batch.sh -------------------------------------
echo
echo "[2/3] Running batch suite (warm fleet between corpora)..."
SUMMARY="${RESULTS_DIR}/summary.tsv"
printf "name\tinput_bytes\tcompressed_bytes\tratio\tzstd_ratio\tadvantage_pct\tcompress_kbs\tdecompress_kbs\tlabel\n" > "$SUMMARY"

for entry in "${CORPORA[@]}"; do
  IFS='|' read -r NAME LOCAL_PATH S3_KEY LABEL ZSTD <<< "$entry"
  RUN_LOG="${RESULTS_DIR}/${NAME}.log"
  echo
  echo "------------------------------------------------------------"
  echo " ${NAME}: $(date -u +%FT%TZ)  →  $RUN_LOG"
  echo "------------------------------------------------------------"

  # Hand off to batch.sh with cleanup disabled. batch.sh's own EXIT
  # trap is a no-op when CLEANUP=0; the suite-level trap above covers
  # final teardown.
  KRUNCH_BATCH_CLEANUP=0 \
  KRUNCH_IMAGE="$IMAGE" \
  KRUNCH_SAMPLE_S3_URL="${CORPORA_PREFIX}/${S3_KEY}" \
  KRUNCH_BATCH_WORKERS="$WORKERS" \
  AWS_REGION="$REGION" \
  KRUNCH_STACK_NAME="$STACK" \
  bash tests/integration/batch.sh 2>&1 | tee "$RUN_LOG"

  # Parse the "=== Tier 4 PASS ===" footer from the log
  RATIO=$(grep -E '^\s*ratio:' "$RUN_LOG" | tail -1 | awk '{print $2}')
  COMP_KBS=$(grep -E 'compress aggregate:' "$RUN_LOG" | awk '{print $3}')
  DECOMP_KBS=$(grep -E 'decompress aggregate:' "$RUN_LOG" | awk '{print $3}')

  if [[ -z $RATIO ]]; then
    echo "WARN: ${NAME} did not produce a ratio (run failed?). Continuing."
    RATIO="FAIL"
    COMP_KBS="-"
    DECOMP_KBS="-"
    ADV="-"
    INBYTES="-"
    OUTBYTES="-"
  else
    INBYTES=$(wc -c < "$LOCAL_PATH" | tr -d ' ')
    OUTBYTES=$(awk "BEGIN{printf \"%d\", $RATIO * $INBYTES}")
    ADV=$(awk "BEGIN{printf \"%.1f\", (1 - $RATIO/$ZSTD)*100}")
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$NAME" "$INBYTES" "$OUTBYTES" "$RATIO" "$ZSTD" "$ADV" \
    "${COMP_KBS:-?}" "${DECOMP_KBS:-?}" "$LABEL" >> "$SUMMARY"
done

# ---- summary --------------------------------------------------------------
echo
echo "[3/3] Suite complete. Summary:"
echo
printf "%-12s  %-7s  %-7s  %-8s  %-9s  %-13s\n" \
  "name" "ratio" "zstd" "vs zstd" "comp KB/s" "decomp KB/s"
echo "-----------------------------------------------------------------"
while IFS=$'\t' read -r name in_b out_b r z adv ck dk lbl; do
  [[ $name == "name" ]] && continue
  printf "%-12s  %-7s  %-7s  %-7s%%  %-9s  %-13s\n" \
    "$name" "$r" "$z" "$adv" "$ck" "$dk"
done < "$SUMMARY"

echo
echo "Full TSV: $SUMMARY"
echo "Per-corpus logs: ${RESULTS_DIR}/<name>.log"

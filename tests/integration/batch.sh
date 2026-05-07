#!/bin/bash
# Tier 4 — end-to-end AWS Batch fan-out roundtrip.
#
# Validates the distributed path that single-instance tests/integration/gpu.sh can't
# reach: `krunch plan --target aws-batch` rendering, GPU array job
# execution, finalize-stitch dependency, and byte-exact decompress
# roundtrip. Same exact CLI path a real user runs.
#
# Pre-flight:
#   - The KrunchStack CDK stack must be deployed (deploy/aws-cdk).
#     Provides: JobQueueArn, CompressJobDefOutput, DecompressJobDefOutput,
#     FinalizeJobDefOutput, BucketName outputs.
#   - aws CLI configured with Batch + S3 + CloudFormation read perms.
#   - jq available locally.
#   - docker available locally (used to invoke `krunch plan` from image).
#
# Sample: defaults to the 100 MB WildChat slice at
#   s3://dmatth1-bnn-checkpoints/krunch-tier3/20260428-105803/sample.bin
# Override with KRUNCH_SAMPLE_S3_URL=s3://your-bucket/path/to/sample.bin.
# The script copies the sample into the deployed KrunchStack's bucket
# (which the Batch job role can read) — so the source bucket only needs
# to be readable by your local AWS credentials, not the Batch role.
#
# Cost: ~$0.30/hr × 4 g5.xlarge × ~5 min = ~$0.10 plus a few cents of S3.
# Spot interruption: jobs retry once via JobDefinition retryStrategy.
#
# Gates: byte-exact roundtrip, ratio matches single-instance gpu.sh,
# finalize task succeeds, parts cleaned up by finalize.
#
# Dry-run mode: KRUNCH_BATCH_DRY_RUN=1 walks the whole script —
# resolves stack outputs, renders both compress + decompress plans via
# `krunch plan`, prints the submit-job calls that *would* run, but does
# NOT actually submit, copy data, or download results. Use this to
# de-risk batch.sh changes without spending: catches stack-output
# typos, plan-rendering arg-passthrough bugs, and dependsOn wiring
# errors. Cost: $0 (only AWS calls are describe-stacks).

set -euo pipefail
cd "$(dirname "$0")/../.."

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REGION="${AWS_REGION:-us-east-1}"
STACK="${KRUNCH_STACK_NAME:-KrunchStack}"
WORKERS="${KRUNCH_BATCH_WORKERS:-4}"
SAMPLE_S3_URL="${KRUNCH_SAMPLE_S3_URL:-s3://dmatth1-bnn-checkpoints/krunch-tier3/20260428-105803/sample.bin}"
KRUNCH_IMAGE_TAG="${KRUNCH_IMAGE:-ghcr.io/dmatth1/krunch:latest}"
TEST_TAG="$(date +%Y%m%d-%H%M%S)"
POLL_INTERVAL="${KRUNCH_POLL_INTERVAL:-15}"
DRY_RUN="${KRUNCH_BATCH_DRY_RUN:-0}"
# Spin everything down on exit (success OR failure) so we don't accrue
# idle costs after the test. Override with KRUNCH_BATCH_CLEANUP=0 to
# keep instances warm (e.g., between back-to-back test runs).
CLEANUP="${KRUNCH_BATCH_CLEANUP:-1}"

# ---------------------------------------------------------------------------
# Cleanup — runs on EXIT (any path). Sets every CE in the queue to
# minvCpus=0 (so ASG drains naturally) AND force-terminates any Batch-
# tagged g5.xlarge running right now (so we don't wait on ASG cooldown).
# Both are safe no-ops if there's nothing to clean up.
# ---------------------------------------------------------------------------
cleanup() {
  [[ $CLEANUP != 1 ]] && return 0
  echo
  echo "[cleanup] Spinning down compute envs + terminating workers..."
  if [[ -n "${QUEUE:-}" ]]; then
    for ce_arn in $(aws batch describe-job-queues --job-queues "$QUEUE" \
                      --region "$REGION" \
                      --query 'jobQueues[0].computeEnvironmentOrder[].computeEnvironment' \
                      --output text 2>/dev/null); do
      local ce_name=${ce_arn##*/}
      aws batch update-compute-environment --compute-environment "$ce_name" \
          --region "$REGION" --compute-resources minvCpus=0 \
          --query 'computeEnvironmentName' --output text >/dev/null 2>&1 \
        && echo "  $ce_name minvCpus=0" \
        || echo "  $ce_name update failed (may be already updating; harmless)"
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
      && echo "  terminated: $ids" \
      || echo "  ec2 terminate-instances failed for: $ids"
  else
    echo "  no Batch-tagged g5.xlarge instances running"
  fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Pre-flight: tools + sample + stack
# ---------------------------------------------------------------------------
for tool in aws jq docker; do
  command -v "$tool" >/dev/null || { echo "FAIL missing required tool: $tool"; exit 1; }
done

echo "=== Tier 4 AWS Batch end-to-end roundtrip — ${TEST_TAG} ==="
echo "  region:       ${REGION}"
echo "  stack:        ${STACK}"
echo "  workers:      ${WORKERS} (compress + decompress array size)"
echo "  sample:       ${SAMPLE_S3_URL}"
echo "  image:        ${KRUNCH_IMAGE_TAG}"
[[ $DRY_RUN == 1 ]] && echo "  *** DRY RUN — no submit-job, no S3 staging, no downloads ***"

# Resolve stack outputs
echo "Resolving CloudFormation stack outputs..."
out_json=$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
            --query 'Stacks[0].Outputs' --output json)
QUEUE=$(echo "$out_json" | jq -r '.[] | select(.OutputKey=="JobQueueArn") | .OutputValue')
COMPRESS_JD=$(echo "$out_json" | jq -r '.[] | select(.OutputKey=="CompressJobDefOutput") | .OutputValue')
DECOMPRESS_JD=$(echo "$out_json" | jq -r '.[] | select(.OutputKey=="DecompressJobDefOutput") | .OutputValue')
FINALIZE_JD=$(echo "$out_json" | jq -r '.[] | select(.OutputKey=="FinalizeJobDefOutput") | .OutputValue')
BUCKET=$(echo "$out_json" | jq -r '.[] | select(.OutputKey=="BucketName") | .OutputValue')

for v in "$QUEUE" "$COMPRESS_JD" "$DECOMPRESS_JD" "$FINALIZE_JD" "$BUCKET"; do
  [[ -n $v && $v != null ]] || { echo "FAIL stack output missing — redeploy KrunchStack"; exit 1; }
done

S3_BASE="s3://${BUCKET}/krunch-tier4/${TEST_TAG}"
SRC_URL="${S3_BASE}/sample.bin"
COMP_URL="${S3_BASE}/sample.krunch"
DECOMP_URL="${S3_BASE}/sample.roundtrip"
echo "  queue:        ${QUEUE##*/}"
echo "  outputs at:   ${S3_BASE}"

# ---------------------------------------------------------------------------
# Stage sample into the stack bucket (S3-to-S3 copy, server-side, fast)
# ---------------------------------------------------------------------------
echo
echo "[1/6] Staging sample into ${SRC_URL}..."
ORIG_BUCKET=$(echo "$SAMPLE_S3_URL" | sed -E 's|^s3://([^/]+)/.*|\1|')
ORIG_KEY=$(echo "$SAMPLE_S3_URL"    | sed -E 's|^s3://[^/]+/(.*)|\1|')
INPUT_LEN=$(aws s3api head-object --region "$REGION" \
              --bucket "$ORIG_BUCKET" --key "$ORIG_KEY" \
              --query ContentLength --output text 2>/dev/null) \
  || { echo "FAIL cannot HEAD ${SAMPLE_S3_URL} — check your local AWS credentials"; exit 1; }
echo "  source ${SAMPLE_S3_URL} is ${INPUT_LEN} bytes ($((INPUT_LEN / 1024 / 1024)) MB)"
if [[ $DRY_RUN == 1 ]]; then
  echo "  [dry-run] would: aws s3 cp ${SAMPLE_S3_URL} ${SRC_URL}"
else
  aws s3 cp --quiet --region "$REGION" "$SAMPLE_S3_URL" "$SRC_URL"
  echo "  copied to ${SRC_URL} (Batch job role has GetObject on ${BUCKET})"
fi

# ---------------------------------------------------------------------------
# Helper — render plan + submit two jobs (main array + finalize), return finalize id
# ---------------------------------------------------------------------------
poll_job() {
  local job_id=$1
  local label=$2
  local last=""
  while true; do
    local resp status arr line
    resp=$(aws batch describe-jobs --region "$REGION" --jobs "$job_id" --output json)
    status=$(echo "$resp" | jq -r '.jobs[0].status')
    arr=$(echo "$resp" | jq -r '.jobs[0].arrayProperties.statusSummary // {} | "p=\(.PENDING // 0)+\(.RUNNABLE // 0)+\(.STARTING // 0) r=\(.RUNNING // 0) s=\(.SUCCEEDED // 0) f=\(.FAILED // 0)"')
    line="  ${label}: ${status} ${arr}"
    [[ $line != "$last" ]] && { echo "$line"; last=$line; }
    case $status in
      SUCCEEDED) return 0 ;;
      FAILED)    echo "FAIL ${label} terminal status FAILED"; echo "$resp" | jq '.jobs[0].statusReason, .jobs[0].attempts'; return 1 ;;
    esac
    sleep "$POLL_INTERVAL"
  done
}

submit_pair() {
  local mode=$1            # compress | decompress
  local jd=$2              # array job definition arn
  local input_url=$3
  local output_url=$4
  local input_len=$5
  local plan_json
  plan_json=$(mktemp)
  # Invoke `krunch plan` via the host wrapper — the same entry point a real
  # user gets from install.sh. Bypassing it would skip arg-passthrough +
  # env-handling that the wrapper does.
  # `krunch plan` auto-resolves --queue / --job-definition /
  # --finalize-job-definition from the deployed CDK stack outputs when
  # --target=aws-batch — no need to thread them in here.
  KRUNCH_IMAGE="$KRUNCH_IMAGE_TAG" scripts/krunch plan \
    --target aws-batch --mode "$mode" \
    --source "$input_url" --dest "$output_url" \
    --workers "$WORKERS" --input-len "$input_len" \
    --image "$KRUNCH_IMAGE_TAG" \
    --run-id "${TEST_TAG}-${mode}" > "$plan_json"

  if [[ $DRY_RUN == 1 ]]; then
    echo "  [dry-run] rendered ${mode} plan to ${plan_json}" >&2
    echo "  [dry-run] .main env (would be submitted to ${jd##*/}):" >&2
    jq '.main.containerOverrides.environment' "$plan_json" >&2
    echo "  [dry-run] .finalize would submit with: " \
         "--depends-on jobId=<main-id>,type=SEQUENTIAL" >&2
    echo "  [dry-run] .finalize env:" >&2
    jq '.finalize.containerOverrides.environment' "$plan_json" >&2
    rm -f "$plan_json"
    return 0
  fi

  # AWS CLI's --cli-input-json file:/// requires a real path
  # (file:///dev/stdin parses unreliably across CLI versions). Stage
  # each spec to a temp file.
  local main_spec finalize_spec
  main_spec=$(mktemp)
  finalize_spec=$(mktemp)
  # AWS Batch requires arrayProperties.size >= 2. For workers=1 we
  # have to submit as a non-array job and replace the Ref:: array-
  # index substitution with a literal "0". This IS a real AWS Batch
  # API limitation that real-user tooling has to work around.
  if [[ $WORKERS -eq 1 ]]; then
    jq -c '.main
              | del(.arrayProperties)
              | .containerOverrides.environment |= map(
                  if .name == "KRUNCH_PART_INDEX" then .value = "0" else . end)' \
        "$plan_json" > "$main_spec"
  else
    jq -c '.main' "$plan_json" > "$main_spec"
  fi
  jq -c '.finalize' "$plan_json" > "$finalize_spec"

  local main_id finalize_id
  main_id=$(aws batch submit-job --region "$REGION" \
              --cli-input-json "file://${main_spec}" --query jobId --output text)
  if [[ -z $main_id || $main_id == None ]]; then
    echo "FAIL ${mode} submit-job returned no jobId; spec was:" >&2
    cat "$main_spec" >&2
    return 1
  fi
  echo "  ${mode} array submitted: ${main_id}" >&2
  poll_job "$main_id" "${mode}-array" >&2 || return 1

  finalize_id=$(aws batch submit-job --region "$REGION" \
                  --cli-input-json "file://${finalize_spec}" \
                  --depends-on "jobId=${main_id},type=SEQUENTIAL" \
                  --query jobId --output text)
  if [[ -z $finalize_id || $finalize_id == None ]]; then
    echo "FAIL ${mode} finalize submit-job returned no jobId; spec was:" >&2
    cat "$finalize_spec" >&2
    return 1
  fi
  echo "  ${mode} finalize submitted: ${finalize_id}" >&2
  poll_job "$finalize_id" "${mode}-finalize" >&2 || return 1

  # Expose the array/main job id to the caller for post-run log parsing.
  LAST_MAIN_ID=$main_id
  rm -f "$plan_json" "$main_spec" "$finalize_spec"
}

# ---------------------------------------------------------------------------
# parse_work_times — pull KRUNCH_WORK_TIME log lines from CloudWatch for
# each child (or single non-array job) and report per-worker real-work
# numbers + aggregate.
#
# AWS-specific (uses /aws/batch/job log group + describe-jobs to get
# log stream names). For non-AWS orchestrators this would need a
# different impl — see TIER_4.md "Real-work measurement" section.
# ---------------------------------------------------------------------------
parse_work_times() {
  local mode=$1       # "compress" | "decompress"
  local parent_id=$2
  local n=$3          # workers count
  local total_bytes=0 max_elapsed=0 streams=()

  if [[ $n -eq 1 ]]; then
    streams+=( "$(aws batch describe-jobs --region "$REGION" --jobs "$parent_id" \
                  --query 'jobs[0].attempts[-1].container.logStreamName' --output text 2>/dev/null)" )
  else
    for i in $(seq 0 $((n-1))); do
      streams+=( "$(aws batch describe-jobs --region "$REGION" --jobs "${parent_id}:${i}" \
                    --query 'jobs[0].attempts[-1].container.logStreamName' --output text 2>/dev/null)" )
    done
  fi

  echo "  --- $mode: real-work timing per worker (excludes cold start, AWS-specific) ---"
  for s in "${streams[@]}"; do
    [[ -z $s || $s == None ]] && { echo "    (no log stream)"; continue; }
    local line bytes elapsed rate part nch
    line=$(aws logs filter-log-events --region "$REGION" --log-group-name /aws/batch/job \
            --log-stream-names "$s" --filter-pattern '"KRUNCH_WORK_TIME"' \
            --query 'events[0].message' --output text 2>/dev/null)
    if [[ -z $line || $line == None ]]; then
      echo "    stream=${s##*/}: no KRUNCH_WORK_TIME line yet (image may pre-date instrumentation?)"
      continue
    fi
    bytes=$(  echo "$line" | sed -nE 's/.*bytes=([0-9]+).*/\1/p')
    elapsed=$(echo "$line" | sed -nE 's/.*elapsed=([0-9.]+).*/\1/p')
    rate=$(   echo "$line" | sed -nE 's/.*rate_kbps=([0-9.]+).*/\1/p')
    part=$(   echo "$line" | sed -nE 's/.*part=([0-9]+).*/\1/p')
    nch=$(    echo "$line" | sed -nE 's/.*n_chunks=([0-9]+).*/\1/p')
    printf "    part=%s  bytes=%s  chunks=%s  real=%ss  per-worker=%s KB/s\n" \
           "$part" "$bytes" "$nch" "$elapsed" "$rate"
    total_bytes=$((total_bytes + bytes))
    awk_cmp=$(awk "BEGIN{print ($elapsed > $max_elapsed) ? 1 : 0}")
    [[ $awk_cmp == 1 ]] && max_elapsed=$elapsed
  done
  if [[ $total_bytes -gt 0 ]]; then
    local agg
    agg=$(awk "BEGIN{printf \"%.1f\", ($total_bytes/1024)/$max_elapsed}")
    printf "    AGGREGATE: %s bytes / %ss max-wall = %s KB/s real work\n" \
           "$total_bytes" "$max_elapsed" "$agg"
  fi
}

# ---------------------------------------------------------------------------
# Compress
# ---------------------------------------------------------------------------
echo
echo "[2/6] Submitting compress (workers=${WORKERS})..."
t0=$(date +%s)
submit_pair compress "$COMPRESS_JD" "$SRC_URL" "$COMP_URL" "$INPUT_LEN" || exit 1
COMPRESS_S=$(( $(date +%s) - t0 ))

# Use a placeholder COMP_LEN under dry-run so step [4] still renders the
# decompress plan; the value never reaches a Batch worker.
if [[ $DRY_RUN == 1 ]]; then
  echo
  echo "[3/6] Compress stats — skipped (dry-run, no compressed artifact in S3)"
  COMP_LEN=$INPUT_LEN  # placeholder for the decompress plan render
else
  echo
  echo "[3/6] Compress stats..."
  COMP_LEN=$(aws s3api head-object --region "$REGION" --bucket "$BUCKET" \
              --key "${COMP_URL#s3://${BUCKET}/}" --query ContentLength --output text)
  RATIO=$(awk "BEGIN{printf \"%.4f\", $COMP_LEN/$INPUT_LEN}")
  COMP_KBS=$(awk "BEGIN{printf \"%.1f\", ($INPUT_LEN/1024)/${COMPRESS_S}}")
  echo "  original:    ${INPUT_LEN} bytes"
  echo "  compressed:  ${COMP_LEN} bytes"
  echo "  ratio:       ${RATIO}"
  echo "  wall:        ${COMPRESS_S}s aggregate (${COMP_KBS} KB/s including cold start)"
  COMPRESS_PARENT_ID=$LAST_MAIN_ID
  parse_work_times compress "$COMPRESS_PARENT_ID" "$WORKERS"
fi

# ---------------------------------------------------------------------------
# Decompress
# ---------------------------------------------------------------------------
echo
echo "[4/6] Submitting decompress (workers=${WORKERS})..."
t0=$(date +%s)
submit_pair decompress "$DECOMPRESS_JD" "$COMP_URL" "$DECOMP_URL" "$COMP_LEN" || exit 1
DECOMPRESS_S=$(( $(date +%s) - t0 ))

if [[ $DRY_RUN == 1 ]]; then
  echo
  echo "[5/6] Roundtrip verify — skipped (dry-run, no decompressed artifact in S3)"
  echo "[6/6] Parts cleanup verify — skipped (dry-run)"
  cat <<EOF

=== Tier 4 dry-run PASS ===
  Stack outputs resolved cleanly.
  Both compress + decompress plans rendered via 'krunch plan'.
  Submit-job calls printed; no AWS spend incurred (only describe-stacks).

To run for real:  unset KRUNCH_BATCH_DRY_RUN  &&  tests/integration/batch.sh
EOF
  exit 0
fi

DECOMP_LEN=$(aws s3api head-object --region "$REGION" --bucket "$BUCKET" \
              --key "${DECOMP_URL#s3://${BUCKET}/}" --query ContentLength --output text)
DECOMP_KBS=$(awk "BEGIN{printf \"%.1f\", ($DECOMP_LEN/1024)/${DECOMPRESS_S}}")
echo "  decompressed: ${DECOMP_LEN} bytes"
echo "  wall:         ${DECOMPRESS_S}s aggregate (${DECOMP_KBS} KB/s including cold start)"
DECOMPRESS_PARENT_ID=$LAST_MAIN_ID
parse_work_times decompress "$DECOMPRESS_PARENT_ID" "$WORKERS"

# ---------------------------------------------------------------------------
# Roundtrip byte-exact
# ---------------------------------------------------------------------------
echo
echo "[5/6] Verifying byte-exact roundtrip..."
ORIG_TMP=$(mktemp); ROUND_TMP=$(mktemp)
trap 'rm -f "$ORIG_TMP" "$ROUND_TMP"' EXIT
aws s3 cp --quiet --region "$REGION" "$SRC_URL"    "$ORIG_TMP"
aws s3 cp --quiet --region "$REGION" "$DECOMP_URL" "$ROUND_TMP"
ORIG_SHA=$(shasum -a 256 "$ORIG_TMP"   | awk '{print $1}')
ROUND_SHA=$(shasum -a 256 "$ROUND_TMP" | awk '{print $1}')
if [[ $ORIG_SHA != "$ROUND_SHA" ]]; then
  echo "FAIL roundtrip mismatch"
  echo "  original sha256:    $ORIG_SHA ($(wc -c <"$ORIG_TMP") bytes)"
  echo "  roundtrip sha256:   $ROUND_SHA ($(wc -c <"$ROUND_TMP") bytes)"
  exit 1
fi
echo "  PASS sha256 match (${ORIG_SHA:0:16}…)"

# ---------------------------------------------------------------------------
# Parts cleanup check
# ---------------------------------------------------------------------------
echo
echo "[6/6] Verifying parts cleanup..."
# `aws s3 ls` exits 1 on a missing prefix — that's the SUCCESS case
# here (cleanup worked, no parts left). Coerce the exit code so
# `set -euo pipefail` doesn't trip on it.
LEFTOVER=$( { aws s3 ls --recursive --region "$REGION" "${COMP_URL}.parts/" 2>/dev/null || true; } | wc -l | tr -d ' ')
LEFTOVER_DECOMP=$( { aws s3 ls --recursive --region "$REGION" "${DECOMP_URL}.parts/" 2>/dev/null || true; } | wc -l | tr -d ' ')
if [[ $LEFTOVER -ne 0 || $LEFTOVER_DECOMP -ne 0 ]]; then
  echo "FAIL leftover parts: compress=${LEFTOVER}, decompress=${LEFTOVER_DECOMP}"
  exit 1
fi
echo "  PASS no leftover parts in ${COMP_URL}.parts/ or ${DECOMP_URL}.parts/"

cat <<EOF

=== Tier 4 PASS ===
  ratio:                  ${RATIO}
  compress aggregate:     ${COMP_KBS} KB/s (${COMPRESS_S}s wall, ${WORKERS} workers, includes cold start)
  decompress aggregate:   ${DECOMP_KBS} KB/s (${DECOMPRESS_S}s wall, ${WORKERS} workers, includes cold start)
  roundtrip:              byte-exact
  parts cleanup:          ok

S3 artifacts (kept for inspection — delete with: aws s3 rm --recursive ${S3_BASE}/):
  ${SRC_URL}     (${INPUT_LEN} bytes original)
  ${COMP_URL}    (${COMP_LEN} bytes compressed)
  ${DECOMP_URL}  (${DECOMP_LEN} bytes roundtrip)
EOF

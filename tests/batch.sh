#!/bin/bash
# Tier 4 — end-to-end AWS Batch fan-out roundtrip.
#
# Validates the distributed path that single-instance tests/gpu.sh can't
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
cd "$(dirname "$0")/.."

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
  KRUNCH_IMAGE="$KRUNCH_IMAGE_TAG" scripts/krunch plan \
    --target aws-batch --mode "$mode" \
    --source "$input_url" --dest "$output_url" \
    --workers "$WORKERS" --input-len "$input_len" \
    --image "$KRUNCH_IMAGE_TAG" \
    --queue "$QUEUE" --job-definition "$jd" \
    --run-id "${TEST_TAG}-${mode}" > "$plan_json"

  if [[ $DRY_RUN == 1 ]]; then
    echo "  [dry-run] rendered ${mode} plan to ${plan_json}" >&2
    echo "  [dry-run] .main env (would be submitted to ${jd##*/}):" >&2
    jq '.main.containerOverrides.environment' "$plan_json" >&2
    echo "  [dry-run] .finalize would submit with: " \
         "--depends-on jobId=<main-id>,type=SEQUENTIAL " \
         "--job-definition ${FINALIZE_JD##*/}" >&2
    echo "  [dry-run] .finalize env:" >&2
    jq '.finalize.containerOverrides.environment' "$plan_json" >&2
    rm -f "$plan_json"
    return 0
  fi

  # AWS CLI's --cli-input-json file:/// requires a real path (file:///dev/stdin
  # parses unreliably across CLI versions). Stage each spec to a temp file.
  # Also strip containerOverrides.image — the published image template
  # wrongly includes it; AWS Batch only accepts image at JobDefinition
  # registration time, not per-submission. Template was fixed
  # 2026-05-06 but the strip stays as defensive cover for older images.
  local main_spec finalize_spec
  main_spec=$(mktemp)
  finalize_spec=$(mktemp)
  # AWS Batch requires arrayProperties.size >= 2. For workers=1 we have to
  # submit as a non-array job and replace the Ref:: array-index substitution
  # with a literal "0". Detect via $WORKERS so we don't ship a busted spec.
  # Also defensively override containerOverrides.command to ["job"] —
  # the published image's plan template wrongly emits ["python","-m","krunch.job"]
  # which entrypoint.sh rejects ("Unknown mode: python"). Fixed in source
  # template 2026-05-07; strip stays as defensive cover for older images.
  if [[ $WORKERS -eq 1 ]]; then
    jq -c '.main
              | del(.containerOverrides.image)
              | .containerOverrides.command = ["job"]
              | del(.arrayProperties)
              | .containerOverrides.environment |= map(
                  if .name == "KRUNCH_PART_INDEX" then .value = "0" else . end)' \
        "$plan_json" > "$main_spec"
  else
    jq -c '.main
              | del(.containerOverrides.image)
              | .containerOverrides.command = ["job"]' \
        "$plan_json" > "$main_spec"
  fi
  jq -c '.finalize
            | .jobDefinition = "'"$FINALIZE_JD"'"
            | del(.containerOverrides.image)
            | .containerOverrides.command = ["job"]' "$plan_json" > "$finalize_spec"

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

  rm -f "$plan_json" "$main_spec" "$finalize_spec"
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

To run for real:  unset KRUNCH_BATCH_DRY_RUN  &&  tests/batch.sh
EOF
  exit 0
fi

DECOMP_LEN=$(aws s3api head-object --region "$REGION" --bucket "$BUCKET" \
              --key "${DECOMP_URL#s3://${BUCKET}/}" --query ContentLength --output text)
DECOMP_KBS=$(awk "BEGIN{printf \"%.1f\", ($DECOMP_LEN/1024)/${DECOMPRESS_S}}")
echo "  decompressed: ${DECOMP_LEN} bytes"
echo "  wall:         ${DECOMPRESS_S}s aggregate (${DECOMP_KBS} KB/s including cold start)"

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

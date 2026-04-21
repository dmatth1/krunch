#!/usr/bin/env bash
# Krunch spike status CLI.
#
# Usage:
#   scripts/spike_status.sh <customerId> <datasetId>
#   scripts/spike_status.sh acme hdfs-spike
#
# Prints:
#   1. Dataset row from DynamoDB (state machine snapshot)
#   2. Latest N model versions from DynamoDB + their held_out ratios
#   3. In-flight AWS Batch training jobs for this dataset
#   4. Last 20 lines of training log (if a job is running or recent)
#   5. Any pending / in-flight compression messages

# intentionally NOT using set -e — we want to run every section even if
# a preceding AWS call returns non-zero (e.g. resource not deployed yet).
set -u

CID="${1:?customer id required}"
DSID="${2:?dataset id required}"
REGION="${AWS_REGION:-us-east-1}"
ENV="${ENV:-dev}"

DATASETS_TABLE="archive-${ENV}-datasets"
MODEL_VERSIONS_TABLE="archive-${ENV}-model-versions"
BUCKET="archive-${ENV}-archive"
JOB_QUEUE_NAME="archive-${ENV}-training-queue"
COMPRESSION_QUEUE_NAME="archive-${ENV}-compression"

sep() { printf '\n%s\n' "========================================================================"; }

sep
echo "dataset ${CID}/${DSID}  (env=${ENV}  region=${REGION})"
sep

echo "## 1. dataset row (DynamoDB)"
aws dynamodb get-item \
  --region "$REGION" \
  --table-name "$DATASETS_TABLE" \
  --key "{\"pk\":{\"S\":\"CUST#${CID}\"},\"sk\":{\"S\":\"DS#${DSID}\"}}" \
  --query 'Item' \
  --output json 2>/dev/null | python3 -c "
import json, sys
try:
  row = json.load(sys.stdin)
  if row is None:
    print('  (no row found — dataset does not exist yet)')
    sys.exit(0)
  def unwrap(v):
    if isinstance(v, dict) and len(v) == 1:
      (tag, val), = v.items()
      if tag == 'NULL' and val: return None
      if tag == 'N': return float(val) if '.' in val else int(val)
      if tag == 'S': return val
      if tag == 'BOOL': return val
    return v
  flat = {k: unwrap(v) for k, v in row.items()}
  for k in ('status','current_model_version','raw_bytes_held','compressed_bytes','total_events_ingested','last_training_started_at','last_training_completed_at','training_job_id'):
    if k in flat:
      print(f'  {k:30s} = {flat[k]}')
except Exception as e:
  print('  (parse failed:', e, ')')
  sys.exit(0)
"

sep
echo "## 2. model versions (DynamoDB)"
aws dynamodb query \
  --region "$REGION" \
  --table-name "$MODEL_VERSIONS_TABLE" \
  --key-condition-expression "pk = :pk" \
  --expression-attribute-values "{\":pk\":{\"S\":\"CUST#${CID}#DS#${DSID}\"}}" \
  --scan-index-forward false \
  --limit 5 \
  --query 'Items' \
  --output json 2>/dev/null | python3 -c "
import json, sys
try:
  text = sys.stdin.read().strip()
  rows = json.loads(text) if text else []
  if not rows or rows is None:
    print('  (no model versions yet)')
    sys.exit(0)
  def unwrap(v):
    if isinstance(v, dict) and len(v) == 1:
      (tag, val), = v.items()
      if tag == 'N': return float(val) if '.' in val else int(val)
      if tag == 'S': return val
    return v
  for r in rows:
    flat = {k: unwrap(v) for k, v in r.items()}
    v = flat.get('sk', '?')
    codec = flat.get('codec', '?')
    hr = flat.get('held_out_ratio', '?')
    zr = flat.get('zstd_baseline_ratio', '?')
    win = ''
    try:
      if isinstance(hr, (int, float)) and isinstance(zr, (int, float)) and zr > 0:
        win = f'  (our ratio / zstd ratio = {hr/zr:.3f})'
    except Exception: pass
    print(f'  v{v}  codec={codec}  held_out_ratio={hr}  zstd_baseline={zr}{win}')
except Exception as e:
  print('  (parse failed:', e, ')')
"

sep
echo "## 3. recent AWS Batch training jobs"
# aws batch list-jobs requires an exact status filter; we aggregate over
# all states. If the job queue doesn't exist yet (pre-deploy), skip.
if ! aws batch describe-job-queues --region "$REGION" --job-queues "$JOB_QUEUE_NAME" --query 'jobQueues[0].jobQueueArn' --output text >/dev/null 2>&1; then
  echo "  (job queue ${JOB_QUEUE_NAME} does not exist yet — training stack probably not deployed)"
  JOBS_JSON='[]'
else
  JOBS_JSON='[]'
  for st in SUBMITTED PENDING RUNNABLE STARTING RUNNING SUCCEEDED FAILED; do
    part=$(aws batch list-jobs --region "$REGION" --job-queue "$JOB_QUEUE_NAME" --job-status "$st" \
      --query 'jobSummaryList[*].[jobId,jobName,status,createdAt]' --output json 2>/dev/null || echo '[]')
    JOBS_JSON=$(python3 -c "
import json, sys
a = json.loads('''$JOBS_JSON''')
b = json.loads('''$part''')
print(json.dumps(a + b))
")
  done
fi
echo "$JOBS_JSON" | python3 -c "
import json, sys, datetime
rows = json.load(sys.stdin)
if not rows:
  print('  (no recent jobs on the training queue)')
  sys.exit(0)
rows.sort(key=lambda r: r[3] or 0, reverse=True)
for jid, jname, status, created_ms in rows[:5]:
  created = datetime.datetime.fromtimestamp((created_ms or 0)/1000).isoformat() if created_ms else '?'
  matches = '${CID}-${DSID}' in (jname or '')
  tag = '← this dataset' if matches else ''
  print(f'  {status:10s}  {jname:50s}  {created}  {tag}')
"

sep
echo "## 4. latest training log lines"
LATEST_JOB_ID=$(echo "$JOBS_JSON" | python3 -c "
import json, sys
rows = json.load(sys.stdin)
rows = [r for r in rows if '${CID}-${DSID}' in (r[1] or '')]
if not rows:
  print('')
else:
  rows.sort(key=lambda r: r[3] or 0, reverse=True)
  print(rows[0][0])
")
if [ -n "$LATEST_JOB_ID" ]; then
  echo "  most recent job id: $LATEST_JOB_ID"
  LOG_STREAM=$(aws batch describe-jobs --region "$REGION" --jobs "$LATEST_JOB_ID" \
    --query 'jobs[0].container.logStreamName' --output text 2>/dev/null || echo "")
  if [ -n "$LOG_STREAM" ] && [ "$LOG_STREAM" != "None" ]; then
    echo "  stream: /aws/batch/job:${LOG_STREAM}"
    aws logs tail "/aws/batch/job" --log-stream-names "$LOG_STREAM" --region "$REGION" --since 1h --format short 2>&1 \
      | tail -20 | sed 's/^/    /' || echo "    (no log output yet)"
  else
    echo "  (job hasn't started a container yet)"
  fi
else
  echo "  (no Batch job found for this dataset)"
fi

sep
echo "## 5. compression queue depth"
COMP_QUEUE_URL=$(aws sqs get-queue-url --region "$REGION" --queue-name "$COMPRESSION_QUEUE_NAME" --query QueueUrl --output text 2>/dev/null || echo "")
if [ -n "$COMP_QUEUE_URL" ]; then
  aws sqs get-queue-attributes --region "$REGION" --queue-url "$COMP_QUEUE_URL" \
    --attribute-names ApproximateNumberOfMessages ApproximateNumberOfMessagesNotVisible \
    --query 'Attributes' --output table 2>&1 | sed 's/^/  /'
else
  echo "  (compression queue not found)"
fi

sep
echo "## 6. S3 objects under dataset"
if ! aws s3 ls "s3://${BUCKET}" --region "$REGION" >/dev/null 2>&1; then
  echo "  (bucket ${BUCKET} does not exist yet — storage stack probably not deployed)"
else
  for sub in raw compressed models; do
    echo "  ${sub}/:"
    out=$(aws s3 ls "s3://${BUCKET}/${CID}/${DSID}/${sub}/" --region "$REGION" --recursive --summarize 2>/dev/null || true)
    if [ -z "$out" ]; then
      echo "    (none)"
    else
      echo "$out" | tail -5 | sed 's/^/    /'
    fi
  done
fi

sep
echo "done."

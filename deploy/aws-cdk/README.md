# Krunch AWS CDK deployer

Reference AWS Batch deployment for distributed krunch jobs. Works on a
fresh AWS account using the default VPC; no pre-existing infra needed.

What gets created:
- Two Batch compute environments — spot (cheap) + on-demand (reliable
  fallback). Both scale to zero when idle.
- A job queue routed primary→fallback per the `spot` prop.
- Three job definitions: `compress` (GPU array), `decompress` (GPU
  array), and `finalize` (CPU stitcher used by both modes).
- An S3 bucket for compressed output + temp parts, with a 3-day
  lifecycle on `*.parts/` to auto-clean orphans.
- A CloudWatch log-retention rule on `/aws/batch/job` (default 30
  days; override via `logRetention` prop).
- Stack-level `Project: krunch` / `ManagedBy: cdk` tags propagated to
  all taggable resources for cost allocation.

There is **no always-on EC2 instance** — Batch spins capacity up + down
per job.

## Prerequisites

- AWS CLI configured (`aws configure`)
- Node.js 18+
- `krunch` CLI installed (see top-level `install.sh`)

## Deploy

```bash
npm install
npx cdk bootstrap          # one-time per account/region
npx cdk deploy
```

Stack outputs (consumed by your `aws batch submit-job` invocation):

| Output | Purpose |
|---|---|
| `JobQueueArn` | Batch job queue |
| `CompressJobDefOutput` | GPU job definition for compress array |
| `DecompressJobDefOutput` | GPU job definition for decompress array |
| `FinalizeJobDefOutput` | CPU job definition for the finalize stitcher |
| `BucketName` | S3 bucket for output + temp parts |

## Submit a job

`krunch plan` emits an orchestrator-agnostic spec — env vars, command,
container overrides, array size, timeout. The orchestrator-specific
fields (`job-queue`, `job-definition`, `dependsOn`) come from the CDK
stack outputs and are passed via the AWS CLI's own flags at submit
time. `aws batch submit-job` merges the flags into the spec. (`--input-
len` auto-resolves from `--source` when it's an S3 URL, so you don't
need to pre-compute it.)

```bash
# 1. Render the spec.
krunch plan --target aws-batch --mode compress \
  --source s3://<your-bucket>/logs/data.jsonl \
  --dest   s3://<your-bucket>/logs/data.krunch \
  --workers 4 > job.json

# 2. Look up the orchestrator-specific ARNs.
QUEUE=$(aws cloudformation describe-stacks --stack-name KrunchStack \
  --query 'Stacks[0].Outputs[?OutputKey==`JobQueueArn`].OutputValue' --output text)
COMPRESS_JD=$(aws cloudformation describe-stacks --stack-name KrunchStack \
  --query 'Stacks[0].Outputs[?OutputKey==`CompressJobDefOutput`].OutputValue' --output text)
FINALIZE_JD=$(aws cloudformation describe-stacks --stack-name KrunchStack \
  --query 'Stacks[0].Outputs[?OutputKey==`FinalizeJobDefOutput`].OutputValue' --output text)

# 3. Submit the array job (compress workers).
ARRAY_ID=$(jq .main job.json | aws batch submit-job \
    --cli-input-json file:///dev/stdin \
    --job-queue "$QUEUE" --job-definition "$COMPRESS_JD" \
    --query jobId --output text)

# 4. Submit the finalize job (waits on the array via dependsOn).
jq .finalize job.json | aws batch submit-job \
    --cli-input-json file:///dev/stdin \
    --job-queue "$QUEUE" --job-definition "$FINALIZE_JD" \
    --depends-on jobId="$ARRAY_ID",type=SEQUENTIAL
```

For decompress, swap `--mode decompress` and use `DecompressJobDefOutput`.

For a working end-to-end example (compress + decompress + byte-exact
roundtrip on 100 MB WildChat), see `tests/integration/batch.sh`.

## Customize

Most defaults work for a fresh account. Edit `bin/app.ts` to override:

```typescript
new KrunchStack(app, "KrunchStack", {
  // Higher worker cap. Default 4 = 16 vCPU = the fresh-account
  // On-Demand G+VT quota in us-east-1. Anything higher needs a
  // service-quota increase first or RunInstances will fail with
  // VcpuLimitExceeded.
  maxWorkers: 16,

  // Custom-AMI optimization to skip the 3.5 GB cold-pull every job.
  // imageId: "ami-0xxxxxxxxxxxxxxxx",

  // On-demand only when spot is reliably unavailable in your region.
  spot: false,

  // Reuse an existing bucket instead of letting the stack create one.
  s3BucketName: "my-existing-bucket",

  // CloudWatch retention for /aws/batch/job. Default 30 days.
  // logRetention: logs.RetentionDays.ONE_WEEK,
});
```

See `lib/krunch-stack.ts` (`KrunchStackProps`) for the full prop list.

## Cold-start

First job on a fresh CE: ~13 min (T4 measurement) — EC2 launch + 3.5 GB
image pull + model load + WKV-kernel JIT. Subsequent jobs on warm
instances: ~30 s. Set `imageId` to a pre-baked AMI (image already in the
docker cache) to drop the 3.5 GB pull.

## Spot capacity caveat

g5.xlarge spot is intermittent in some regions (us-east-1 has been dry
during recent windows). If your jobs sit RUNNABLE forever with
`instance-terminated-no-capacity` in the EC2 spot history, either set
`spot: false` in `bin/app.ts` and `cdk deploy`, OR temporarily disable
the spot CE: `aws batch update-compute-environment --compute-environment
SpotEnv-... --state DISABLED`. The on-demand CE at priority 2 picks
up automatically.

## Tear down

```bash
npx cdk destroy
```

Compute environments scale to zero when idle, so leaving the stack up
costs effectively nothing (CloudWatch + the empty S3 bucket). The
bucket has `RemovalPolicy: RETAIN` — `cdk destroy` leaves it behind;
delete manually if you want it gone.

## Logs

Per-task logs land in CloudWatch under `/aws/batch/job`:

```bash
# List recent log streams across all tasks
aws logs describe-log-streams --log-group-name /aws/batch/job \
  --order-by LastEventTime --descending --max-items 5

# Get a specific job's log stream + status
aws batch describe-jobs --jobs <job-id> \
  --query 'jobs[0].[status,statusReason,attempts[0].container.logStreamName]' \
  --output text
```

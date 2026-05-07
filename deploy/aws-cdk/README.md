# Krunch AWS CDK deployer

Deploys an AWS Batch environment for distributed krunch compression jobs.
Works on a fresh AWS account using the default VPC; no pre-existing
infra needed.

What gets created:
- Two Batch compute environments (spot + on-demand fallback) using
  g5.xlarge instances, scale-to-zero when idle
- A job queue routed to whichever environment the `--spot` prop selects
- Three job definitions: `compress` (GPU array task), `decompress`
  (GPU array task), and `finalize` (CPU stitcher used by both modes)
- An S3 bucket for compressed output and temporary parts (3-day
  lifecycle on `*.parts/` to clean orphans)

There is **no always-on EC2 instance** — Batch spins up and tears down
spot capacity per job.

## Prerequisites

- AWS CLI configured (`aws configure`)
- Node.js 18+
- `krunch` CLI installed on your machine (see top-level `install.sh`)

## Deploy

```bash
npm install
npx cdk bootstrap          # one-time per account/region
npx cdk deploy
```

Stack outputs (read by `krunch plan --target aws-batch`):

| Output | Purpose |
|---|---|
| `JobQueueArn` | Batch job queue to submit to |
| `CompressJobDefOutput` | GPU job definition for compress array tasks |
| `DecompressJobDefOutput` | GPU job definition for decompress array tasks |
| `FinalizeJobDefOutput` | CPU job definition for the finalize stitcher |
| `BucketName` | S3 bucket for output + temp parts |

## Submit a compression job

`krunch plan` emits an orchestrator-agnostic spec — env vars, command,
container overrides, array size, timeout. The orchestrator-specific
fields (job-queue, job-definition, dependsOn) you supply via the AWS
CLI's own flags at submit time. `aws batch submit-job` merges the
flags into the spec, so the rendered JSON stays portable.

`--input-len` auto-resolves from `--source` when it's an S3 URL.

```bash
# Render. krunch plan knows nothing about your queue or job defs.
krunch plan --target aws-batch --mode compress \
  --source s3://<your-bucket>/logs/data.jsonl \
  --dest   s3://<your-bucket>/logs/data.krunch \
  --workers 4 > job.json

# Look up the orchestrator-specific ARNs from the CDK stack outputs.
QUEUE=$(aws cloudformation describe-stacks --stack-name KrunchStack \
  --query 'Stacks[0].Outputs[?OutputKey==`JobQueueArn`].OutputValue' --output text)
COMPRESS_JD=$(aws cloudformation describe-stacks --stack-name KrunchStack \
  --query 'Stacks[0].Outputs[?OutputKey==`CompressJobDefOutput`].OutputValue' --output text)
FINALIZE_JD=$(aws cloudformation describe-stacks --stack-name KrunchStack \
  --query 'Stacks[0].Outputs[?OutputKey==`FinalizeJobDefOutput`].OutputValue' --output text)

# Submit the array job (compress workers).
ARRAY_ID=$(jq .main job.json | aws batch submit-job \
    --cli-input-json file:///dev/stdin \
    --job-queue "$QUEUE" --job-definition "$COMPRESS_JD" \
    --query jobId --output text)

# Submit the finalize job (CPU stitcher), waiting on the array.
jq .finalize job.json | aws batch submit-job \
    --cli-input-json file:///dev/stdin \
    --job-queue "$QUEUE" --job-definition "$FINALIZE_JD" \
    --depends-on jobId="$ARRAY_ID",type=SEQUENTIAL
```

For decompress, swap `--mode decompress` and use `DecompressJobDefOutput`.
`--workers` controls the array size (parallel GPU instances). The
compute environment caps total parallelism via `maxWorkers` (default 4
— matches the fresh-account 16 vCPU On-Demand G+VT quota in us-east-1;
override higher only if your AWS quota allows).

For a working end-to-end example, see `tests/integration/batch.sh`.

See `tests/integration/batch.sh` at the repo root for a full working end-to-end
example that compresses + decompresses + verifies byte-exact roundtrip
on a 100 MB WildChat sample.

## Customize

Edit `bin/app.ts`:

```typescript
new KrunchStack(app, "KrunchStack", {
  // Larger GPU per worker (more VRAM headroom for >1 MB chunks)
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.G5, ec2.InstanceSize.X2LARGE),

  // On-demand if spot availability is unreliable in your region
  spot: false,

  // Higher cap on concurrent GPU instances
  // Higher cap on concurrent GPU instances — first request a vCPU
  // service-quota increase for "Running On-Demand G and VT instances"
  // in your region (default 16 vCPU = 4× g5.xlarge). Without that, AWS
  // will reject RunInstances with VcpuLimitExceeded regardless of this
  // setting.
  maxWorkers: 16,

  // Reuse an existing bucket instead of creating a new one
  s3BucketName: "my-existing-bucket",
});
```

## Cold-start behavior

First job on a fresh compute environment: ~3-5 minutes overhead before
the first task runs (EC2 spot launch + image pull + container start).
Subsequent jobs on warm instances: ~30 seconds overhead.

To eliminate cold-pull time entirely, bake the image into a custom AMI
and set `imageId` on the compute resources. Worth doing only if you
run many small jobs.

## Tear down

```bash
npx cdk destroy
```

Compute environments scale to zero when idle, so leaving the stack up
costs essentially nothing (just CloudWatch + the empty S3 bucket).
The bucket has a `RemovalPolicy: RETAIN`, so `cdk destroy` leaves it
behind — delete manually if you want it gone.

## Logs

Per-task logs go to CloudWatch under `/aws/batch/job` by default.
Find them via:

```bash
aws logs describe-log-streams --log-group-name /aws/batch/job \
  --order-by LastEventTime --descending --max-items 5
```

Or check job status directly:

```bash
krunch status --job-id <id-from-submit>
```

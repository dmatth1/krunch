# Krunch AWS CDK deployer

Deploys an AWS Batch environment for distributed krunch compression jobs.
Works on a fresh AWS account using the default VPC; no pre-existing
infra needed.

What gets created:
- Two Batch compute environments (spot + on-demand fallback) using
  g5.xlarge instances, scale-to-zero when idle
- A job queue routed to whichever environment the `--spot` prop selects
- Two job definitions: `compress` (GPU array task) and `assemble`
  (CPU stitcher)
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

Stack outputs (used by `krunch submit`):

| Output | Purpose |
|---|---|
| `JobQueueArn` | Batch job queue to submit to |
| `CompressJobDefOutput` | Job definition for compress array tasks |
| `AssembleJobDefOutput` | Job definition for the assemble task |
| `BucketName` | S3 bucket for output + temp parts |

## Submit a compression job

After deploy, `krunch submit` reads the outputs from CloudFormation and
fans out an array job + assemble:

```bash
krunch submit \
  --source s3://<your-bucket>/logs/data.jsonl \
  --dest   s3://<your-bucket>/logs/data.krunch \
  --workers 8
```

`--workers` controls the array size (parallel GPU instances). The
compute environment caps total parallelism via `maxWorkers` (default 10).

## Customize

Edit `bin/app.ts`:

```typescript
new KrunchStack(app, "KrunchStack", {
  // Larger GPU per worker (more VRAM headroom for >1 MB chunks)
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.G5, ec2.InstanceSize.X2LARGE),

  // On-demand if spot availability is unreliable in your region
  spot: false,

  // Higher cap on concurrent GPU instances
  maxWorkers: 50,

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

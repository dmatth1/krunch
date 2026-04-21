# learned-archive — CDK infrastructure

AWS CDK app that provisions the service architecture described in
[`../docs/SERVICE_ARCHITECTURE.md`](../docs/SERVICE_ARCHITECTURE.md).

## Docker images are built by CodeBuild, not by `cdk deploy`

Training + compression images are built by **AWS CodeBuild** (project
`krunch-image-build`, `buildspec.yml` at the repo root), not by
the laptop during `cdk deploy`. A cold push of the 5 GB PyTorch CUDA
devel image from home takes 30-90 min; CodeBuild runs in AWS and
pushes over the AWS backbone in ~3 min. Cost is ~$0.01 per build.

Iteration loop:

```bash
# 1. Edit Dockerfile / entrypoint / requirements / scripts / vendor patch
# 2. (Optional but recommended) shake down locally — catches import bugs
#    in 20 s instead of 10 min of Batch-submit roundtrip
docker buildx build --platform linux/amd64 -f cdk/docker/training/Dockerfile \
    --load -t krunch-train-check .
docker run --rm --entrypoint python krunch-train-check -c \
    "import sys; sys.path[:0]=['/app','/app/vendor/L3TC']; \
     from scripts.train_l3tc_phase11 import build_model; print('ok')"
# 3. Commit + push — CodeBuild clones from GitHub
git push origin main
# 4. Trigger CodeBuild; BUILD ~140 s cold, ~30 s warm cache
aws codebuild start-build --project-name krunch-image-build --region us-east-1
# 5. Watch phases
aws codebuild batch-get-builds --ids <build-id> --region us-east-1 \
    --query 'builds[0].{S:buildStatus,P:currentPhase}' --output table
# 6. Grab the image URI from the artifact
aws s3 cp s3://krunch-codebuild-artifacts-584956668248/builds/image-info.zip /tmp/ \
    && unzip -p /tmp/image-info.zip image-info.json
# 7a. Fast path: register a new Batch JobDefinition revision pointing
#     at the new tag + submit a one-off job (bypasses CDK entirely):
aws batch register-job-definition --cli-input-json file://jobdef-new.json \
    --region us-east-1
aws batch submit-job --job-queue archive-dev-training-queue \
    --job-definition archive-dev-training-jobdef:<NEW_REV> \
    --job-name manual-run-$(date +%s) --region us-east-1
# 7b. Proper path: `cdk deploy` (still does a local build today —
#     PRODUCTION_TODO item 10 tracks swapping the DockerImageAsset
#     for ContainerImage.fromEcrRepository so cdk only touches CFN).
```

## Stack inventory

| stack | contains |
|---|---|
| `archive-dev-storage` | S3 bucket + `datasets` + `model_versions` DynamoDB tables |
| `archive-dev-queues` | SQS queues (training-submit, compression) + DLQs |
| `archive-dev-api` | API Gateway REST API + PUT/GET Lambdas |
| `archive-dev-ingest` | S3 ObjectCreated → Ingest Lambda (with DDB concurrency guard) |
| `archive-dev-training` | AWS Batch (EC2 spot) + launcher + completion Lambdas + EventBridge rule |
| `archive-dev-compression` | ECS Fargate worker (auto-scaled 0 → N by queue depth) |

## Prerequisites

- AWS CLI configured (`aws sts get-caller-identity` returns your account)
- Node 20+ and npm
- Docker Desktop running locally — only needed for the shake-down
  import check (see above). The real image build runs in CodeBuild.
- Your AWS account bootstrapped for CDK in the target region (one-time):
  ```bash
  npx cdk bootstrap aws://<ACCOUNT_ID>/us-east-1
  ```

## Install & deploy

```bash
cd cdk
npm install
npx cdk synth --context env=dev       # verify synthesis
npx cdk deploy --all --context env=dev
```

Deploy time: ~12-15 min on first run (Docker image builds + push to
ECR dominate). Subsequent deploys are faster (~2-5 min).

## Destroy

```bash
npx cdk destroy --all --context env=dev
```

In dev, the S3 bucket is `autoDeleteObjects=true` so `cdk destroy`
cleans up customer data. In prod, the bucket has `RETAIN` so the
cleanup is manual — by design.

## Testing end-to-end (post-deploy)

```bash
# Grab the API endpoint + key from stack outputs
API_URL=$(aws cloudformation describe-stacks --stack-name archive-dev-api \
  --query 'Stacks[0].Outputs[?OutputKey==`ApiEndpoint`].OutputValue' --output text)
API_KEY_ID=$(aws cloudformation describe-stacks --stack-name archive-dev-api \
  --query 'Stacks[0].Outputs[?OutputKey==`ApiKeyId`].OutputValue' --output text)
API_KEY=$(aws apigateway get-api-key --api-key "$API_KEY_ID" --include-value \
  --query 'value' --output text)

# Upload some logs
curl -X PUT "${API_URL}v1/customers/acme/datasets/nginx-logs/events" \
  -H "Content-Type: application/x-ndjson" \
  -H "X-Api-Key: ${API_KEY}" \
  --data-binary @some_logs.ndjson

# Poll dataset metadata
curl "${API_URL}v1/customers/acme/datasets/nginx-logs" \
  -H "X-Api-Key: ${API_KEY}"
```

## File layout

```
cdk/
├── bin/app.ts                       # CDK entry — stack instantiation
├── lib/
│   ├── storage-stack.ts             # S3 + DynamoDB
│   ├── queue-stack.ts               # SQS
│   ├── api-stack.ts                 # API Gateway + PUT/GET lambdas
│   ├── ingest-stack.ts              # S3 event → Ingest Lambda
│   ├── training-stack.ts            # Batch compute + launcher + completion
│   ├── compression-stack.ts         # Fargate compression worker
│   └── shared/constants.ts          # env names, resource naming
├── src/
│   ├── handlers/
│   │   ├── put.ts                   # PUT Lambda handler
│   │   ├── get.ts                   # GET Lambda handler (metadata stub)
│   │   ├── ingest.ts                # S3 ObjectCreated handler (concurrency guard)
│   │   ├── training-launcher.ts     # SQS → Batch.submitJob
│   │   └── training-complete.ts     # EventBridge → DDB + compression sweep
│   └── shared/
│       ├── ddb.ts                   # DynamoDB helpers
│       └── types.ts                 # shared TS types
├── docker/
│   ├── training/
│   │   ├── Dockerfile               # Python training image
│   │   ├── requirements.txt         # Python deps (CPU PyTorch)
│   │   └── train_entrypoint.sh      # sync → train → eval → upload
│   └── compression/
│       ├── Dockerfile               # Rust compression image
│       └── compression_worker.py    # SQS poll + l3tc compress + DDB update
├── cdk.json
├── package.json
└── tsconfig.json
```

## Key guarantees

**No duplicate training jobs.** The ingest Lambda uses a DynamoDB
conditional update to atomically flip `status=awaiting_corpus →
training`. If two concurrent S3 events race, only one wins the
conditional check; the loser silently skips job submission. See
`src/shared/ddb.ts::claimTrainingSlot` and `docs/SERVICE_ARCHITECTURE.md`
for the full state machine.

**Tagging.** Every CDK-managed resource is tagged with `Project=learned-archive`
+ `Environment=<env>` + `ManagedBy=cdk` for cost attribution.

## Current status

Spike 1 in flight. All 6 stacks deployed. `measure_held_out_ratio.py`
implemented. CodeBuild project (`krunch-image-build`) operational.

See [`../PRODUCTION_TODO.md`](../PRODUCTION_TODO.md) for the gap list
between "spike works" and "ready for customers".

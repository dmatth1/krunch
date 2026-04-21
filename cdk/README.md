# learned-archive — CDK infrastructure

AWS CDK app that provisions the service architecture described in
[`../docs/SERVICE_ARCHITECTURE.md`](../docs/SERVICE_ARCHITECTURE.md).

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
- Docker running locally (CDK builds training + compression images)
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

Pre-spike scaffold. The stacks synthesize clean (`cdk synth` passes)
and are deployable. No production hardening yet. No tests yet.

## TODO (post-Spike 1)

- Implement `scripts/measure_held_out_ratio.py` (used by training
  entrypoint to decide `codec=l3tc` vs `codec=zstd_fallback`).
- Implement GET range retrieval (currently stub that returns metadata).
- Wire DELETE endpoints + lifecycle policies.
- Add CloudWatch dashboards (queue depth, training duration, ratios).
- Add integration tests using LocalStack + jest.
- Switch S3 bucket lifecycle rule from tag-based to prefix-based (the
  current `tagFilters: {lifecycle: raw}` is correct but requires the
  PUT Lambda to tag on upload; verify that's happening).

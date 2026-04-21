# Service architecture

Concrete AWS architecture for the learned-archive service MVP.
This doc is a working reference — if the stacks under `cdk/` drift
from this, the stacks are ground truth; update this doc.

## Non-goals

- Auth beyond API keys (SAML/OAuth later)
- Content search / query language (we're storage, not analytics)
- Multi-region (single-region MVP)
- Cost optimization (AWS free tier covers everything at spike scale)

## Tech stack

- **IaC:** AWS CDK v2, TypeScript
- **Compute:** Lambda (API + ingest + orchestration), AWS Batch on EC2 spot (training), ECS Fargate (compression worker)
- **Storage:** S3 (objects), DynamoDB on-demand (metadata)
- **Messaging:** SQS (pipeline decoupling), EventBridge (Batch completion)
- **Gateway:** API Gateway REST, private (VPC endpoint)
- **Training image:** the existing `scripts/train_l3tc_phase11.py` wrapped in a Docker image
- **Compression image:** the Rust `l3tc-rust` binary in a Docker image

## Data flow

```
 customer
    │
    │ PUT /v1/customers/{cid}/datasets/{dsid}/events  (NDJSON body)
    ▼
┌─────────────────┐
│ API Gateway     │  private REST, API key auth
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│ PUT Lambda      │  · small payload (<10MB): write NDJSON to S3 raw/
│                 │  · large payload: return pre-signed S3 URL
│                 │  · upsert DynamoDB datasets row (status=awaiting_corpus
│                 │    if new, else unchanged)
└───────┬─────────┘
        │
        ▼ (small-upload path) OR (S3 ObjectCreated event for pre-signed path)
┌─────────────────┐
│ S3 bucket:      │  s3://archive-{env}/{cid}/{dsid}/raw/{uuid}.ndjson
│ raw holding     │
└───────┬─────────┘
        │ ObjectCreated event
        ▼
┌─────────────────┐
│ Ingest Lambda   │  Reads dataset DDB row:
│                 │   · status=awaiting_corpus → try to claim for training
│                 │     (atomic DDB update with ConditionExpression)
│                 │   · status=training → skip (compression queued later)
│                 │   · status=ready → send msg to compression SQS
└───────┬─────────┘
        │
        ├───(if claimed training slot)────▶ Training SQS
        │                                      │
        │                                      ▼
        │                             ┌─────────────────┐
        │                             │ Launcher Lambda │ submits AWS Batch job
        │                             └─────────┬───────┘
        │                                       │
        │                                       ▼
        │                             ┌───────────────────┐
        │                             │ AWS Batch job     │ EC2 spot
        │                             │  1. pulls raw     │
        │                             │     from S3       │
        │                             │  2. trains model  │
        │                             │     (Python)      │
        │                             │  3. evals vs zstd │
        │                             │  4. writes model  │
        │                             │     + metadata    │
        │                             │     to S3         │
        │                             │  5. completion    │
        │                             │     event via     │
        │                             │     EventBridge   │
        │                             └──────────┬────────┘
        │                                        │
        │                                        ▼
        │                             ┌───────────────────┐
        │                             │ Completion Lambda │ updates DDB
        │                             │                   │ status=ready,
        │                             │                   │ current_model_ver,
        │                             │                   │ sends msgs to
        │                             │                   │ compression SQS
        │                             │                   │ for every pending
        │                             │                   │ raw/ object
        │                             └──────────┬────────┘
        │                                        │
        └───(status=ready path)────────────────▶ │
                                                 ▼
                                        ┌────────────────────┐
                                        │ Compression SQS    │
                                        └──────────┬─────────┘
                                                   │
                                                   ▼
                                        ┌─────────────────────┐
                                        │ ECS Fargate task    │
                                        │  (Rust l3tc binary) │
                                        │  · reads raw        │
                                        │  · reads model      │
                                        │  · writes compressed│
                                        │    blob to S3       │
                                        │  · deletes raw      │
                                        │  · updates DDB      │
                                        │    byte counters    │
                                        └─────────────────────┘
```

## The "no duplicate training jobs" guard

This is the key correctness property: **for a given `(cid, dsid)`, at
most one training job is in flight at any time.**

Implementation via DynamoDB conditional update in the ingest Lambda:

```typescript
// In ingest Lambda, after S3 ObjectCreated for a new raw file:
const res = await ddb.update({
  TableName: DATASETS_TABLE,
  Key: { pk: `CUST#${cid}`, sk: `DS#${dsid}` },
  UpdateExpression:
    "SET #s = :training, last_training_started_at = :now, raw_bytes_held = raw_bytes_held + :delta",
  ConditionExpression: "#s = :awaiting OR attribute_not_exists(#s)",
  ExpressionAttributeNames: { "#s": "status" },
  ExpressionAttributeValues: {
    ":training": "training",
    ":awaiting": "awaiting_corpus",
    ":now": new Date().toISOString(),
    ":delta": objectSize,
  },
});
```

- If the condition passes (status was `awaiting_corpus` or row didn't exist),
  we atomically flip to `training` and get exclusive ownership. Then we
  submit the Batch job.
- If the condition fails (status was already `training` or `ready`),
  we catch `ConditionalCheckFailedException` and either:
  - skip (if `training`) — the in-flight job will process this new
    raw object via the "completion sweep" step
  - send to compression SQS (if `ready`)

No distributed lock needed. DynamoDB conditional updates are atomic.

Additional safety nets:
- **AWS Batch job name is deterministic**: `{cid}-{dsid}-train-{timestamp}`.
  Batch deduplicates concurrent jobs with identical `jobName` within the
  same queue at submit time, giving us a second line of defense.
- **SQS FIFO with `MessageGroupId = {cid}-{dsid}`** for the training
  queue — guarantees serialized processing per dataset if we ever need
  retries to go through the queue. MVP uses standard SQS; upgrade to FIFO
  if we see issues.
- **Batch job env var `EXPECTED_DDB_CURRENT_MODEL_VERSION`** — the job
  reads DDB on start, aborts if status is no longer `training` (e.g.
  someone manually rolled back state). Prevents stale jobs from writing
  new model data over a reset state.

## DynamoDB schema

### Table: `datasets` (on-demand billing)

Primary key:
- `pk` (string): `CUST#{cid}`
- `sk` (string): `DS#{dsid}`

Attributes:
- `created_at` (ISO 8601 string)
- `status` (string): `awaiting_corpus` | `training` | `ready` | `retraining` | `fallback_zstd`
- `current_model_version` (number, null until first model ready)
- `raw_bytes_held` (number): bytes in S3 `raw/` awaiting compression
- `compressed_bytes` (number): bytes in compressed blobs
- `total_events_ingested` (number): count of NDJSON lines ever PUT
- `last_training_started_at` (string, ISO)
- `last_training_completed_at` (string, ISO)
- `training_job_id` (string, Batch job id if running)
- `training_corpus_bytes` (number): bytes used to train current model

### Table: `model_versions` (on-demand billing)

Primary key:
- `pk` (string): `CUST#{cid}#DS#{dsid}`
- `sk` (number): version number (1, 2, 3, …)

Attributes:
- `created_at` (string)
- `s3_model_path` (string)
- `s3_tokenizer_path` (string)
- `training_seconds` (number)
- `training_corpus_bytes` (number)
- `held_out_ratio` (number): our model on held-out 20%
- `zstd_baseline_ratio` (number): zstd-22 on same 20%
- `codec` (string): `l3tc` | `zstd_fallback`

## S3 layout

Single bucket, environment-suffixed: `archive-{env}` (e.g. `archive-dev`).

```
s3://archive-{env}/
  {cid}/
    {dsid}/
      raw/
        {uuid}.ndjson         # pre-compression holding
      compressed/
        {YYYY}/{MM}/{DD}/{HH}/
          {uuid}.bin          # compressed blobs, one per PUT batch
      models/
        v1.bin
        v1.tokenizer.model
        v1.metadata.json      # training run metadata
        v2.bin
        ...
```

Lifecycle policies:
- `raw/` → lifecycle rule: delete objects after 7 days (fallback cleanup
  if compression stage fails; normally deleted by compression worker
  immediately)
- `compressed/` → no auto-deletion
- `models/` → no auto-deletion (needed to decompress old data forever)

## SQS queues

All standard SQS (not FIFO) for MVP. All have DLQ with `maxReceiveCount = 5`.

| queue | purpose | message payload |
|---|---|---|
| `training-submit` | ingest Lambda → launcher Lambda | `{cid, dsid, trigger: "initial" \| "retrain"}` |
| `training-complete` | EventBridge → completion Lambda | Batch job completion event |
| `compression` | completion Lambda → Fargate | `{cid, dsid, s3_raw_key, model_version}` |
| `*-dlq` | dead-letter queues | same payload, 5× failed deliveries |

Visibility timeout per queue:
- `training-submit`: 1 minute (launcher Lambda is quick)
- `training-complete`: 1 minute (completion Lambda is quick)
- `compression`: 30 minutes (Fargate compress task can be long)

## API specification (MVP)

Base path: `/v1/customers/{cid}/datasets`

### PUT `/{dsid}/events`

Request:
- Content-Type: `application/x-ndjson` (or `text/plain` for lines)
- Body: newline-delimited events
- Query params: `timestamp_field` (optional, JSONPath)
- Headers: `X-Api-Key` (API key auth)

Responses:
- 202 Accepted (small body): `{batch_id, bytes_accepted}`
- 307 Temporary Redirect (large body): Location header = pre-signed S3 URL
- 400 (malformed body)
- 401 (bad API key)
- 429 (rate limited)

### GET `/{dsid}/events`

Request:
- Query: `start_ts` (required), `end_ts` (required), `limit` (default 10000),
  `cursor`, `format` (default `ndjson`)

Responses:
- 200: NDJSON body of events in range, sorted by timestamp
- 202 (dataset exists but model still training): `{status, estimated_ready_at}`
- 404 (unknown dataset)

### DELETE `/{dsid}/events`

Query: `before_ts` required. Deletes all compressed blobs with
`end_ts_unix_ms < before_ts`.

### Dataset lifecycle

- `POST /{dsid}` — explicit create (optional, auto-created on first PUT)
- `DELETE /{dsid}` — delete dataset, all data, all models
- `GET /{dsid}` — metadata: size, status, model version, ratio
- `GET /` — list datasets for customer

## CDK stack layout

```
cdk/
├── bin/
│   └── app.ts                    # entry: instantiates all stacks
├── lib/
│   ├── storage-stack.ts          # S3 bucket, DynamoDB tables
│   ├── queue-stack.ts            # SQS queues + DLQs
│   ├── api-stack.ts              # API Gateway, PUT/GET Lambdas
│   ├── ingest-stack.ts           # S3 event → ingest Lambda
│   ├── training-stack.ts         # Batch compute env, job def, launcher Lambda, completion Lambda
│   ├── compression-stack.ts      # ECS Fargate task def, worker Lambda dispatcher
│   └── shared/
│       └── constants.ts
├── src/
│   ├── handlers/
│   │   ├── put.ts                # PUT Lambda handler
│   │   ├── get.ts                # GET Lambda handler
│   │   ├── ingest.ts             # S3 event handler
│   │   ├── training-launcher.ts  # SQS → Batch.submitJob
│   │   └── training-complete.ts  # EventBridge → update DDB, enqueue compression
│   └── shared/
│       ├── ddb.ts                # DynamoDB client + helpers
│       └── types.ts              # shared TS types
├── docker/
│   ├── training/
│   │   └── Dockerfile            # Python training image (wraps train_l3tc_phase11.py)
│   └── compression/
│       └── Dockerfile            # Rust l3tc-rust binary image for Fargate
├── package.json
├── tsconfig.json
├── cdk.json
└── README.md                     # how to deploy
```

## Environments

For MVP: `dev` only. Once ratios validate, add `prod`.

Stacks suffix all named resources with `{env}`, e.g. `archive-dev-put-fn`.

## Deployment

```
cd cdk/
npm install
npx cdk bootstrap      # once per account+region
npx cdk deploy --all --context env=dev
```

Teardown: `npx cdk destroy --all --context env=dev`.

All resources created by CDK are tagged with `Project=learned-archive`
for cost explorer grouping.

## Cost expectations (MVP, zero-to-low traffic)

| service | monthly cost |
|---|---:|
| API Gateway | <$1 |
| Lambda | <$1 (free tier covers) |
| DynamoDB on-demand | <$1 |
| SQS | <$1 (free tier) |
| S3 storage (~100 GB during spike) | ~$2 |
| ECS Fargate (compression worker) | <$5 (runs briefly) |
| AWS Batch (EC2 spot training) | $5-20 (few training runs) |
| EventBridge | free |
| **Total** | **~$15-30/month** during spike phase |

Scales with traffic. Nothing surprising.

## Open follow-ups (post-MVP)

- Metrics / CloudWatch dashboard for ratio, training time, queue depth
- Drift monitor: periodic ratio-on-sample check, scheduled retrain trigger
- Client SDK (Python, Go) that handles pre-signed URL upload flow
- OTel ingest endpoint (target market heavily uses OTel already)
- S3-compatible gateway (huge adoption lever; harder to build right)
- Cross-tenant shared models for similar data shapes (v1.1+)
- Multi-region replication for compliance

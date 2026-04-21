# Krunch

> **Repo home:** [github.com/dmatth1/Krunch](https://github.com/dmatth1/Krunch) (private, pre-MVP)

_Internal codename: learned-archive. AWS resources are named `archive-{env}-*`._

Managed storage service for text-heavy data. Customer PUTs events via
an S3-ish API; we train a small RWKV-v4 model on their specific data
distribution (one model per dataset), compress with that model,
store on S3. Ratio target: 2-4× better than `zstd -22` on homogeneous
log-like data, retrieval latency comparable to S3 Standard (not
Glacier's 12-48 h).

**Positioning vs. the category:** Datadog Flex Logs, Elastic Frozen
Tier, Loki, Axiom all ship cheap log archive. All use generic
compression. The wedge is per-customer learned compression, which is
3-5× tighter on homogeneous data than anything off-the-shelf. Detailed
pitch + landscape + API surface + economics in
[`STORAGE_SERVICE.md`](STORAGE_SERVICE.md).

## Status (2026-04-21)

Spike 1 in progress. AWS service is deployed end-to-end (6 CDK
stacks: storage, queues, api, ingest, training, compression); first
training run on a real customer-shaped dataset (HDFS logs from Loghub,
1.39 GB NDJSON) is executing. The pass gate is
`held_out_ratio < 0.98 × zstd_baseline_ratio` — must beat zstd-22 by
≥2% on held-out data from the same distribution. Running log:
[`SPIKE_1_LOG.md`](SPIKE_1_LOG.md).

## Docs index

| doc | purpose |
|---|---|
| [`STORAGE_SERVICE.md`](STORAGE_SERVICE.md) | Product spec: pitch, API, economics, risks, spike plan |
| [`docs/SERVICE_ARCHITECTURE.md`](docs/SERVICE_ARCHITECTURE.md) | AWS architecture reference: stacks, data flow, DDB schemas |
| [`TRAINING_FLOW.md`](TRAINING_FLOW.md) | End-to-end walkthrough of one training job: tokenizer → RWKV → ratio → metadata |
| [`SPIKE_1_LOG.md`](SPIKE_1_LOG.md) | Running log of Spike 1 (HDFS on real service) |
| [`PRODUCTION_TODO.md`](PRODUCTION_TODO.md) | 10-item gap list between "spike works" and "ready for customers" |
| [`docs/ARCHIVE_l3tc.md`](docs/ARCHIVE_l3tc.md) | Historical context: what carries over from the archived l3tc-prod CLI project |
| [`cdk/README.md`](cdk/README.md) | CDK stack inventory + deploy instructions |
| [`l3tc-rust/README.md`](l3tc-rust/README.md) | Rust inference runtime (used at compression/decompression time) |
| [`bench/`](bench/) | Historical L3TC CLI benchmarks (pre-pivot; kept as ratio evidence for the pitch) |

## Tech stack

- **IaC:** AWS CDK v2, TypeScript.
- **Compute:** Lambda (API + ingest), AWS Batch on EC2 (training), ECS
  Fargate (compression worker).
- **Storage:** S3 + DynamoDB on-demand.
- **Messaging:** SQS (pipeline decoupling), EventBridge (Batch
  completion).
- **Training:** per-dataset 16 K-vocab SentencePiece unigram tokenizer
  + L3TC-200K RWKV-v4 architecture (~200 K params, 2 layers, d=96).
  bf16 mixed precision on g5.xlarge (A10G) via AWS Batch.
- **Inference (Spike 2+):** Rust runtime in `l3tc-rust/` reads a `.bin`
  checkpoint. Spike 1 defers this — storage uses `zstd --long=27 -22`
  while the model's entropy-bound ratio is measured and recorded in
  metadata.
- **Image builds:** AWS CodeBuild (triggered on git push) pushes to
  ECR over the AWS backbone; `buildspec.yml` at the repo root.

## Quick start

```bash
# Deploy everything to your own AWS account
cd cdk
npm install
npx cdk bootstrap
npx cdk deploy --all --context env=dev --require-approval never

# PUT a dataset via API (see STORAGE_SERVICE.md for endpoint shape)
curl -X PUT \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/x-ndjson" \
  --data-binary @events.ndjson \
  "$API_ENDPOINT/v1/customers/acme/datasets/my-logs/events"

# Poll pipeline state
./scripts/spike_status.sh acme my-logs
```

## License

Apache-2.0. See [LICENSE](LICENSE).

Compression technology derives from L3TC (AAAI 2025) and RWKV-LM
(Apache-2.0). See [NOTICE](NOTICE) for attribution.

## History

Forked from `l3tc-prod` on 2026-04-21. The CLI compressor direction
was archived when it couldn't beat `zstd` on the dimensions that
matter (speed, distribution, heterogeneous-text ratio). The
compression tech carries over; see
[`docs/ARCHIVE_l3tc.md`](docs/ARCHIVE_l3tc.md) for what's load-bearing.

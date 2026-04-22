# Krunch

> **Repo home:** [github.com/dmatth1/Krunch](https://github.com/dmatth1/Krunch) (private, pre-MVP)

_Internal codename: learned-archive. AWS resources are named `archive-{env}-*`._

## Thesis

**We store your data cheaper than raw on S3 — and cheaper than the
best classical compression (zstd) on S3. We do this by learning a
compression model on *your specific data* and applying it
automatically. You PUT data to our API; we handle model training,
per-chunk codec selection, and storage. You GET it back decompressed
on demand.**

The core bet: the same AI techniques that let large language models
predict the next word can be used, at small scale, to predict the
next byte of *your* data. A compression model trained on one
customer's logs / documents / records knows that customer's
vocabulary, structure, and style in ways no off-the-shelf codec
ever could. That knowledge shows up as bytes saved every month for
the life of the archive.

In practice it's a **hybrid dispatcher**: a small neural model for
text-like content (prose, code, JSON with free-text fields, chat,
medical / legal / audit records — where it beats zstd by 15–40%),
with classical codecs (zstd with a trained dictionary, bzip3, CLP
for templated logs, brotli for near-duplicate documents) dispatched
per-chunk for the content types where classical still wins.

See the full product positioning in
[`STORAGE_SERVICE.md`](STORAGE_SERVICE.md), the evidence behind the
ratio claims in [`CUSTOMER_PROFILE.md`](CUSTOMER_PROFILE.md), and the
hybrid dispatcher design in
[`HYBRID_CODEC_DESIGN.md`](HYBRID_CODEC_DESIGN.md).

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
| [`CUSTOMER_PROFILE.md`](CUSTOMER_PROFILE.md) | What realistic customer data looks like per vertical, why the codec design is shaped around it |
| [`HYBRID_CODEC_DESIGN.md`](HYBRID_CODEC_DESIGN.md) | Dispatcher architecture: neural + classical codec menu, detector, metrics |
| [`COMPETITIVE_LANDSCAPE.md`](COMPETITIVE_LANDSCAPE.md) | Who sells cheap compressed storage today, who claims AI-powered compression, where the defensible moat is |
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

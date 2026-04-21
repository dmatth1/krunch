# Krunch

> **Repo home:** [github.com/dmatth1/Krunch](https://github.com/dmatth1/Krunch) (private, pre-MVP)

_Internal codename: learned-archive. AWS resources are named `archive-{env}-*`._

> **Status: pre-spike.** No service code yet. The compression
> technology from the archived `l3tc-prod` project is reusable but the
> service itself hasn't been validated against real customer data. See
> `STORAGE_SERVICE.md` for the full product spec and spike plan.

A managed storage service for text-heavy data. Customer PUTs events
via an S3-compatible API; we train a custom compression model on their
data distribution, compress, store. GET decompresses and returns in
seconds. Targets 2-4× better compression than `zstd -22` on
homogeneous log-like data, at retrieval latency comparable to
S3 Standard — meaningfully better than Glacier Deep Archive's
12-48h SLA.

## The pitch

Your logs / audit records / transaction streams / document archives
are costing you too much on S3 Standard + zstd, or too much hassle on
Glacier with rehydrate + re-index tax, or too much per GB on
Axiom/Datadog/Elastic. We train a **small model on YOUR data** once,
then compress everything with that model from then on. Per-customer
specialization produces ratios that generic compression cannot match,
so our storage bill is dramatically smaller — and we pass most of the
savings on.

Retrieval is sub-10 seconds for typical ranges, not the 12+ hours that
Glacier Deep Archive imposes.

## Why this is the right product shape (vs. the archived CLI)

The previous direction (`l3tc-prod`) tried to be a CLI compressor
competing with `zstd`. Three structural problems killed it:
- zstd is 1000× faster and shipped in every OS
- Per-user model distribution is operationally ugly
- Ratio win on heterogeneous text is marginal

The service model fixes all three:
- Compression runs on our infrastructure; customer never waits
- Model lives with us forever, no distribution concern
- **Per-customer models on homogeneous data win by 3-5×, not 1.3×**

See `STORAGE_SERVICE.md` for the detailed argument.

## Competitive landscape

Category exists (cheap log / text archive). Differentiator is
compression tech, not product shape.

| product | compression approach |
|---|---|
| Datadog Flex Logs / Flex Frozen | Generic; rehydrate tax on search |
| Elastic Frozen Tier (searchable snapshots) | Generic; zstd-class ratios |
| Grafana Loki | Index-free chunked; generic LZ |
| Axiom | "95% compression" via hand-tuned data store; generic |
| Humio / Logscale | Streaming; generic |
| **this project** | **Per-customer learned models** |

Nobody in this category ships per-customer trained compression. That
is the wedge.

## API (planned — not built)

Time-sharded log ingest, following Grafana Loki / Elastic conventions:

```
POST   /v1/customers/{cid}/datasets             # create dataset
DELETE /v1/customers/{cid}/datasets/{dsid}      # delete dataset + data

PUT    /v1/customers/{cid}/datasets/{dsid}/events
  # body: application/x-ndjson
  # 202 Accepted + batch-id

GET    /v1/customers/{cid}/datasets/{dsid}/events
  # ?start_ts=<iso8601>&end_ts=<iso8601>&limit=N&cursor=...
  # returns decompressed NDJSON

DELETE /v1/customers/{cid}/datasets/{dsid}/events?before_ts=...
  # compliance / GDPR retention

GET    /v1/customers/{cid}/datasets             # list
GET    /v1/customers/{cid}/datasets/{dsid}      # metadata + model status
```

No query language, no content search, no tags, no dashboards.
Storage, not analytics.

## Ingest flow

```
PUT /events
  → stored raw in s3://bucket/{cid}/{dsid}/raw/
  → if dataset has trained model: compress, delete raw
  → else: queued; compress when model is ready

First N GB to a new dataset:
  → accumulate in raw/
  → kick off training job on GPU
  → model registered
  → raw data compressed, raw cleaned up
  → subsequent PUTs compress inline

Background:
  drift monitor samples incoming events
  if ratio degrades > threshold → schedule retrain
  new model_version_id for new data; old data stays on old model
```

## Current state

**All compression technology carries over from the archived
`l3tc-prod` project.** The Rust inference runtime, RWKV-v4 training
pipeline, `.pth → .bin` converter, Metal backend, Phase 11 spot-fleet
training infra — all usable as-is for the compression engine.

**None of the service is built.** Planned spike sequence:

1. **Spike 1 (5 days):** validate compression ratio on 3 real log
   corpora (nginx access, JSON events, syslog). Pass = ≥2× `zstd -22`
   on at least 2 of 3.
2. **Spike 2 (3 days):** corpus-size ablation (how much data does a
   customer need to upload before their model asymptotes).
3. **Spike 3 (1 week):** end-to-end retrieval latency. Single EC2 +
   S3 + pre-trained model. Pass = 10 MB decompressed in ≤5 seconds.
4. **Spike 4 (1 day):** unit economics with measured numbers.

Full plan: `STORAGE_SERVICE.md`.

## Target customer

- Multi-TB to multi-PB text-heavy data
- 5-30 year retention (compliance, audit, regulatory)
- Rare retrieval (<5% per year)
- OK with sub-10s retrieval latency (not ms-sensitive)
- Homogeneous data within a dataset (one format per bucket)

Verticals:
- Finance / fintech transaction archives
- Healthcare / pharma trial data
- Legal / e-discovery document archives
- SaaS platform log retention (7+ year compliance)

## Install / use

Not yet. Pre-spike. See `STORAGE_SERVICE.md` for the plan.

## License

Apache-2.0. See [LICENSE](LICENSE).

Compression technology derives from L3TC (AAAI 2025) and RWKV-LM
(Apache-2.0). See [NOTICE](NOTICE) for attribution.

## History

Forked from `l3tc-prod` on 2026-04-21. See
`docs/ARCHIVE_l3tc.md` (pending) for l3tc history.

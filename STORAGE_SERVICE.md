# Learned Archive — compressed cold-ish text storage as a service

**Status (2026-04-21):** pivoted here from the l3tc-prod CLI compressor
project. The compression technology, training pipelines, and runtime
from that project are all load-bearing for the service. The CLI/runtime
product is archived; what ships is a managed storage service.

## The one-paragraph pitch

A managed storage service for text-heavy data — logs, audit trails,
transaction records, documents. Customer PUTs data via an S3-ish API;
we train a small model on their specific data distribution (one
model per dataset), compress with that model, store on cheap backend
(S3 Standard). On GET, we retrieve and decompress. Ratio is 2-4×
better than `zstd -22` on homogeneous domain data. Retrieval takes
seconds, not hours (unlike Glacier Deep Archive). Price point:
meaningfully below what customers pay for either `zstd + S3 Standard`
(for fast retrieval) or `zstd + Glacier Deep Archive` + rehydrate
tax (for compliance archives).

## Why this works when the CLI didn't

The CLI pitch — "replace zstd with a neural compressor" — failed on
comparison to zstd: 1000× slower, incremental ratio win, no
distribution story. The service model changes the comparison:

- **The ratio lever is per-customer specialization.** Customer logs
  are homogeneous within a dataset. A 200K-param specialist trained
  on *their* nginx logs, *their* webhook stream, *their* event bus
  crushes generic compression in a way no off-the-shelf algorithm
  can match. This is a 3-5× ratio win on real log-like data, not
  the 1.3× win l3tc got on Wikipedia.
- **Compute cost is ops-side, not user-side.** Customer never waits
  for a model to compress their PUT. We compress async in background
  and replace raw-held storage once compressed. Customer's view is
  S3-speed PUT latency.
- **Training cost amortizes over retention.** One training run per
  dataset, reused forever on incremental PUTs. Retrain only on
  meaningful data drift.
- **We are never OOD.** The model was literally trained on this
  customer's data. Drift is the only risk, and we monitor and
  retrain when needed.
- **No distribution problem.** Model lives on our infrastructure.
  No shipping, no compatibility matrix, no binary-size limits.

## Competitive landscape (current, as of 2026-04)

What exists in text/log archival:

| product | compression | key differentiator | what we do differently |
|---|---|---|---|
| Datadog Flex Logs | Generic LZ | Hot-queryable cold tier; $0.05/M events min 30-day | Higher ratio, no rehydrate tax |
| Datadog Archive Search | Generic (customer S3) | Query without re-index | Backend compression is 3-5× tighter |
| Elastic Frozen Tier | Searchable snapshots | Decoupled compute/storage, ~50% storage reduction | Higher ratio, no search-cluster dependency |
| Grafana Loki | Index-free, chunked | Object-store-native from day 1 | Same storage pattern, dramatically better compression |
| Axiom | "95% compression" (hand-tuned data store) | Unlimited retention pitch | Learned per-customer model instead of one-size-fits-all |
| Humio/Logscale | Index-free | Streaming ingest | Same backend pattern |

**The category already buys "cheap compressed log archive."** We're
not inventing the product shape. We're plugging dramatically better
compression into an existing product shape.

**Nobody in this category is doing per-customer learned compression.**
Academic papers exist (LogShrink, NeurLZ, ELISE, DeepZip) but none
productized. Axiom's "95%" is generic compression on structured log
fields. Elastic's ratios are zstd-level. This is the unclaimed wedge.

---

## Product surface (MVP)

### REST API

Following Grafana Loki / Elasticsearch time-sharded log ingest
conventions. No query language, no content search — just
time-range log storage and retrieval.

```
POST   /v1/customers/{cid}/datasets              # create dataset (optional;
                                                 # auto-created on first PUT)
DELETE /v1/customers/{cid}/datasets/{dsid}       # delete dataset + all data
GET    /v1/customers/{cid}/datasets              # list datasets
GET    /v1/customers/{cid}/datasets/{dsid}       # metadata: size, ratio, model status

PUT    /v1/customers/{cid}/datasets/{dsid}/events
  # Content-Type: application/x-ndjson
  # Body: newline-delimited JSON events (or newline-delimited plain text)
  # Query: timestamp_field=<jsonpath>  (optional; defaults to ingest time)
  # Response: 202 Accepted + batch-id

GET    /v1/customers/{cid}/datasets/{dsid}/events
  # Query: start_ts=<iso8601> & end_ts=<iso8601>
  #        limit=<N> & cursor=<opaque>
  #        format=ndjson|json-array|raw  (default ndjson)
  # Response: decompressed events in the requested range

DELETE /v1/customers/{cid}/datasets/{dsid}/events
  # Query: before_ts=<iso8601>  (compliance: delete events older than timestamp)
```

Intentionally narrow. No:
- Content filters, tags, labels
- Query language
- Full-text search
- Stream pub/sub
- Dashboards

Anything that drifts into Datadog/Axiom/Loki territory is deferred
until we have paying customers asking for it. The positioning is
**"compressed storage, not log analytics."**

### Storage backend

- **Per-customer S3 prefix:** `s3://<bucket>/{cid}/{dsid}/`
- **Time-bucketed compressed blobs** (hour or day granularity):
  `{cid}/{dsid}/YYYY/MM/DD/HH/{batch-uuid}.bin`
- **Model registry:** `{cid}/{dsid}/models/v{version}.bin` +
  `tokenizer/v{version}.model`
- **Raw holding area for pre-model data:** `{cid}/{dsid}/raw/{uuid}.ndjson`
  (GC'd after compression replaces it)
- **Metadata store:** DynamoDB or Postgres — dataset configs,
  model versions, ratio stats, quota, billing accounting

### File header format

Each compressed blob has a fixed header (evolved from l3tc v4):

```
[4B magic "LA01"] [2B fmt ver] [16B model_version_id]
[8B original_size] [8B compressed_size] [16B sha256 truncated]
[8B start_ts_unix_ms] [8B end_ts_unix_ms]
[1B codec_id] [N*B codec-specific metadata]
```

`codec_id`:
- `0x01` — L3TC specialist (learned model, codec versioned by model_version_id)
- `0x02` — zstd-22 fallback (for bad/untrainable data — see "fallback" below)
- `0x03+` — reserved

Per-file codec means we can mix: early data on zstd before model is ready,
later data on learned model. Decompress handles both transparently.

---

## Ingest lifecycle (the key state machine)

```
PUT /events → body stored raw in {cid}/{dsid}/raw/{uuid}.ndjson
                                    ↓
                  dataset has trained model?
                                  ↓ yes         ↓ no
               compress with model,          queue for compression
               write compressed blob,        when model is ready
               delete raw
                                              ↓
                                 model training completes
                                 → compress all queued raw,
                                   write compressed blobs,
                                   delete raw

PUT on NEW dataset
  → first N GB accumulate in raw/
  → training job kicked off once N GB or T time elapsed
  → model lands in registry, events compressed
  → subsequent PUTs compress inline

Background:
  drift monitor measures ratio on incoming sample
  if ratio degrades > X% from training baseline
    → schedule retrain
    → new model_version_id, new files use it
    → old files stay on old model (perpetual decode support)
```

### Open questions I want to resolve before Spike 2

1. **Minimum training corpus size.** 1 GB was a guess. Spike 1 should
   also ablate: how much data to reach near-asymptotic ratio? 100 MB?
   500 MB? 2 GB? Feeds into "how long until customer sees full
   compression benefit" UX.

2. **Drift detection trigger.** Options:
   - Sample N events from new batch, measure ratio, compare to moving
     baseline. Threshold: retrain if ratio degrades >15%.
   - Compute KL divergence between empirical token distribution now
     vs. at training time. Threshold: retrain if KL > X.
   - Simpler: scheduled monthly retrain on most recent N GB.
   Recommend the first (ratio-based) + monthly safety net.

3. **Fallback behavior when trained model is worse than zstd.** Rare
   (very non-homogeneous data), but possible. Protocol:
   - After training, run benchmark: trained model ratio vs zstd-22 on
     held-out sample.
   - If trained is worse, mark dataset as `codec=zstd_fallback`.
   - Customer gets zstd-level pricing, not our "learned" pricing.
   - Track fraction of datasets on fallback — tells us where our
     tech isn't winning.

4. **Training on shared models across customers (v1.1+).** If many
   customers all ingest nginx-format access logs, a shared
   "nginx-access-log" specialist might amortize training cost and
   improve ratios. Not for MVP. Private-model per dataset is the
   starting default.

---

## Spike plan (before any service code)

Three gates. Each failing kills the next.

### Spike 1: Ratio on real homogeneous logs *(the critical one)*

Take 3 real corpora representative of target customers:
- **Nginx / apache access logs** — Loghub (Zenodo), BGL, HDFS logs
- **JSON event stream** — public Stripe webhook fixtures, or synthesize
  from a public schema
- **Syslog / kubelet / app logs** — Loghub OpenStack logs

For each:
1. Split 80/20 train/test
2. Train a fresh 2L × 96H × 16K specialist on 80% using existing
   `scripts/train_l3tc_phase11.py`
3. Measure on 20%: ratio of trained model vs `zstd -22 --long=27` vs
   `zstd --train=... --maxdict=<N>MB` (zstd with trained dictionary)
4. Also ratio vs `gzip -9`, just for landscape context

**Pass criterion: ≥2× ratio improvement over `zstd -22` on at least
2 of 3 corpora.** Strong signal: 3-5× on any.

**Failure mode to watch for:** if our ratio is only 1.2-1.5× zstd-22
on real logs, the service economics don't pencil out and we should
stop.

**Timeline: 3-5 days** including data gathering, training 3 models
(each ~4 hrs on cloud GPU), and measurement.

### Spike 2: Per-customer training cost + corpus-size curve

Using a corpus from Spike 1 (the best-performing):
1. Ablate training corpus size: 100 MB, 500 MB, 1 GB, 2 GB, 5 GB
2. Measure: ratio at each size, training time, training cost
   (cloud GPU hours × price)
3. Check ratio stability on data 30 days later (if data is available;
   otherwise synthesize drift by mixing 2 sources)

**Pass criteria:**
- Ratio reaches 90% of asymptote by ≤1 GB
- Training cost ≤ $50/model on spot GPU
- Ratio holds within 10% on mildly drifted data

**Timeline: 2-3 days**

### Spike 3: End-to-end decompression latency

Stand up a minimum E2E: single EC2, single S3 bucket, single dataset
with a trained model from Spike 1.

1. Compress 10 GB of test data → store in S3
2. On query "give me events from 2pm-3pm yesterday":
   - Fetch relevant time-bucketed blobs from S3
   - Decompress on single Rust process
   - Return NDJSON stream
3. Measure latency for: 1 MB range, 10 MB range, 100 MB range

**Pass criterion: 10 MB range returns in ≤ 5 seconds.** Equivalent
to S3 Standard-IA retrieval UX.

**Timeline: 1 week** (includes writing the minimal service shim).

### Spike 4: Unit economics sheet

Plug measured numbers from Spikes 1-3 into a spreadsheet:
- Customer cost model: GB/yr in, ratio, storage cost, retrieval cost,
  retraining cost (amortized)
- Our cost model: backend storage, compute, bandwidth, platform overhead
- Pricing scenarios: $X/GB-month stored, break-even volume, margin
- Compare to closest competitors: Axiom, Datadog Flex, Elastic Frozen

**Pass criterion: can price at ≥30% below closest direct competitor
while maintaining >30% gross margin.**

**Timeline: 1 day**

### If all four pass

Start building the real MVP (a week of API + ingest pipeline + model
training worker). At that point the decision is product/go-to-market,
not tech risk.

---

## Target customer profile (same as the original doc, still holds)

- **Petabyte-scale** text-or-text-like data (tens of TB minimum,
  sweet spot at low PB)
- **Long retention** (5-30 years) — compression cost amortizes
- **Rare retrieval** (<5% of data per year)
- **Seconds-acceptable retrieval SLA** (NOT hours — this is our
  meaningful improvement over Glacier Deep Archive)
- **Homogeneous data within a dataset** — so per-dataset model
  specializes hard

Best verticals:
- Compliance archives (finance, healthcare, law): 7-30yr retention
- SaaS platform log archival
- Enterprise document / email / ticket archives
- Scientific / sequencing / genomic data (where text-heavy)

Worst fits:
- Heterogeneous general text (we won't beat zstd much)
- Binary / already-compressed / encrypted payloads (we can't help)
- Interactive sub-100ms retrieval requirements

---

## Scope boundaries (what we are NOT)

- **Not a log analytics product.** No search, no queries, no
  dashboards. That's Datadog/Splunk/Axiom territory.
- **Not a backup product.** We don't do versioning, snapshots, PITR.
  That's a separate product.
- **Not an ETL.** We don't transform data. In is out.
- **Not SIEM.** No alerting, correlation, detection.
- **Not a database.** No transactions, no consistency guarantees
  beyond single-object.

The entire pitch is: **take the existing S3-like mental model, make
the storage bill dramatically cheaper for text-heavy data, preserve
sub-second retrieval latency.** That's it.

---

## What carries over from l3tc-prod

Substantial. The compression and training technology is all directly
reusable:

- `scripts/train_l3tc_phase11.py` — RWKV-v4 training loop
- `scripts/convert_checkpoint.py` — `.pth` → runtime format
- `scripts/retokenize_corpus.py` — parallel tokenization pipeline
- `l3tc-rust/src/rwkv.rs` — Rust inference runtime
- `l3tc-rust/src/backend/mtl.rs` + `batched.rs` — Metal path (for
  eventual GPU batch inference on the service backend)
- `l3tc-rust/src/tensor.rs` — NEON INT8 matvec
- `l3tc-rust/src/checkpoint.rs` — .bin loader
- `vendor/L3TC/` — reference implementation
- Phase 11 corpus build + spot fleet infra — runs the per-customer
  training jobs
- The v4 file format + header design — evolves into the per-blob
  header above

What's new to build:

- REST API service (FastAPI or Axum)
- Per-customer + per-dataset job orchestrator
- S3 object lifecycle (raw → compressed, GC, versioning)
- Metadata store (DynamoDB / Postgres)
- Training worker pool (ECS / EKS)
- Drift monitoring
- Auth (API keys for MVP, OAuth later)
- Billing / metering
- Compliance scaffolding (SOC2, HIPAA)

---

## Open questions for GTM (not engineering)

1. **Inside sales or self-serve?** Target customer (petabyte-scale,
   compliance-driven) is inside-sales. But a self-serve tier
   (hobbyist, <100 GB) would demo the product publicly and build
   credibility.
2. **Pricing model.** $/GB-month stored (Axiom style), or $/GB-input
   with retention commitment (archive style), or $/M events (Datadog
   style)? Favors different customer psychology.
3. **First design partner.** Who do we know in the vertical? Do we
   have a friend at a Stripe-shaped company with multi-TB webhook
   traffic?
4. **S3-compatible gateway or custom SDK?** Gateway is 10x more
   valuable for adoption but 10x more effort to build right.
5. **Geographic + compliance footprint at v1.** US-only? Single AWS
   region? How much of compliance is MVP-blocker vs. contract-clause?

These matter once Spikes 1-3 pass. Park them until then.

---

## Honest risks

1. **Spike 1 could fail.** Real logs don't compress as well as
   Wikipedia for neural compressors. Hashes, UUIDs, IP addresses
   are high-entropy. Homogeneous structural patterns are what we
   win on. If real log corpora don't have enough structural
   regularity beyond what zstd catches, the ratio thesis dies.
2. **Customer onboarding complexity.** PUT → train → compress is a
   multi-hour flow on first load. UX design matters; this isn't
   literally S3.
3. **Compliance is real work.** SOC2 Type 2 is ~$50-300K + 9-12
   months. Required by the target market.
4. **A competitor bolts neural compression onto their platform.**
   Axiom, Datadog, or Elastic could do this with 2-3 ML engineers
   in 6 months. The technical moat compresses fast; the operational
   moat (per-customer models, training pipeline, model registry)
   takes longer to replicate but isn't infinite.
5. **Sales cycle length.** Petabyte-scale enterprise archive
   decisions are 6-12 month procurement.
6. **Egress / bandwidth.** If retrieval volume is high, network
   egress from S3 eats margin. Priced into unit economics but a
   variable that goes sideways under heavy use.

---

## Relationship to the archived l3tc-prod docs

The original `docs/STORAGE_SERVICE_VISION.md` in l3tc-prod was
written during Phase 4b of the compression-CLI work as a "if we
ever pivot, here's the thinking." This document supersedes it; the
original is preserved in l3tc-prod's git history for provenance.

Key shifts vs. the original:

1. **Usability over cost economics.** Original anchored on Deep
   Archive pricing comparisons that made the service look
   uneconomical at CPU-only speeds. Reality: the competition isn't
   Deep Archive (12-48h retrieval), it's Axiom / Datadog Flex /
   Elastic Frozen (all seconds-SLA, all generic compression). Our
   value is comparable UX at dramatically lower price.

2. **Concrete MVP API surface.** Original was exploratory; this
   doc specifies PUT/GET/DELETE + dataset lifecycle explicitly
   following Loki's pattern.

3. **Spike plan with pass/fail criteria.** Four gated experiments
   instead of handwaving.

4. **GPU-batch-inference isn't the critical path.** Original
   positioned it as the binding constraint. Reality: CPU inference
   is fine for MVP + small-to-mid customers. GPU matters for
   petabyte-scale unit economics but can be added later as a
   backend optimization.

5. **Acknowledgment that competitors closed part of the gap.** Axiom,
   Datadog Flex, Elastic Frozen all launched "cheap searchable cold"
   since the original doc. We're no longer inventing the category;
   we're plugging learned compression into it.

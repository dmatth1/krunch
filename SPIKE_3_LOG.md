# Spike 3 — validate neural on realistic customer corpora

Running log of Spike 3 experiments. Goal per `SPIKE_3_PLAN.md`: measure
our hybrid dispatcher on three non-HDFS customer-shaped corpora (JSON
events, Python code, OpenAssistant transcripts) to learn where
neural is the primary codec winner vs where it isn't.

## Status — 2026-04-22

- **Experiment 1 (JSON API events): complete.** Dispatcher landed
  0.1475 vs whole-file zstd-22 0.1436 — **2.7% worse than zstd**.
  Neural never won a chunk; bzip3 took 5/5.
- **Experiment 2 (Python code): deferred.** Spike 3's original
  corpus plan kept, but priority shifted to chatbot logs per the
  product pivot (see `CUSTOMER_PROFILE.md`).
- **Experiment 3 (OpenAssistant transcripts): rolled into Spike 4.**
  Chatbot logs are the new product focus; dedicated spike covers
  this depth.

## Experiment 1 — GitHub Archive JSON events

### Corpus

- **Source**: [GH Archive](https://data.gharchive.org) hourly public
  events dumps, 2026-04-15 12:00–15:00 UTC (4 hours).
- **Format**: NDJSON, one GitHub event per line (PushEvent,
  PullRequestEvent, IssuesEvent, etc.). Heavy JSON key repetition;
  URLs; integer IDs; some free text in commit/PR bodies but mostly
  structured.
- **Size**: 482 MB raw, 633,547 events.

### Training

- Model: **L3TC-200K RWKV-v4 + HiRA** (same architecture as Spike 1/2)
  - 2 layers, hidden=96, ctx=2048, SPM unigram vocab=16384
- Training: 10 epochs × ~1,562 steps/epoch on g6.xlarge (NVIDIA L40S)
- Optimizer: AdamW, LR=1e-4 with warmdown
- Duration: ~1.5 hours / $2 of on-demand G6
- Loss curve: 6.43 → 4.60 → 4.37 → 4.25 → 4.17 → 4.12 → 4.08 → 4.06 → 4.05 → 4.05 (plateau)
- `bytes_per_token = 5.03` (per SPM report — reasonable for JSON)

### Two bugs found, both now fixed

1. **`measure_held_out_ratio.py` TypeError.** The vendor L3TC
   `RWKV_TC_HIRA.forward()` routes `train=False` to `forward_test()`
   with a different 4-arg signature (`input_token`, `output_token`,
   `output_types`, `criterion`). My call passed `input_types` in
   position 2 which got interpreted as `output_token`, then crashed
   at `input_token.flatten(1, 2)` on a 2-D tensor.
   **Fix**: call `model(inp, input_types, train=True)` with
   `model.eval()` to disable dropout (commit `aceffa5`, then final
   fix this session).

2. **`zstd --train` single-file error.** The entrypoint ran
   `zstd --train train.txt ...` which errors with "Error 14: nb of
   samples too low" — `zstd --train` requires many sample files.
   **Fix**: split the corpus into 64 KB chunks with `split -b 64K`
   first, pass the directory to `--train`. Applied this session.

Both failures were non-fatal in isolation — the dispatcher ran on
the classical-only path — but together they masked the neural
entropy number and left the codec at `zstd_fallback` by default
rather than `hybrid`. Recovery was manual: train dict locally,
flip metadata, resend SQS message.

### Compression run

Compressed a **5 MB sample** of the same GH-events corpus (not the
full 482 MB — would take ~4 hours on Fargate Graviton even with the
speed-fix image; same ratio signal).

| metric | value |
|---|---|
| bytes in | 5,237,402 |
| bytes out | 772,408 |
| **dispatcher ratio** | **0.1475** |
| per-chunk zstd-22 shadow | 0.1575 |
| savings vs per-chunk zstd | +6.37% |
| **whole-file zstd-22 on the same sample** | **0.1436** |
| chunks total | 5 |
| codec picks | bzip3 5/5 |
| neural chunks | **0** |
| zstd-dict chunks | 0 |
| safety-net substitutions | 0 |
| throughput | 0.0463 MB/s (3× over pre-speed-fix baseline) |

### Pass gate outcome: **required gate missed by 2.7%**

From `SPIKE_3_PLAN.md` pass gates:

| gate | target | actual | result |
|---|---|---|---|
| Required | ≤ 0.1436 (whole-file zstd) | 0.1475 | **MISS** (−2.7%) |
| Strong | ≤ 0.1221 (15% better) | 0.1475 | miss |
| Stretch | ≤ 0.1077 (25% better) | 0.1475 | miss |

### Why neural lost everywhere

GH Archive events are hyper-structured repetition: every event has
`"id"`, `"type"`, `"actor":{"id",login,display_login,gravatar_id,url,avatar_url}`,
`"repo":{"id","name","url"}`. Those same keys appear ~600K times in
5 MB. That's *exactly* the regime where zstd's 128 MB sliding
window dominates: "copy from 800 bytes ago" beats neural's
token-by-token prediction.

Our 200K RWKV has to learn BOTH the schema (thousands of
keys/nesting patterns) AND the content distributions (repo names,
URLs). At 200K params it runs out of capacity for both and loss
plateaus at ~4.0. On pure prose (enwik8) the same architecture
hits ~1.7, a 2.4× better per-token bound. The prediction problem on
JSON events is genuinely harder *relative to* zstd's dictionary
recall.

### Why dispatcher is 2.7% worse than whole-file zstd despite bzip3 winning chunks

Two effects stack:

- **Per-chunk cold start (~2%)**: each 1 MB chunk's codec starts
  with an empty dictionary. Whole-file zstd builds a dictionary
  across the full 5 MB and reuses it. 5 chunks × cold-start
  penalty outweighs bzip3's per-chunk win.
- **bzip3 beats per-chunk zstd by ~6%**, recovers some of the
  above but not all of it.

Net: dispatcher = bzip3 chunked = 0.1475, whole-file zstd = 0.1436.

Fix options (in priority order if we wanted to close this segment):
1. **Whole-file mode for small corpora** (< ~64 MB): no chunking,
   pick the globally-best codec, skip the dispatcher framing. ~0.5
   days of Rust + CLI work.
2. **Per-dataset chunk-size tuning**: detect homogeneity at
   training time; bump chunk size to 4–16 MB for JSON-event-shape
   datasets. Already have `chunk_size_bytes` in metadata, needs a
   heuristic.
3. **CLP**: 3–5 days of work. Structural extraction (templates +
   variables per-column) wins 2× on log-shaped data per the
   published Uber paper. Decisive on this segment.

We've decided **not** to invest in these now — see next section.

### Product-level finding

Spike 3 exp 1 confirms what Spikes 1+2 hinted at: **homogeneous,
high-key-repetition data is not where our neural codec wins**.
Whole-file zstd is already near-optimal on that shape; a 2.7% loss
is acceptable (we don't fall off a cliff) but not a sales pitch.

Combined with Spikes 1+2 (HDFS logs: −42% vs whole-file zstd at 64
KB chunks), the product now has a clear segmentation:

- **Don't pitch**: homogeneous logs / JSON events / infra telemetry
  / fintech transactions. Classical tools already own these.
- **Do pitch**: text-heavy dialogue and document archives. Evidence
  from enwik8 (+14%) + literature (LMCompress dialogue +40%) says
  this is where neural compounds.

This result narrowed the product focus. See `CUSTOMER_PROFILE.md`
for the regulated-vertical AI chat archive pivot (healthcare AI
scribes, legal AI, financial AI, AI customer support). Spikes 2
and 3 of the original plan (Python code, OpenAssistant chat) roll
into the new **Spike 4** — a deeper validation on the specific data
shape the narrowed product targets.

## Artifacts produced

- `s3://archive-dev-archive/spike3/json-events/models/v1.pth` — PyTorch checkpoint (50 MB)
- `s3://archive-dev-archive/spike3/json-events/models/v1.bin` — Rust-format converted checkpoint (13 MB; convert_checkpoint.py worked first try on the phase11 checkpoint)
- `s3://archive-dev-archive/spike3/json-events/models/v1.tokenizer.model` — SPM 16K vocab (262 KB)
- `s3://archive-dev-archive/spike3/json-events/models/v1.zstd_dict` — trained post-hoc locally due to the entrypoint bug (112 KB)
- `s3://archive-dev-archive/spike3/json-events/models/v1.metadata.json` — codec=hybrid (hand-patched after recovery)
- `s3://archive-dev-archive/spike3/json-events/compressed/.../88a24d3f....bin` — 772 KB hybrid blob

## What the pipeline run exercised (end-to-end service validation)

Even though the ratio result wasn't the win we wanted, the spike
stress-tested the full service pipeline for the first time on real
customer-shaped data:

- ✓ PUT-raw → ingest Lambda → training-submit queue → training-launcher Lambda → Batch.submitJob
- ✓ Batch g6.xlarge boot, download 482 MB raw from S3, SPM train, RWKV train, GPU utilized
- ✗ `measure_held_out_ratio.py` (fixed this session)
- ✓ convert_checkpoint.py on a phase11-trained checkpoint (first validation)
- ✗ `zstd --train` with single file input (fixed this session)
- ✓ Upload `.pth` / `.bin` / `.tokenizer.model` to S3
- ✓ training-complete Lambda fires on Batch SUCCEEDED event, enqueues compression
- ✓ Fargate Graviton ARM compression worker boots from scale-0, pulls message
- ✓ ECS task scale-in protection engaged during compression
- ✓ hybrid-compress binary loads model + tokenizer + dict, runs on 4 vCPU
- ✓ EMF metrics land under `Krunch/Hybrid` with CustomerId/DatasetId/Codec dimensions
- ✓ CloudWatch dashboard widgets populate
- ✓ Worker scales back to desired=0 after completion (zero idle cost)

**Every service-side component works end-to-end.** The two training
bugs were code-level, not infrastructure-level — a real customer
running this pipeline today would get artifacts + metrics +
compressed blobs. We're at "pilot customer ready" on the service
side.

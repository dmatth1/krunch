# Spike 3 — validate neural-first compression on realistic customer corpora

## Why this spike exists

Spike 1 + Spike 2 ran 10+ training experiments on HDFS logs. The
answer: our RWKV + AC pipeline loses to zstd on HDFS because HDFS
is pathologically templated. The revised product thesis (per user
direction + supporting research):

**Neural is the primary codec for text-like content. Classical codecs
are dispatched for specific content types where classical genuinely
wins (templated logs, near-duplicates, binary).**

Spike 3 validates this by measuring our pipeline on **corpora where
neural should win** — the complement of HDFS.

## What we already have (skip re-running)

From `bench/results/enwik8-l3tc.md` and `bench/results/enwik8-classical.md`
(pre-pivot era, same L3TC-200K pipeline we're measuring now):

| codec | enwik8 ratio | compress | decompress |
|---|---|---|---|
| RWKV 200K + SPM + AC | **0.2166** | 110 KB/s | 117 KB/s |
| xz -9e | 0.2483 | 1.09 MB/s | 83.46 MB/s |
| zstd -22 --long=27 | 0.2527 | 0.93 MB/s | 779.67 MB/s |
| bzip2 -9 | 0.2901 | 17.57 MB/s | 39.38 MB/s |

**On prose, we already beat zstd-22 by 14.3% (0.2166 vs 0.2527)
with round-trip verified.** That's a real win on a canonical LM
benchmark. Decompress is 117 KB/s single-stream on Apple Silicon —
below the 1 MB/s floor, which means customer retrieval on prose
needs GPU batching or tolerates tens-of-seconds latency. Both are
acceptable per our stated envelope.

Skipping the enwik8 experiment. Proceed directly to the corpora we
haven't measured.

## Pass gate

On each corpus:
- **Required**: our held-out ratio ≤ `zstd --long=27 -22` ratio
  (loose — "don't lose").
- **Strong**: our ratio ≤ 0.85 × zstd-22 (≥ 15% smaller), comparable
  to the enwik8 result.
- **Stretch**: ≤ 0.75 × zstd-22 (≥ 25% smaller).

Failing the required gate on ≥1 corpus is a signal that the content
type should *not* be a neural-primary dispatch and instead go to the
classical path.

## Corpora (three, all non-HDFS-shape)

Ordered by diagnostic value (fastest-feedback first):

| # | corpus | size | shape | why |
|---|---|---|---|---|
| 1 | **GitHub webhooks / JSON API events** | ~1 GB | JSON event stream with repeated keys + free-text bodies | The biggest realistic customer target — SaaS audit + event archives — and a stress test for Tier 1 dispatcher (some chunks will look log-shaped, some prose-shaped). |
| 2 | **Python source code** (codeparrot/github-code subset) | ~500 MB | Structured + semantic | Validates neural beyond prose. Different vocabulary class, different prediction profile. |
| 3 | **OpenAssistant conversations / transcripts** | ~500 MB | Long-form text with technical content | Dense vocabulary, good LM territory. Chat/email/ticket archives are a realistic compliance target. |

Skipped:
- enwik8 (data exists, ratio 0.2166 validated)
- HDFS variants (Spike 1 + 2 exhausted that corpus)
- Binary / pre-compressed (out of scope for neural)
- Near-duplicate docs (dispatched to brotli shared-dict, out of scope for model measurement)

## Baselines to record per corpus

For every experiment:
- `zstd -3` (fast default, reference floor)
- `zstd -22 --long=27` (our pass-gate baseline)
- `bzip3 -16` (classical high-ratio, known strong on text)
- `xz -9e` (optional; bzip3 has overtaken it in 2025 benchmarks)
- **Our RWKV 200K + SPM + AC via `measure_held_out_ratio.py`**
- (Optional) our RWKV 3M (C5 architecture) for capacity sensitivity

All ratios on the same val split (20% of corpus, byte-offset split).
Record compress MB/s and peak RAM per codec for the inference-envelope
plot.

## Experiments

### Experiment 1 — JSON API events (≤ 2 hr training, ≤ $2)

- **Corpus**: Fetch a realistic JSON event stream. Options:
  - GitHub public events archive (gharchive.org) — free, 1 GB/day available.
  - Synthesized Stripe-style webhook payloads.
  - Public Loghub JSON-formatted event samples.
- **Config**: Spike 1 baseline (16 K vocab, 2 L × 96 H × 2048 ctx,
  batch 32, 10 epochs). Same architecture we've been measuring.
- **Predicted ratio**: 0.18–0.28 (zstd-22 typically 0.25–0.40 on
  JSON-with-text).

Decision after experiment 1:
- Neural wins ≥ 15% over zstd-22: validates the thesis on event streams.
- Neural within ±5% of zstd-22: neural still acceptable as primary
  for JSON; dispatcher will pick zstd-dict on the chunks where
  zstd-dict wins.
- Neural loses by >10% on JSON: concerning, but not fatal — JSON has
  structured repetition that could go to CLP or zstd-dict; we'd dispatch
  those chunks to classical.

### Experiment 2 — Python source code (≤ 1.5 hr, ≤ $1.50)

- **Corpus**: ~500 MB from `codeparrot/github-code` filtered to Python.
- **Config**: Same as experiment 1.
- **Predicted ratio**: 0.20–0.28 (vs zstd-22 ~0.28–0.33 on code).

### Experiment 3 — OpenAssistant transcripts (≤ 1.5 hr, ≤ $1.50)

- **Corpus**: OpenAssistant's public conversation dump (Apache-2.0).
- **Config**: Same.
- **Predicted ratio**: 0.20–0.28.

## What we'll report

Markdown table per corpus in `SPIKE_3_LOG.md`:

| codec | ratio | compress MB/s | decompress MB/s | peak RAM |
|---|---|---|---|---|
| zstd -3 | | | | |
| zstd -22 --long=27 | | | | |
| bzip3 -16 | | | | |
| xz -9e | | | | |
| RWKV 200K + SPM + AC | | | | |

Plus one combined Pareto plot across all three corpora (+ the
existing enwik8 + HDFS data) showing `ratio` vs `compress
throughput` log-log, with zstd-22 as the anchor. This is the one
chart that makes the "when to pick neural vs classical" design
concrete.

## Decision this spike unlocks

| outcome | next step |
|---|---|
| All three corpora win by ≥ 15% vs zstd-22 | Commit to neural-primary dispatcher; start Tier 1 engineering per `HYBRID_CODEC_DESIGN.md`. |
| 2/3 win, 1 marginal | Commit; the marginal corpus (if structural) goes through the dispatcher's probe-encode path where classical can win specific chunks. |
| 1/3 win, 2 lose | Architecture signal — neural is too narrow. Revisit recipe (BPE tokenizer? bigger ctx? Mamba-2?) before building dispatcher. |
| 0/3 win | Neural is only useful on prose; product needs rethink or a materially better model. |

## Infrastructure reuse

All experiments run on the existing Spike 2 infrastructure:
- Existing training Docker image (Phase C4b already supports the configs we need).
- CodeBuild pipeline for any needed image rebuilds.
- Batch job submission with container env overrides.
- S3 + DDB + training-complete Lambda flow.

No new engineering required before running these experiments.

## Execution order

1. Fetch + prepare JSON API event corpus; upload to S3 at
   `acme/json-events/raw/`.
2. Submit training run #1 (JSON events) with baseline config; measure.
3. Fetch Python code corpus; upload; run; measure.
4. Fetch OpenAssistant corpus; upload; run; measure.
5. Compile `SPIKE_3_LOG.md` with per-corpus tables + the Pareto plot.
6. Make the design call.

Estimated total: ~1 day of AWS compute (~$5–6), ~3–4 hours of local
corpus prep + writeup.

## Out of scope (for later spikes / work)

- Detector engineering (Spike 4 / Tier 1 implementation).
- Dispatcher + blob format (Spike 4).
- CLP IR port (Spike 5 / Tier 1 continuation).
- Tier 2 field-aware codec for structured data (later).
- Fargate-worker changes to download per-dataset zstd dict (Tier 1 engineering).

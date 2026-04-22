# Spike 4 — validate neural on chat / dialogue corpora

## Why this spike

The product focus narrowed on 2026-04-22 to **regulated-vertical AI
chat archives** (see `CUSTOMER_PROFILE.md`): healthcare AI scribes,
legal AI, financial AI, AI customer support. All four segments
store the same data shape — turn-by-turn user ↔ LLM dialogue —
and all share the same retention pressure (HIPAA 6y, FINRA 3–7y,
attorney-client 7y+).

Spike 3 experiment 1 validated that our dispatcher does NOT win on
homogeneous repetitive data (GH events JSON: −2.7% vs whole-file
zstd). The remaining product question is whether it **decisively
wins** on the data shape we're actually targeting.

This spike answers that. Spike 3 experiments 2 and 3 (Python code,
OpenAssistant transcripts) from `SPIKE_3_PLAN.md` fold into this
spike — chat dialogue is the priority target given the pivot.

Pass here = the product thesis has a defensible empirical basis.
Fail = the thesis is broken and the pivot was wrong.

## Corpora

Two publicly available, reproducible corpora, ordered by
diagnostic value. Both are real LLM chat transcripts — the exact
data shape our target customers retain.

### Corpus A: OASST2 (`OpenAssistant/oasst2`)

- **Source**: [OpenAssistant/oasst2 on Hugging Face](https://huggingface.co/datasets/OpenAssistant/oasst2)
- **Content**: human-generated conversations with OpenAssistant-style
  assistant, 208,584 messages across 70,642 conversation trees. 101
  languages (English dominant). Quality-annotated.
- **Size**: ~150–250 MB raw (estimate; HF API reports message count
  but not bytes)
- **Character**: assistant-style Q&A, multi-turn, mostly English
  prose. Analog to what a legal-AI or healthcare-AI platform
  would archive: user asks, model answers at length.

### Corpus B: LMSYS-Chat-1M (`lmsys/lmsys-chat-1m`)

- **Source**: [lmsys/lmsys-chat-1m on Hugging Face](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
- **Content**: 1,000,000 real conversations between humans and 25
  LLMs (Vicuna demo + Chatbot Arena). English heavy, 5-month
  collection window in 2023. Public, reproducible.
- **Size**: ~2–3 GB raw estimated (1M × avg 6 turns × avg ~500
  bytes/turn)
- **Character**: most realistic proxy for production LLM chat
  logs — diverse user questions, multiple model styles of
  response. Closest to what a multi-tenant AI customer support
  platform would have.

### Corpus choice rationale

**OASST2 first** (shake-down, ~200 MB): small enough to train a
model in <1 hour on g6.xlarge and run full compression in <15 min
on 4-vCPU ARM Fargate. Fast feedback loop.

**LMSYS-Chat-1M second** (real test, ~1 GB English subset): at
training-realistic scale, confirms OASST2 result isn't a small-corpus
artifact. Filter to English-only to avoid multilingual vocab
blowup in the 16K SPM model.

Skipping the older Spike 3 corpora (Python code, pure
OpenAssistant) — the pivot targets dialogue specifically, not
general text.

## Pass gates (stricter than Spike 3)

Measured on the dispatcher's output ratio vs whole-file zstd-22
`--long=27` on the same val split:

| gate | target | interpretation |
|---|---|---|
| Required | dispatcher ratio ≤ **0.85 ×** zstd-22 (≥ 15% smaller) | the product thesis floor; anything less and we're not actually winning |
| Strong | dispatcher ratio ≤ **0.75 ×** zstd-22 (≥ 25% smaller) | matches LMCompress dialogue literature; pitchable result |
| Stretch | dispatcher ratio ≤ **0.65 ×** zstd-22 (≥ 35% smaller) | punches above 200K weight class |

**Why stricter than Spike 3**: Spike 3's "don't lose to zstd" was
the floor for a broad-horizontal product. Post-pivot we're pitching
"we beat zstd by 20–40% on your archives." Required gate ≥ 15% is
what makes that claim defensible. If we miss this, the narrow
focus pitch collapses.

### Expected outcome (my predictions)

From published literature + our own enwik8 measurement:

| corpus | expected dispatcher ratio | vs zstd-22 | confidence |
|---|---|---|---|
| OASST2 | 0.15–0.22 | 20–35% better | high — English dialogue is the L3TC sweet spot |
| LMSYS-Chat-1M | 0.16–0.24 | 18–32% better | medium — more content diversity, per-user model may not generalize as cleanly |

These predictions are informed by:
- `bench/results/enwik8-l3tc.md`: 200K RWKV on enwik8 prose → 14%
  better than zstd (our own measurement, same architecture).
- [LMCompress paper (Nature MI 2025)](https://www.nature.com/articles/s42256-025-01033-7):
  LLM compressor on dialogue → ~40% better than zstd (bigger model).
  Our 200K model should close some but not all of that gap.
- [DeepMind "Language Modeling Is Compression" (ICLR 2024)](https://arxiv.org/abs/2309.10668):
  modest transformers on text → 30–40% better than zstd.

## Infrastructure reuse

All the plumbing from Spikes 1–3 + Phase 1 stays:
- Training: same Batch job definition (jobdef rev 17 after the two
  bug fixes), same g6.xlarge compute env, same training entrypoint.
- Service loop: PUT raw → ingest Lambda → training-submit SQS →
  training-launcher Lambda → Batch → `.pth` + `.bin` + `.zstd_dict` +
  metadata → training-complete Lambda → compression SQS → Fargate
  Graviton ARM worker → hybrid-compress → EMF metrics +
  CloudWatch dashboard.
- Bug fixes applied this session (must-land in a fresh training
  image before running):
  - `measure_held_out_ratio.py` calls `model(inp, input_types,
    train=True)` under `model.eval()` instead of `train=False`
  - `zstd --train` now feeds many 64 KB sample files instead of
    one corpus

## Execution plan

### Step 1 — rebuild training image + register jobdef

CodeBuild a fresh training image carrying the two bug fixes (commit
after this plan lands). Register as jobdef revision 18. ~10 min
build + 10 sec register.

### Step 2 — OASST2 experiment (~1 hr training + ~5 min compression)

1. Download OASST2 "ready" split from Hugging Face → flatten to
   NDJSON locally → upload to
   `s3://archive-dev-archive/spike4/oasst2/raw/`
2. Submit training via SQS to `archive-dev-training-submit` with
   `{cid: spike4, dsid: oasst2, trigger: initial}`. Defaults are
   fine (vocab 16K, 2L×96h, 10 epochs).
3. Monitor via rich-log monitor. Expect: SPM + tokenize in ~5 min,
   RWKV train in ~45 min, measure + convert + zstd-dict in ~2 min,
   upload + compression enqueue automatic.
4. Compression worker runs on the full corpus (should be <15 min
   at ~200 MB / 0.25 MB/s if the speed-fixes compound).
5. Read EMF metrics + compare to zstd-22 baseline.

### Step 3 — LMSYS-Chat-1M experiment (~2 hr training + ~1 hr compression)

If Step 2 hits the required gate, scale up to the 1 M conversation
corpus (English subset). Same flow, bigger data.

If Step 2 **misses** the required gate, STOP before Step 3 and
re-examine. Possible causes + fixes before spending 2 hours:
- 200K too small → bump to 1M params (bigger hidden_size)
- Chunk size too small → bump to 4 MB
- SPM vocab wrong for dialogue → retrain with different
  `max_piece_length`

### Step 4 — writeup

`SPIKE_4_LOG.md` with:
- Per-corpus table: ratio, zstd shadow, savings %, codec
  distribution, throughput
- Pareto plot vs the existing enwik8 + HDFS + GH-events data
  points (5 datapoints total across the three spikes)
- Decision: does the narrowed product pitch hold empirically?

## Cost estimate

| step | compute | wall | $ |
|---|---|---|---|
| Rebuild training image | CodeBuild | 10 min | ~$0.10 |
| OASST2 training | g6.xlarge on-demand | ~1 hr | ~$1.20 |
| OASST2 compression | Fargate 4 vCPU ARM | ~15 min | ~$0.02 |
| LMSYS training (if gate hit) | g6.xlarge on-demand | ~2 hr | ~$2.40 |
| LMSYS compression (if gate hit) | Fargate 4 vCPU ARM | ~1 hr | ~$0.08 |
| **Total if full plan runs** | | ~4 hr wall | **~$3.80** |

Negligible $ but half a working day of wall clock.

## Decision tree

| outcome | product impact | next step |
|---|---|---|
| OASST2 + LMSYS both hit STRONG gate (≥ 25%) | Pitch defensible at "20–40% smaller than zstd on chat archives"; first customer outreach (Abridge / Nabla / similar) | Begin compliance cert work (SOC 2 / BAA) + fix the per-conversation-deletion gap called out in CUSTOMER_PROFILE.md |
| Both hit REQUIRED gate (≥ 15%) but miss STRONG | Pitch is "15–25% smaller on chat" — weaker but still meaningful, ~$20K/year savings on a 100 TB Harvey-scale customer | Keep the 200K model but plan a path to 1M–10M params for customers who want more savings |
| OASST2 misses REQUIRED | Thesis problem. Either 200K is too small for dialogue or the architecture is wrong for this shape. Stop before LMSYS. Re-think. | Architecture investigation: bigger model, different tokenizer, rethink segment_bytes |
| OASST2 strong, LMSYS weak | Per-customer specialization works but single-model-for-many-users doesn't. The pitch holds for dedicated customers (most regulated-vertical) but not for multi-tenant platforms | Refine the product to require per-customer models; no shared-model mode |

## Out of scope (defer to later spikes / work)

- Decompression-throughput benchmark on dialogue data (matters for
  the GET endpoint but not pass/fail here)
- Per-conversation deletion engineering (called out in
  `CUSTOMER_PROFILE.md`; not a spike, it's a design task)
- Compliance certifications (BAA, SOC 2 Type II, HITRUST) — ops
  work, 6–12 month timeline
- Bigger neural models (1M+ params) — would change the throughput
  envelope and the inference pipeline

## Success criterion

**Spike 4 succeeds if at least the OASST2 experiment hits the
REQUIRED gate (dispatcher ratio ≤ 0.85 × zstd-22 on the val
split).**

That's the floor for making "we beat zstd 20–40% on AI chat
archives" into a defensible product claim to show customers.

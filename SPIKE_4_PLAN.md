# Spike 4 — validate neural on chat / dialogue corpora

## Update 2026-04-22 — pivot to English-only

The first run (full-language WildChat-4.8M subset, 200 MB) exposed a
multilingual capacity problem: trained held_out_ratio came in at
0.568 (vs zstd 0.185) — the 200K RWKV + 16K SPM unigram couldn't
cover the ~25+ languages in WildChat simultaneously. `bytes_per_token`
dropped to 3.22 (vs GH events' 5.03 and enwik8's ~3.5), suggesting
the tokenizer fragmented into many short multilingual pieces.

This isn't a problem for target customers (Harvey, Abridge, Hebbia
are English-first enterprise) but it's blocking the spike measurement.

**Pivot**: primary experiment is now `WildChat-4.8M filtered to
language=="English"` — same corpus, ~40–60% of the rows.

Also surfaced three training-entrypoint bugs, all fixed this session:
- `measure_held_out_ratio.py` call signature (passed `input_types`
  + `train=True` + `model.eval()` to mirror the trainer)
- `zstd --train` single-file → multi-sample split before training
- Shell capture of measure-script stdout included ninja build noise
  → grep for bare-float-only lines

See `SPIKE_4_LOG.md` for the measurement + bug trail.

Model-size decision for this spike: **staying at 200K params.**
Bigger model (1M / 10M) is the right next move for Spike 5 (GPU
compression path), but mixing English-only with a model-size change
would confound the language fix with the capacity change. Clean
control first, scale second.

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

Two publicly available chat corpora chosen to separate **"neural
wins on dialogue"** from **"neural wins on *production* dialogue
specifically"**. Running both gives us a diagnostic signal
beyond a simple pass/fail.

### Corpus A (primary): WildChat-4.8M English-only (`allenai/WildChat-4.8M`)

- **Source**: [allenai/WildChat-4.8M on Hugging Face](https://huggingface.co/datasets/allenai/WildChat-4.8M)
- **License**: ODC-BY (Open Data Commons Attribution). Ungated,
  scriptable download.
- **Content**: 3,199,860 real conversations between human users
  and ChatGPT, non-toxic filtered, coverage through August 2025.
  **Filter to `language == "English"` at stream time** — the
  multilingual run exposed a capacity problem with 200K params
  across 25+ languages.
- **Character**: **production LLM chat transcripts** — real users
  typing real questions, real GPT-4/3.5 responses, including typos,
  retries, abandoned threads, vocabulary drift. This is the closest
  public analog to what Harvey's attorneys, Abridge's clinicians,
  or Hebbia's analysts actually generate every day. English-only
  also matches the target customer deployment (Fortune 500 US/EU
  enterprise).
- **Size**: ~200 MB subset
- **Why primary**: if the product thesis is "we win on regulated-
  vertical AI chat archives," WildChat-English is the representative
  proxy. Passing on it is the defensible signal; failing kills the
  thesis.
- **Baseline**: zstd-22 on the 200 MB English subset = **0.1526**
  (tighter than the multilingual baseline 0.1723 because English
  has more structural repetition zstd exploits). Pass gates:
  required ≤ 0.1297, strong ≤ 0.1145, stretch ≤ 0.0992.

### Corpus B (diagnostic): OASST2 (`OpenAssistant/oasst2`)

- **Source**: [OpenAssistant/oasst2 on Hugging Face](https://huggingface.co/datasets/OpenAssistant/oasst2)
- **Content**: 208,584 curated human-generated conversations across
  70,642 trees, quality-annotated in 101 languages (English dominant).
- **Character**: **curated assistant-training data** — volunteers
  coached to produce high-quality Q&A, not real production traffic.
  Cleaner, more structured, less noise. Closer to the text the
  model was trained on (OpenAssistant conversations are the
  L3TC-era RWKV training neighborhood) than to what customers
  actually archive.
- **Size**: ~150–250 MB raw
- **Why diagnostic**: if WildChat passes and OASST2 passes, neural
  wins on dialogue broadly (simple product story). If WildChat
  passes and OASST2 fails, production messiness helps us. If
  WildChat fails and OASST2 passes, we can only win on curated
  corpora — a narrower product that requires upstream content
  filtering before ingest. If both fail, the thesis is broken.

### Corpus choice rationale

Two experiments let us read the *shape* of the win, not just
pass/fail. Previous draft plan had OASST2 first for
convenience — WildChat is the correct primary because it matches
the product's target data shape. OASST2 stays as the second
experiment specifically because its curation gap vs WildChat is
the diagnostic signal.

Skipped: LMSYS-Chat-1M (access-gated on HF, not scriptable),
ShareGPT (redistribution-unclear), Python code and raw OpenAssistant
from the original Spike 3 plan (wrong data shape post-pivot).

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
| WildChat-4.8M (production chat) | 0.17–0.25 | 18–30% better | medium — real production diversity is a harder prediction problem than curated data |
| OASST2 (curated) | 0.15–0.22 | 20–35% better | high — English dialogue is the L3TC sweet spot, quality-filtered lowers noise |

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

### Step 2 — WildChat experiment (~1 hr training + ~5–15 min compression)

1. Fetch a ~200 MB subset of WildChat-4.8M from Hugging Face
   (`pip install datasets; load_dataset("allenai/WildChat-4.8M",
   split="train", streaming=True)` → take conversations until
   ~200 MB). Flatten each conversation into a NDJSON record with
   `{conversation_id, turn, role, content, timestamp}` — same
   shape production chat logs use.
2. Upload to `s3://archive-dev-archive/spike4/wildchat/raw/`.
3. Submit training via SQS to `archive-dev-training-submit` with
   `{cid: spike4, dsid: wildchat, trigger: initial}`. Entrypoint
   defaults OK (vocab 16K, 2L × 96 h, 10 epochs, LR 1e-4).
4. Rich monitor tails: expect SPM train ~5 min, corpus tokenize
   ~5 min, RWKV 10 epochs ~45 min, post-training (convert +
   zstd-dict + measure + metadata upload) ~2 min, compression
   auto-enqueued.
5. Compression on the 5 MB val split (or the full raw if we're
   patient) via the 4-vCPU Graviton Fargate path. 5 MB ≈ 3 min at
   current throughput; expand if faster.
6. Read EMF metrics under `Krunch/Hybrid` (CustomerId=spike4,
   DatasetId=wildchat). Compare ratio to whole-file zstd-22 on
   the same val split.

### Step 3 — OASST2 experiment (~45 min training + ~5 min compression)

Run **regardless** of WildChat result (both runs cheap, diagnostic
signal is the point).

1. Download OASST2 "ready" split from Hugging Face
   (`load_dataset("OpenAssistant/oasst2", split="train")`).
2. Flatten conversation trees → NDJSON, same schema as WildChat
   step.
3. Upload to `s3://archive-dev-archive/spike4/oasst2/raw/`, submit
   training, monitor, compress same as Step 2.

### Step 4 — writeup

`SPIKE_4_LOG.md` with:
- Per-corpus table: ratio, zstd shadow, savings %, codec
  distribution, throughput, per-epoch loss curve summary
- Pareto plot: ratio vs throughput across all corpora measured
  so far (enwik8, HDFS, GH events, WildChat, OASST2) — the chart
  that makes the narrowed product claim concrete
- Diagnostic read of WildChat × OASST2 × production-proxy quality
  (see Decision tree below)
- Decision: does the narrowed product pitch hold empirically?
  Update `CUSTOMER_PROFILE.md` if the result refines the segment
  fit.

## Cost estimate

| step | compute | wall | $ |
|---|---|---|---|
| Rebuild training image | CodeBuild | 10 min | ~$0.10 |
| WildChat training | g6.xlarge on-demand | ~1 hr | ~$1.20 |
| WildChat compression | Fargate 4 vCPU ARM | ~5–15 min | ~$0.02 |
| OASST2 training | g6.xlarge on-demand | ~45 min | ~$0.90 |
| OASST2 compression | Fargate 4 vCPU ARM | ~5 min | ~$0.01 |
| **Total** | | ~2–3 hr wall (parallelizable) | **~$2.25** |

Both experiments can run serially on the same Batch queue; the two
training jobs don't share state so they could also run in parallel
if we bump max vCPU on the compute env. For now: serial, one
afternoon.

## Decision tree (WildChat × OASST2 matrix)

| WildChat | OASST2 | read | next step |
|---|---|---|---|
| **STRONG (≥25%)** | **STRONG (≥25%)** | Clean win. "20–40% smaller on AI chat archives" is defensible empirically. | First customer outreach (Abridge / Nabla / Hebbia / similar series-C vertical AI). Begin BAA + SOC 2 work + per-conversation-deletion engineering per `CUSTOMER_PROFILE.md`. |
| STRONG | REQUIRED (15–25%) | Production data is *easier* than curated to compress (unusual but possible — repetitive retry patterns in WildChat help us). Neural wins decisively where it matters most. | Same first-customer outreach. Note: we'd need to tell customers "dirtier data compresses better with us" which is actually a nice talking point. |
| REQUIRED (15–25%) | STRONG | Curated text is easier than production to compress; neural still wins on real chat but less. | Pitch "15–25% smaller on your real chat traffic." Still meaningful — ~$20K/yr savings for a 100 TB Harvey-scale customer. Consider 1M-param model roadmap. |
| REQUIRED | REQUIRED | Thesis holds but at the weak end. Product story is real but less dramatic. | Ship at current quality for early-design-partner customers. Plan path to 1M-param model to expand the margin. |
| **MISS** | STRONG | Production messiness breaks our model. Per-customer training on an individual customer's own data might fix it. | Investigate: try training WildChat in per-conversation-tree splits, or try smaller domain-specific slices. May mean the product REQUIRES true per-customer fine-tuning (which we already have, but maybe with more data). |
| MISS | MISS | **Thesis failure.** 200K model does not decisively beat zstd on real-world dialogue. | Stop the pivot. Architecture investigation: bigger neural (1M–10M params), different tokenizer (e.g. 32K vocab), different backbone (Mamba-2 instead of RWKV-v4). ~1–2 weeks of research before any more customer discussion. |

The two-experiment design specifically exists to distinguish the
last two rows from the first four. A single-corpus spike can't tell
us whether failure is in our model or in the corpus.

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

**Spike 4 succeeds if WildChat hits the REQUIRED gate (dispatcher
ratio ≤ 0.85 × zstd-22 on the val split).** OASST2 is diagnostic
only — a strong OASST2 alone doesn't validate the product because
OASST2 is not the target data shape.

That's the floor for making "we beat zstd by 15–40% on AI chat
archives" into a defensible product claim we can show to Abridge /
Nabla / Hebbia / other first-customer candidates.

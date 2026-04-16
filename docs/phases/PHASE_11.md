# Phase 11 — Broader corpus + universal text compression

**Goal:** train a model that compresses all text and structured
data types (prose, code, JSON, logs, CSV, XML) at ratios
competitive with or better than classical compressors, while
maintaining usable speed.

**Status as of 2026-04-15:** Experiment D complete (6L × 96H ×
32K). 12L run preparing — 64 GB custom corpus built, uploading
to S3. LR scheduler bug fixed. Launch imminent.

---

## Key findings so far

### Tokenizer matters — a lot

| tokenizer | enwik6 B/T | JSON B/T | logs B/T | code B/T |
|---|---:|---:|---:|---:|
| enwik8 SPM 16K | 3.53 | **1.70** | 1.91 | 1.95 |
| Pile SPM 16K | 3.25 | 2.35 | 2.09 | 2.13 |
| **Pile SPM 32K** | **3.50** | **2.57** | **2.44** | **2.22** |

The enwik8 tokenizer is terrible on structured text (1.70 B/T
on JSON = every ~1.7 bytes needs a model prediction). Pile 32K
nearly matches enwik8 on Wikipedia (3.50 vs 3.53) while being
51% more efficient on JSON and 28% on logs. Best of both worlds.

**Downside:** vocab 32K doubles the head matvec + cum_freqs cost
per token (~80 µs vs ~40 µs). Partially offset by fewer tokens
per byte. Net speed ~1.3× slower per byte.

### Model capacity — not tokenization — is the main bottleneck

| experiment | architecture | tokenizer | enwik6 ratio | webster ratio | speed | epoch measured |
|---|---|---|---:|---:|---:|---|
| Default | 2L × 96H (1.03M non-embed) | enwik8 16K | 0.1699 | 1.2613 | 131 KB/s | 20 (L3TC shipped) |
| Pass 2 | 2L × 96H, Pile corpus | enwik8 16K | 0.3156 | 0.4886 | ~130 KB/s | 5 of 10 |
| Pass 3 | 2L × 96H, Pile corpus | Pile 16K | 0.3429 | 0.5676 | 76 KB/s | 2 of 10 |
| **Exp D** | **6L × 96H (3.1M non-embed)** | **Pile 32K** | **0.3349** | **tbd** | **49 KB/s** | **9 of 10** |

Pass 2 proved: 200K non-embed params can learn one domain well
(enwik8 → 0.17) or spread thin (Pile → 0.32 enwik6 / 0.49
webster). Cannot do both. The floor (0.20) was broken.

Pass 3 proved: retraining the tokenizer at the same vocab size
(16K) made things WORSE. Broader tokenizer + same vocab =
more tokens per byte = slower AND worse ratio on every corpus.

### Training recipe

AdamW (weight_decay=0.01) + cosine warmup (500 steps → peak
1e-4 → decay to 1e-6) + bf16 mixed precision. Replaces L3TC's
broken double-stepping StepLR. Validated on enwik8 (5 epochs,
pipeline end-to-end clean, no NaN over 125K steps).

### Cloud training conventions

- **Pre-baked AMI** (`ami-07a4fc98c4ed4e19e`): everything
  installed, boots in ~90 sec. Stale `at` shutdown jobs from
  the bake trigger on reboot — clear them on SSH.
- **Always set auto-shutdown safety net** via `at`.
- **S3 sync via cron every 3 min** (not background subshells —
  those broke from quoting issues).
- **PYTHONUNBUFFERED=1** for visible training logs.
- **On-demand for short runs** (<6 hr), spot for long runs.
- **g5.xlarge** (A10G 24 GB) for ≤6L models. **g6e.xlarge**
  (L40S 48 GB) for 12L+ models at vocab 32K.
- **torch.compile disabled** — L2Wrap can't be traced by dynamo.
- **AMI cost:** $10/month at 200 GB. Delete when Phase 11 complete.

### Eval metric bug (fixed)

`approx_bpb` in `train_l3tc_phase11.py` had a hardcoded 3.5
bytes/token divisor from the enwik8 tokenizer. With different
tokenizers this was wrong. Fixed: now reports `bits_per_token`
(tokenizer-independent). Use actual compression ratio from the
Rust runtime for honest measurement.

---

## Current experiment: D (6L × 96H × vocab 32K)

**Architecture:** 6 layers × 96 hidden × 96 intermediate,
rwkv_rank 4, **vocab 32768**. Total 9.4M params (3.1M
non-embed transformer, 6.3M embed+head).

**Tokenizer:** Pile-trained SPM BPE 32K. Trained locally on
1 GB of raw Pile text. Uploaded to S3.

**Corpus:** 10 GB Pile dedup, re-tokenized with the 32K SPM.
~2.7B tokens. 500K epoch_length × 10 epochs.

**Instance:** g5.xlarge on-demand (A10G 24 GB). Batch 8 +
grad_accum 4. 10.13 it/s. ~102 min/epoch, ~17 hours total.
20-hour auto-shutdown.

**Instance note:** switched from batch 8 / grad_accum 4 (10.13
it/s) to batch 12 / grad_accum 3 (6.98 it/s) at epoch 4 to
use more VRAM. Batch 16 OOM'd — L2Wrap backward allocates
`zeros_like(logits)` which is 4 GB at batch 16 × 2048 × 32768.

**Training progress:**

| epoch | train loss | eval CE (nats) | bits/token |
|---|---:|---:|---:|
| 0 | 5.97 | 3.40 | 4.90 |
| 1 | 5.21 | 3.23 | 4.66 |
| 2 | 5.01 | 3.15 | 4.55 |
| 3 | 4.89 | 3.10 | 4.47 |
| 4 | 4.81 | 3.07 | 4.43 |
| 5 | 4.76 | 3.05 | 4.39 |
| 6 | 4.72 | 3.03 | 4.37 |
| 7 | 4.69 | 3.01 | 4.34 |
| 8 | 4.66 | 2.99 | 4.31 |
| 9 (final) | 4.65 | 2.98 | 4.30 |

**Training complete.** 10 epochs, ~17 hours on g5.xlarge.
Final eval CE: 2.98 nats (projected ~2.95, close).

**Final compression ratios (epoch 9, measured on M1, clean machine):**

| file | type | exp D ratio | default 200K | speed |
|---|---|---:|---:|---:|
| enwik6 | Wikipedia | 0.3349 | 0.1699 | 49 KB/s |
| json_api | structured | 0.3342 | untested | 42 KB/s |
| python_source | code | 0.3544 | 0.4732 | 30 KB/s |
| fiction | fiction | 0.4074 | 0.4161 | 31 KB/s |
| nginx_log | logs | 0.4717 | untested | 37 KB/s |
| csv_data | tabular | 0.5937 | untested | 31 KB/s |
| c_source | C code | 0.6130 | 0.5535 | 7 KB/s |
| html | markup | 1.0011 | 1.0011 | — |
| xml_silesia | markup | — | — | non-UTF8 error |

**Key findings:**
- JSON is now the **best domain** (0.334) — better than enwik6
- Ratios converged: enwik6/JSON/python all 0.33-0.35
- Python source 25% better than enwik8-specialized 200K model
- CSV (0.59) and logs (0.47) lag — need more training data
  in these domains (motivates the custom corpus for 12L)
- HTML still broken (OOD, ratio ≥1.0)
- xml_silesia eval file has non-UTF8 bytes — needs a clean file
- Speed 30-49 KB/s across domains (2.8× slower than 200K,
  expected from 3× layers + 2× vocab). See
  `docs/SPEED_OPTIMIZATIONS.md` for improvement opportunities.

**Speed note:** early measurements showed ~9 KB/s due to a
concurrent Pile download eating CPU. Clean-machine benchmarks
match the theoretical ~49 KB/s. See `docs/EVALUATION_GUIDELINES.md`.

Checkpoints: `s3://dmatth1-bnn-checkpoints/l3tc/experiment_d_6l_32k/`
(epochs 0-9 + latest). Instance terminated.

---

## Pivot: speed-first generalist (2L × 96H × 32K)

**Decision (2026-04-15):** the project goal is a CLI tool that
ships fast lossless compression. Speed is a hard constraint,
not a soft one. Larger models (6L, 12L) progressively eroded
the speed advantage that originally motivated this work
(Default 200K hits 130 KB/s; Exp D 6L hits 49 KB/s; 12L
projected at 25-35 KB/s). Pivoting back to small + fast.

**Architecture:** 2 layers × 96 hidden × vocab 32K (~6.4M
total params, ~200K non-embed). Same shape as the original
L3TC-200K but with the broader 32K tokenizer. Estimated speed:
~75-90 KB/s on M1 (vs Default 200K at 130 KB/s — 32K head
matvec doubles per-token cost, partially offset by ~8% fewer
tokens via better tokenizer).

**Hypothesis we never tested directly:** can a 2L × 96H model
generalize across all text domains given enough diverse data
and the right tokenizer? Pass 2 (2L × 96H × 16K, Pile corpus)
got enwik6 0.32 with 5 epochs. With 50 GB diverse corpus +
Pile 32K SPM + 20 epochs, can we get to ~0.25-0.30 across all
domains while keeping 75+ KB/s?

**Instance:** g5.xlarge on-demand (A10G 24 GB). 2L is small
enough that batch 32 should fit easily. ~$0.76/hr × ~24-36h =
~$20-30. No need for L40S.

**Training:** 20 epochs × 500K epoch_length (~20B tokens,
~1.5× corpus coverage), batch 32 + grad_accum 1. Bf16 mixed
precision. 24-36h total wall time. 48-hour auto-shutdown.

**LR schedule:** AdamW 1e-4 peak, cosine → 1e-6 with the fixed
scheduler (counts optimizer steps, not micro-batch steps).

**Corpus: ~52 GB diverse, deduplicated, all real data.**

| source | type | size | notes |
|---|---|---|---|
| Pile dedup (seed 2024) | prose, web, papers | 40 GB | HF streaming, no overlap with Exp D |
| nick007x/github-code-2025 (random) | natural GitHub mix: python, JS/TS, Java, C/C++, Go, Rust, HTML, CSS, markdown, etc. | 5 GB | open HF, 882 GB parent dataset |
| lumees structured | YAML, JSON, SQL, XML, shell, Dockerfile, TOML, Makefile, Groovy, Gradle | 5 GB | open HF, real GitHub configs |
| Zenodo Loghub full | BGL, Spark, HDFS system logs | 1 GB | real production logs |
| data.gov + GitHub CSV | diverse tabular data (200+ schemas) | 1 GB | government + public data |
| **Total** | | **~52 GB raw** | all deduplicated via MD5 |

**Tokenizer (NEW — balanced unigram):**

The existing Pile 32K SPM was trained on 1 GB of straight Pile
text (BPE) — never saw real logs, CSVs, configs, HTML, or
modern GitHub code. Retrained a new SPM on a 200 MB balanced
sample with these proportions:

| balance share | source |
|---|---|
| 50% (100 MB) | Pile (preserves prose vocabulary) |
| 25% (50 MB) | nick007x diverse code |
| 12% (24 MB) | lumees structured (config/SQL/YAML) |
| 6% (12 MB) | Zenodo logs |
| 7% (14 MB) | data.gov CSV |

**Algorithm choice: unigram (not BPE).** Unigram trains 5-6×
faster than BPE on the same corpus (parallel EM iterations
vs serial pair merges) and gives slightly better B/T. Tried
1 GB unigram first but suffix array construction blew past L3
cache (~5 GB working memory) and slowed dramatically. 200 MB
took 2.1 minutes with 32K vocab; 1 GB was projected at
60-90 minutes. 200 MB sample is sufficient — production SPMs
typically train on 100-500 MB.

**Tokenizer comparison (B/T, higher is better):**

| domain | Pile 32K BPE | Balanced 32K unigram | change |
|---|---:|---:|---:|
| Wikipedia | 3.50 | 3.52 | +0.4% (preserved) |
| **python_source** | 2.23 | **4.04** | **+81.4%** |
| **c_source** | 2.42 | **3.53** | **+45.8%** |
| fiction | 3.26 | 3.93 | +20.6% |
| csv_data | 2.19 | 2.22 | +1.4% |
| html | 2.39 | 2.40 | +0.2% |
| json_api | 2.57 | 2.56 | -0.4% |
| nginx_log | 2.45 | 2.42 | -0.8% |

The +81% on Python and +46% on C are game-changing — almost
2× fewer tokens needed for the same source code, which
directly translates to 2× faster compression on code domains
AND better ratios (model sees longer semantic units).

Wikipedia preserved (+0.4%) — no regression on prose.

JSON/logs/nginx essentially unchanged because they were
already efficient at 2.4-2.6 B/T on Pile BPE; hard to improve
further at 32K vocab.

**Tokenizer file:**
`tokenizer_balanced_32k/spm_balanced_unigram_32768.model`

**Sequence:**
1. ✅ Stream 5 GB from nick007x/github-code-2025 → added to
   corpus (567K files, 5.00 GB, all unique)
2. ✅ Train new SPM 32K unigram on balanced 200 MB sample
3. ✅ Validated B/T improvements on eval suite (above table)
4. ⏳ Re-tokenize the full ~52 GB corpus with the new SPM
   (~3 hours: decode existing Pile + tokenize raw structured)
5. Upload to S3, launch 2L × 32K training on g5.xlarge
6. Eval on full 9-domain suite

**Success criteria:** ≥75 KB/s on M1 AND ≤0.30 ratio on at
least 5 of 8 domains (vs. 6L Exp D's 49 KB/s and 0.33-0.59
ratios). If we hit that, this is the shipping tier.

**Backup plans if 2L isn't enough:**
- 4L × 32K — moderate capacity bump, still ~50-60 KB/s
- Multi-tier product: ship 200K (specialized prose) + 2L×32K
  (generalist) + 6L Exp D (max ratio) — let users pick
- The 6L Exp D model already exists and works as the "max
  ratio" tier if needed

**12L plan deferred.** The 64 GB corpus and uploaded 12L
training data are still valid if the 2L approach doesn't
generalize. We can always come back to 12L for a "max ratio"
tier later. The 2L experiment costs $20 vs $100-140 for 12L,
so it's the right cheap test first.

---

## Evaluation suite

Built via `scripts/build_eval_suite.py`. 9 domain-specific
test files in `bench/corpora/eval_suite/`:

| file | type | size | source |
|---|---|---|---|
| enwik6.txt | Wikipedia prose | 1 MB | LTCB |
| webster.txt | dictionary | 41 MB | Silesia |
| fiction.txt | fiction | 0.15 MB | Canterbury |
| json_api.txt | structured data | 5 MB | generated |
| nginx_log.txt | log file | 5 MB | generated |
| python_source.txt | code | 1.1 MB | CPython stdlib |
| csv_data.txt | tabular data | 5 MB | generated |
| xml_silesia.txt | markup | 5 MB | Silesia |
| c_source.txt | C code | 0.01 MB | Canterbury |

Default 200K baseline (enwik8 tokenizer):

| file | ratio | speed | notes |
|---|---:|---:|---|
| enwik6 | 0.1699 | 89 KB/s | good (in-distribution) |
| fiction | 0.4161 | 45 KB/s | mediocre |
| python_source | 0.4732 | 53 KB/s | mediocre |
| c_source | 0.5535 | 25 KB/s | bad |
| html | 1.0011 | — | **worse than raw** |

---

## Future tokenizer optimizations

1. **Unigram instead of BPE** at same vocab. 10-min experiment,
   same SPM library. May produce 5-10% fewer tokens.
2. **Vocab size sweep** (8K, 16K, 32K, 64K). Optimal vocab
   minimizes `tokens_per_byte × bits_per_token`.
3. **Augment tokenizer training data** with log/CSV samples
   (the Pile doesn't contain these).

---

## After Phase 11 — the shipping track

Phase 9 (fuzzing) should run next regardless of Phase 11's
outcome — mandatory for any public release.

**If experiment D succeeds** (broad-domain ratios ≤ 0.30 and
enwik6 stays reasonable):

| order | phase | what | effort |
|---|---|---|---|
| 1 | Phase 9 | Fuzzing + input caps | ~1 week |
| 2 | Phase 7 | Cross-platform determinism | ~2-3 weeks |
| 3 | Phase 6 | Release builds + CI | ~1-2 weeks |
| 4 | Phase 10 | Distribution + bindings | ~2-4 weeks |

**If experiment D fails:** try 8L × 96H, or accept multi-tier
product (enwik-specialized 200K fast tier + broader 6L/8L
generalist tier). Then same shipping track.

---

## S3 artifacts

| path | contents |
|---|---|
| `l3tc/corpora/train_12l_corpus_32k.txt` | **12L corpus**: 64 GB, 13.9B tokens (Pile 40 GB + structured 7 GB) |
| `l3tc/corpora/train_pile_32k.txt` | Exp D corpus: Pile 10 GB tokenized with Pile SPM 32K |
| `l3tc/corpora/tokenizer_pile_32k/` | Pile SPM 32K .model + .vocab |
| `l3tc/corpora/tokenizer_pile/` | Pile SPM 16K .model + .vocab |
| `l3tc/corpora/pile_raw_1gb.txt` | raw Pile text (1 GB) for tokenizer training |
| `l3tc/corpora/spm_enwik8.tar.gz` | enwik8 SPM tokenizer files |
| `l3tc/experiment_d_6l_32k/` | Experiment D checkpoints (6L, epochs 0-9) |
| `l3tc/phase11_pass2/` | Pass 2 checkpoints (2L, enwik8 SPM) |
| `l3tc/phase11_pass3/` | Pass 3 checkpoints (2L, Pile SPM 16K) |

## Infrastructure

- **AMI:** `ami-07a4fc98c4ed4e19e` (200 GB, $10/month)
- **Instances:** g5.xlarge (A10G 24 GB, ≤6L), g6e.xlarge (L40S 48 GB, 12L+)
- **S3:** `s3://dmatth1-bnn-checkpoints/l3tc/` (shared bnn bucket)
- **IAM:** `bnn-s3-access` instance profile
- **Key:** `swarm-ec2`
- **Scripts:** `scripts/train_l3tc_phase11.py`,
  `scripts/build_pile_corpus.py`, `scripts/build_corpus_v2.py`,
  `scripts/build_eval_suite.py`, `scripts/run_eval_suite.py`,
  `scripts/launch-spot-train.sh`, `scripts/launch-ondemand.sh`

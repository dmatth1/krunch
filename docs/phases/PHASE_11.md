# Phase 11 — Universal text + structured-data compression

**Goal:** train a model that compresses all text and structured
data types (prose, code, JSON, logs, CSV, XML) competitively
with classical compressors, while keeping the speed advantage
that motivated the original L3TC port.

**Status as of 2026-04-16:** pivoted to speed-first approach.
2L × 96H × 32K with retrained balanced tokenizer.
Re-tokenizing 52 GB diverse corpus locally (parallel × 10
workers, ~25 min). Then upload to S3 and train.

---

## Key findings (chronological)

### Tokenizer matters — but vocab size matters more

| tokenizer | enwik6 B/T | JSON B/T | logs B/T | code B/T |
|---|---:|---:|---:|---:|
| enwik8 SPM 16K | 3.53 | 1.70 | 1.91 | 1.95 |
| Pile SPM 16K | 3.25 | 2.35 | 2.09 | 2.13 |
| Pile SPM 32K | 3.50 | 2.57 | 2.44 | 2.22 |
| **Balanced unigram 32K** | **3.52** | 2.56 | 2.42 | **4.04 (py)** |

Doubling vocab from 16K → 32K let one tokenizer win on both
prose AND structured text (the 16K Pile broadened domains but
hurt prose). The **balanced unigram 32K** went further: trained
on a corpus-balanced sample (50% Pile / 25% code / 12% structured /
6% logs / 7% CSV) using unigram instead of BPE, it produces
+81% B/T on Python source, +46% on C source, +21% on fiction,
while preserving prose performance.

### Model architecture experiments

| experiment | architecture | tokenizer | enwik6 ratio | speed | epoch |
|---|---|---|---:|---:|---|
| Default L3TC-200K | 2L × 96H | enwik8 16K | **0.170** | **131 KB/s** | 20 |
| Pass 2 | 2L × 96H | enwik8 16K | 0.316 | ~130 KB/s | 5 of 10 |
| Pass 3 | 2L × 96H | Pile 16K | 0.343 | ~120 KB/s | 2 of 10 |
| **Exp D** | **6L × 96H** | **Pile BPE 32K** | **0.335** | **49 KB/s** | 9 of 10 |
| (next) 2L × 32K | 2L × 96H | balanced unigram 32K | tbd | tbd | — |

The 200K Default specialized in prose (0.17 enwik) but breaks
on structured text (>1.0 on HTML). Exp D generalized
(0.33-0.59 across domains) but at 2.7× speed cost. The pivot:
test if 2L can generalize given proper tokenizer + 5× more
diverse data than Pass 2/3.

### Pivot to speed-first (2026-04-15)

Larger models (6L, 12L) progressively erode the speed
advantage that originally motivated this work. For a CLI tool,
speed is a hard constraint. Reverting to 2L × 96H — the
original L3TC architecture — but with the broader 32K
balanced tokenizer and 5× more diverse training data than
prior 2L runs.

### Bugs found and fixed

1. **LR scheduler counted micro-batch steps, not optimizer
   steps.** Exp D's cosine never reached LR_MIN (1e-6),
   stalled at 7.3e-5 (73% of peak). Fixed —
   `total_steps = (steps_per_epoch * epochs) // grad_accum`.
2. **Eval `approx_bpb`** had hardcoded 3.5 bytes/token from
   enwik8. Now reports tokenizer-independent `bits_per_token`.

---

## Current run: 2L × 96H × vocab 32K (balanced unigram)

**Architecture:** 2 layers × 96 hidden × vocab 32768 (~6.4M
total params, ~200K non-embed). Same shape as the original
L3TC-200K but with the broader 32K balanced tokenizer.
Estimated speed: **~75-90 KB/s on M1**.

**Hypothesis:** can a 2L model generalize across all text
domains given (a) the right tokenizer (balanced unigram 32K,
+81% B/T on code) and (b) 5× more diverse data than Pass 2?

**Instance:** g5.xlarge on-demand (A10G 24 GB). Batch 32 +
grad_accum 1. ~$0.76/hr × ~24-36h = ~$20-30. No need for L40S.

**Training:** 20 epochs × 500K epoch_length (~20B tokens, 1.5×
corpus coverage). bf16, AdamW lr=1e-4 cosine → 1e-6.
48-hour auto-shutdown.

**Success criteria:** ≥75 KB/s on M1 AND ≤0.30 ratio on at
least 5 of 8 domains (vs Exp D's 49 KB/s and 0.33-0.59).

---

## Corpus: 52 GB diverse, deduplicated, real data

| source | size | content | dedup |
|---|---|---|---|
| Pile dedup (seed 2024) | 40 GB | prose, web, books, papers | HF-deduplicated |
| nick007x/github-code-2025 (random) | 5 GB | natural GitHub mix: python (274K), rust (63K), c/cpp (57K), java (32K), js/ts (35K), html (21K), markdown (41K), css, etc. | MD5 chunk dedup |
| lumees structured | 5 GB | YAML, JSON, SQL, XML, shell, Dockerfile, TOML, Makefile, Groovy, Gradle | MD5 chunk dedup |
| Zenodo Loghub | 1 GB | BGL, Spark, HDFS production logs (2.4M real lines) | n/a (different source) |
| data.gov + GitHub CSV | 1 GB | 200+ tabular schemas | n/a (different source) |
| **Total** | **~52 GB** | all real, 0.04% cross-file overlap | ✓ verified |

**S3 path (after upload):**
`s3://dmatth1-bnn-checkpoints/l3tc/corpora/train_2l_corpus_balanced_32k.txt`

---

## Tokenizer: balanced 32K unigram

**Training data:** 200 MB sample drawn from the full 52 GB
corpus with corpus-aware proportions:

| share | source |
|---|---|
| 50% (100 MB) | Pile (preserves prose vocabulary) |
| 25% (50 MB) | nick007x diverse code |
| 12% (24 MB) | lumees structured |
| 6% (12 MB) | Zenodo logs |
| 7% (14 MB) | data.gov CSV |

**Algorithm:** unigram (not BPE). Trains 5-6× faster than BPE
on M1 (parallel EM iterations vs serial pair merges) and gives
slightly better B/T. Also tested 1 GB sample — suffix array
construction blew past L3 cache (~5 GB working memory) and
slowed dramatically. 200 MB sample took 2.1 minutes; 1 GB was
projected at 60-90 minutes. 200 MB is sufficient — production
SPMs typically train on 100-500 MB.

**B/T comparison (higher is better):**

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

The +81% Python and +46% C are game-changing — almost 2×
fewer tokens for the same source code, which directly
translates to faster compression AND better ratios on code.
Wikipedia preserved (+0.4%). JSON/logs/nginx unchanged
(already efficient on Pile BPE at 32K, hard to improve at
this vocab size).

**Tokenizer file:**
`tokenizer_balanced_32k/spm_balanced_unigram_32768.model`

---

## Re-tokenization (in progress)

The existing tokenized corpus uses the OLD Pile BPE 32K. We're
re-tokenizing everything with the new balanced unigram via
`scripts/retokenize_corpus.py`:

- **Pile (40 GB):** decode existing tokens with old SPM →
  re-encode with new SPM. Validated: decode + re-encode
  preserves token counts exactly (96.36% character similarity,
  whitespace normalization only — no impact on training).
- **Other 4 sources (12 GB raw):** tokenize fresh from raw
  text files we already have locally.

**Parallelization:** 10 workers (one per M1 core), each
processing a byte range of the input file. Brought
estimated time from 2.5 hours (single-thread) → ~25 minutes.

After completion: concatenate per-worker output → single
training file (~65 GB tokens) → upload to S3 → launch
training.

---

## Eval suite

`bench/corpora/eval_suite/` (built via `scripts/build_eval_suite.py`):

| file | type | size | source |
|---|---|---|---|
| enwik6.txt | Wikipedia prose | 1 MB | LTCB |
| webster.txt | dictionary | 41 MB | Silesia |
| fiction.txt | fiction | 0.15 MB | Canterbury |
| json_api.txt | structured | 5 MB | generated |
| nginx_log.txt | logs | 5 MB | generated |
| python_source.txt | Python code | 1.1 MB | CPython stdlib |
| csv_data.txt | tabular | 5 MB | generated |
| xml_silesia.txt | markup | 5 MB | Silesia |
| c_source.txt | C code | 11 KB | Canterbury |

Run via `scripts/run_eval_suite.py`. See
`docs/EVALUATION_GUIDELINES.md` for measurement pitfalls
(background CPU load, single-thread profile vs parallel
compress, small-file penalty).

---

## Cloud training conventions

- **Pre-baked AMI** `ami-07a4fc98c4ed4e19e` (200 GB).
  Stale `at` jobs trigger on reboot — clear them on SSH.
- **Auto-shutdown safety net** via `at`. Match to expected
  training time + 50% margin.
- **S3 sync via cron every 3 min** (background subshells
  broke from quoting issues).
- **PYTHONUNBUFFERED=1** for visible training logs.
- **Instances:** g5.xlarge (A10G 24 GB) for ≤6L models;
  g6e.xlarge (L40S 48 GB) for 12L+ at vocab 32K.
- **torch.compile disabled** — L2Wrap can't be traced by dynamo.

---

## Decision tree after this run

If 2L × 32K hits ≥75 KB/s AND ≤0.30 ratio on ≥5 domains:
ship as the generalist tier. Move to Phase 9 (fuzzing).

If 2L can't generalize:
- Try 4L × 32K (intermediate, ~50-60 KB/s)
- Multi-tier product: ship 200K (specialized prose) + 6L Exp D
  (max ratio) — both already exist
- 12L × 32K still possible but deferred (cost $100-140 vs $20-30)

---

## After Phase 11 — shipping track

| order | phase | what | effort |
|---|---|---|---|
| 1 | Phase 9 | Fuzzing + safety hardening | ~1 week |
| 2 | Phase 7 | Cross-platform determinism | ~2-3 weeks |
| 3 | Phase 6 | Release builds + CI | ~1-2 weeks |
| 4 | Phase 10 | Distribution + bindings | ~2-4 weeks |

Phase 9 is independent of training and can run in parallel.
Already in progress: 3 fuzz harnesses, 2 bugs fixed (checkpoint
OOM + integer overflow), safety hardening applied to varint /
cum_freqs / NEON unsafe code. Long-duration fuzz runs (48h
each) pending.

---

## S3 artifacts (current, post-cleanup)

| path | size | purpose |
|---|---|---|
| `l3tc/corpora/train_2l_corpus_balanced_32k.txt` | tbd | **2L corpus** (after upload) |
| `l3tc/corpora/tokenizer_pile_32k/` | small | Pile SPM 32K BPE (Exp D) |
| `l3tc/corpora/tokenizer_pile/` | small | Pile SPM 16K BPE (Pass 3) |
| `l3tc/corpora/pile_raw_1gb.txt` | 1 GB | raw Pile reference |
| `l3tc/corpora/enwik9.xz` | 233 MB | reference data |
| `l3tc/corpora/spm_enwik8.tar.gz` | small | L3TC original tokenizer |
| `l3tc/corpora/train_enwik9_bpe_16384_0.999.txt` | 1.4 GB | L3TC reference training |
| `l3tc/experiment_d_6l_32k/` | ~1 GB | Exp D checkpoints (epochs 0-9) |
| `l3tc/phase11_pass2/` | small | Pass 2 checkpoints |
| `l3tc/phase11_pass3/` | small | Pass 3 checkpoints |
| `l3tc/enwik8_recipe_validation/` | small | recipe validation |

Cleanup performed 2026-04-16: deleted 113 GB of stale
tokenized corpora (synth_7gb, train_pile_dedup,
train_pile_new_spm, train_pile_32k, train_12l_corpus_32k).
All tokenizable from raw sources if needed.

## Infrastructure

- **S3:** `s3://dmatth1-bnn-checkpoints/l3tc/` (shared bnn bucket)
- **IAM:** `bnn-s3-access` instance profile
- **Key:** `swarm-ec2`
- **Scripts:**
  - `train_l3tc_phase11.py` — training
  - `build_pile_corpus.py` — Pile streaming + tokenization
  - `build_corpus_v2.py` — structured data builder
  - `retokenize_corpus.py` — parallel re-tokenization
  - `build_eval_suite.py` / `run_eval_suite.py` — eval
  - `launch-spot-train.sh` / `launch-ondemand.sh` — cloud

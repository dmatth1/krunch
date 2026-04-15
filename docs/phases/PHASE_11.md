# Phase 11 — Broader corpus + universal text compression

**Goal:** train a model that compresses all text and structured
data types (prose, code, JSON, logs, CSV, XML) at ratios
competitive with or better than classical compressors, while
maintaining usable speed.

**Status as of 2026-04-14:** Experiment D running (6L × 96H ×
vocab 32K on 10 GB Pile dedup). Earlier experiments established
that both more model capacity AND a better tokenizer are needed.

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
| **Exp D** | **6L × 96H (3.1M non-embed)** | **Pile 32K** | **0.3579** | **tbd** | **48 KB/s** | **2 of 10** |

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
- **Always set auto-shutdown safety net** (8-20 hours via `at`).
- **S3 sync via cron every 3 min** (not background subshells —
  those broke from quoting issues).
- **PYTHONUNBUFFERED=1** for visible training logs.
- **On-demand for short runs** (<6 hr), spot for long runs.
- **g5.xlarge** (A10G 24 GB, $0.76/hr) is the right instance.
  Batch 16 + grad_accum 2 for 2L models. Batch 8 + grad_accum 4
  for 6L models at vocab 32K.
- **torch.compile disabled** — L2Wrap can't be traced by dynamo.
- **AMI cost:** $10/month at 200 GB. Future re-bake at 50 GB =
  $2.50/month. Delete when Phase 11 complete.

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

**Epoch 6 compression ratios (measured on M1, clean machine):**

| file | type | exp D ratio | default 200K | speed |
|---|---|---:|---:|---:|
| enwik6 | Wikipedia | 0.3391 | 0.1699 | 49 KB/s |
| python_source | code | 0.3538 | 0.4732 | ~30 KB/s |
| fiction | fiction | 0.4052 | 0.4161 | ~20 KB/s |
| c_source | C code | 0.5997 | 0.5535 | ~8 KB/s |
| html | markup | 1.0011 | 1.0011 | — |

Ratios converging across domains. enwik6 improved from 0.358
(epoch 2) to 0.339 (epoch 6). Python source now 25% better
than the enwik8-specialized 200K model. Fiction comparable.
HTML still broken (OOD). Speed is 2.8× slower than 200K
(expected from 3× layers + 2× vocab, offset by fewer tokens).

**Speed note:** early measurements showed ~9 KB/s due to a
concurrent Pile download eating CPU. Clean-machine benchmarks
match the theoretical ~49 KB/s. See `docs/EVALUATION_GUIDELINES.md`.

Checkpoints syncing to
`s3://dmatth1-bnn-checkpoints/l3tc/experiment_d_6l_32k/`.

Loss deceleration per epoch: 0.17 → 0.08 → 0.05 → 0.03.
Projected epoch 10 eval CE: ~2.95. Projected enwik6 ratio:
~0.32. Still above the 0.20 target — 6L likely needs either
more training or more layers to reach it.

---

## Next run: 12L × 96H × vocab 32K on custom corpus

**Architecture:** 12 layers × 96 hidden, vocab 32768 (~12.5M
total params, ~6.2M non-embed). Estimated speed: ~35 KB/s
before optimization (INT8 head + top-K cum_freqs could push
to ~50 KB/s).

**Instance:** g6e.xlarge (L40S 48 GB VRAM) via spot fleet.
Needed because 12L at vocab 32K uses ~24 GB+ VRAM — too tight
for A10G 24 GB. Use `INSTANCE_TYPE=g6e.xlarge` with
`scripts/launch-spot-train.sh`.

**Training:** 20-30 epochs × 500K epoch_length, batch 8 +
grad_accum 4 (or larger batch if VRAM allows on L40S). With
memmap token loading, any corpus size fits on any instance.
Estimated cost: ~$80-100 on spot over 2-3 days.

**Corpus: custom compressor-oriented mix (~50 GB).** The Pile
is a good LLM pre-training corpus but has gaps for compression:
no logs, no CSV, no YAML/config, stale code (2020). Building
a custom corpus from REAL data (not synthetic):

| source | type | amount | notes |
|---|---|---|---|
| RedPajama/Pile (base) | prose, code, web | ~40 GB | HuggingFace streaming |
| The Stack v2 | real YAML, SQL, JSON, XML, Dockerfiles, shell | ~5 GB | real GitHub files (needs HF_TOKEN, gated) |
| Loghub (GitHub) | real system logs (Apache, HDFS, Linux, etc.) | ~1 GB | direct download from logpai/loghub |
| Public CSV datasets | real tabular data (UCI, Kaggle public) | ~1 GB | direct download |

~85% general text + ~15% real structured data from production
sources. No synthetic generation — all real files from real
repos and systems. Built via `scripts/build_real_corpus.py`.

**Status:** corpus assembly in progress locally.

Completed:
- ✅ 16 real system log types from loghub (Apache, BGL, HDFS,
  Linux, OpenSSH, OpenStack, Spark, etc.) — 4.5 MB
- ✅ 17 real CSV datasets (GDP, airports, COVID, financial,
  ML datasets from UCI + GitHub datasets org) — 29.5 MB

Running:
- 🚧 Pile GitHub subset filter: streaming the Pile dedup,
  extracting real YAML/JSON/SQL/XML/Dockerfile/Makefile/shell
  files by content pattern matching. Target 3 GB. Currently
  at ~76 MB and growing. ETA ~30-60 min.

After completion: upload structured data to S3, combine with
base Pile corpus (40 GB), tokenize with 32K SPM, train 12L
model on spot fleet.

**Tokenizer:** reuse the Pile SPM 32K (already trained and
proven). The 32K vocab covers structured text well (2.57 B/T
on JSON, 2.44 on logs). Can augment tokenizer training data
with log/CSV samples later if eval shows gaps.

**Decision after this run:** if 12L × 32K on the custom corpus
hits ≤ 0.20 on enwik6 AND ≤ 0.30 on structured domains, ship
it as the generalist tier at ~35-50 KB/s. If not, investigate
deeper models (16L+) or architectural changes (context
length, output layer).

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
| `l3tc/corpora/enwik9.xz` | compressed enwik9 (1 GB) |
| `l3tc/corpora/train_pile_dedup.txt` | Pile 10 GB tokenized with enwik8 SPM 16K |
| `l3tc/corpora/train_pile_new_spm.txt` | Pile 10 GB tokenized with Pile SPM 16K |
| `l3tc/corpora/train_pile_32k.txt` | Pile 10 GB tokenized with Pile SPM 32K |
| `l3tc/corpora/pile_raw_1gb.txt` | raw Pile text (1 GB) for tokenizer training |
| `l3tc/corpora/tokenizer_pile/` | Pile SPM 16K .model + .vocab |
| `l3tc/corpora/tokenizer_pile_32k/` | Pile SPM 32K .model + .vocab |
| `l3tc/corpora/spm_enwik8.tar.gz` | enwik8 SPM tokenizer files |
| `l3tc/enwik8_recipe_validation/` | enwik8 recipe validation checkpoints |
| `l3tc/phase11_pass2/` | Pass 2 checkpoints (2L, enwik8 SPM) |
| `l3tc/phase11_pass3/` | Pass 3 checkpoints (2L, Pile SPM 16K) |
| `l3tc/experiment_d_6l_32k/` | Experiment D checkpoints (6L, Pile SPM 32K) |

## Infrastructure

- **AMI:** `ami-07a4fc98c4ed4e19e` (200 GB, $10/month)
- **Instance:** g5.xlarge on-demand ($0.76/hr)
- **S3:** `s3://dmatth1-bnn-checkpoints/l3tc/` (shared bnn bucket)
- **IAM:** `bnn-s3-access` instance profile
- **Key:** `swarm-ec2`
- **Scripts:** `scripts/train_l3tc_phase11.py`,
  `scripts/build_pile_corpus.py`, `scripts/train_tokenizer_pile.py`,
  `scripts/build_eval_suite.py`, `scripts/run_eval_suite.py`,
  `scripts/launch-ondemand.sh`, `scripts/launch-spot-fleet.sh`

# Spike 5 — log

Running trail of Spike 5 measurements + decisions. Plan lives in
`SPIKE_5_PLAN.md`.

## Status — 2026-04-23

- **Training complete** on g6.xlarge (L40S), 10 epochs, WildChat-English
  200 MB corpus, L3TC-12M config (4L × 384 hidden × 1024 intermediate,
  HiRA rank=4 → ~55M actual parameters).
- **held_out_ratio (entropy bound, CPU path)**: **0.2208** vs zstd
  baseline **0.1669** → neural ~32% **worse** than zstd on held-out.
- **Pass gate** (≤0.1145, 25% better than zstd): **missed by wide
  margin**. 12M config not enough to beat zstd on chat at 200 MB.
- **Dispatcher ratio on GPU**: **not measured** — deferred to user
  return (Fargate CPU on a 55M-param RWKV would be glacial, and we
  only provisioned the g6 for training, not for a full compression
  sweep).
- Artifacts uploaded to `s3://archive-dev-archive/spike5/wildchat-en/models/`:
  v1.bin (77 MB), v1.pth (666 MB), v1.tokenizer.model, v1.zstd_dict,
  v1.metadata.json.

## Setup

- Corpus: 200 MB WildChat-4.8M English-only subset (same corpus as
  Spike 4 exp 2, so results are directly comparable).
- Hardware: g6.xlarge on-demand (NVIDIA L40S, 48 GB HBM).
- Model: L3TC-12M target config from the AAAI 2025 paper —
  `num_layers=4`, `hidden_size=384`, `intermediate_size=1024`,
  `rwkv_rank=4`, `ctx_len=2048`, vocab=16384 (SPM unigram, shared
  across all spikes).
- Training: 10 epochs × 50000 steps × batch=16 (had to drop from 32
  after OOM — HiRA with rank=4 inflated the param count from the
  paper's 12M "inference-time" count to 55M active params during
  training).
- Elapsed: ~4.8h on-demand compute.

## Training loss curve (selected epochs)

| epoch | avg_loss (nats) | bits/token |
|---|---|---|
| 0 | 4.87 | 7.02 |
| 3 | 2.92 | 4.21 |
| 5 | 2.52 | 3.64 |
| 7 | 2.34 | 3.37 |
| 9 | 2.199 | 3.17 |

Eval epoch 9: `avg_ce_nats=3.5114`, `bits_per_token=5.066`.

## Train/val gap

- Train loss: **2.199 nats/token**
- Val loss: **3.511 nats/token**
- Gap: **+1.31 nats/token (+60%)** — significant overfitting.
- At 55M params on 200 MB corpus (~61M tokens), memorization was
  always a risk; training deeper/wider than the paper's 12M config
  ate the regularization headroom the paper counted on.

## Interpretation

Three compounding factors:

1. **Capacity mismatch.** Paper configures 12M params for Wikipedia
   article distillation at 1–3 GB corpora. We applied it to 200 MB
   WildChat-English — the per-parameter token budget is ~1/10th of
   the paper's regime, so overfitting is the default.
2. **Chat text is irreducibly noisier than enwik8.** Spike 4 already
   showed English-only WildChat chews through a 200K model without
   producing dispatcher-wins. The 12M bump halved train-loss but the
   held-out entropy didn't follow — val plateaued at 3.5 nats, which
   is *worse* bits/token than enwik8 at the same param count in the
   paper (3.0-ish).
3. **Held-out numbers validate the bug fix.** Pre-fix, the measure
   script was loading a random-init model (checkpoint key mismatch)
   and emitting ~0.54 regardless of actual training. Now we get
   0.2208 — which is both (a) strictly better than random-init
   entropy (0.51) and (b) informative enough to reject the model.
   The number moved in response to training, so the fix is working.

## Decision

**Do not ship L3TC-12M on chat data.** The path forward is one of:

- **(a) Larger corpus, same model.** Retrain 12M on WildChat-English
  1 GB+ to restore the paper's token-per-param ratio. Cheapest test.
- **(b) Scale to L3TC-3.2M.** Paper's intermediate rung. Smaller
  train/val gap per param, still GPU-parallel at decode.
- **(c) Abandon general chat, go vertical.** The customer-profile
  pivot (Harvey / Nabla / Abridge) assumes the corpus is per-tenant
  (one law firm's docket, one clinician's notes). That corpus is
  narrower and likely more compressible — we've been benchmarking
  on the hardest possible general corpus.

Recommend (c) then (a): get a representative single-tenant corpus
from a design-partner-style source (e.g., CourtListener opinions,
SEC EDGAR filings filtered to one filer, MIMIC-III notes) and
re-run Spike 5 config before any further model-scaling work.

## Artifacts ready for GPU measurement phase

Deferred to user's return. When resumed:

1. Spin up g5.xlarge (on-demand, 1 hr max) with ONNX Runtime CUDA.
2. Export `v1.pth` → ONNX, validate parity vs PyTorch on a 1 MB
   holdout slice (bits/token within 0.01 of the measure script).
3. Run full dispatcher sweep over 200 MB corpus measuring:
   - neural-selection rate per chunk
   - real dispatcher ratio (vs the entropy-bound 0.2208)
   - end-to-end KB/s on the GPU path (pass gate ≥300 KB/s).
4. Only then decide whether the codec is worth shipping to Tier 1.

## Bug trail (Spike 5–specific)

- **OOM at batch=32.** Paper's 12M config inflates to 55M with HiRA
  rank=4. Fix: `BATCH_SIZE=16` env override.
- **S3 corpus race.** First submission hit an empty `raw/` because
  the pre-compression cleanup from Spike 4 had already deleted it.
  Fix: re-uploaded local `data/wildchat-en/` copy to
  `s3://archive-dev-archive/spike5/wildchat-en/raw/`.
- **held_out_ratio bug** (inherited from Spike 4). Fixed in
  `scripts/measure_held_out_ratio.py` before this run — checkpoint
  load now uses key `"model"` first and asserts ≥10% of params load.
  First spike where the number is real.

## What not to read into this

- The 0.2208 number is the **entropy bound** on held-out val. The
  *dispatcher* ratio will be strictly lower because chunks where
  neural loses are routed to zstd/bzip3. Spike 4 showed the
  dispatcher close-but-not-beat zstd even when neural was much
  worse than bzip3 per chunk. Real dispatcher-ratio measurement
  still pending GPU sweep.
- The train/val gap is a *corpus-size* problem first, *model-size*
  problem second. A 1 GB WildChat-English run may close most of it
  without any architecture change.

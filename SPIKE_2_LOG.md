# Spike 2 — running log

Goal from [`SPIKE_2_PLAN.md`](SPIKE_2_PLAN.md): beat
`zstd --long=27 --ultra -22` on HDFS within a production envelope
(training ≤ 2 hr / ≤ $10 per run, decompress ≥ 1 MB/s single-stream
on L4 GPU).

## Baseline from Spike 1 (HDFS_v1)

| metric | value |
|---|---|
| corpus | HDFS_v1 NDJSON, 1.39 GB raw / 278 MB val |
| zstd --long=27 --ultra -22 | **0.0466** (0.373 bits/byte) |
| pass gate | **held_out_ratio < 0.98 × zstd = 0.0457** |
| baseline model v1 (200 K, 16 K vocab, 2 K ctx) | 0.1405 (1.124 bits/byte) — 3× worse |

## Experiment log

_(Each row appended as experiments run. `held_out_ratio` is the
number emitted by `measure_held_out_ratio.py`; if that sentinel-1.0,
the in-process eval number from `train_l3tc_phase11.py` is used and
noted.)_

| version | phase | vocab | num_layers | ctx | sample_mb | epochs | batch | held_out | zstd | gate | walltime | cost | pass? | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 | (Spike 1 baseline) | 16 384 | 2 | 2048 | 200 | 10 | 32 | 0.1405* | 0.0466 | 0.0457 | 67 min | $0.89 | FAIL | *from in-process eval; sentinel 1.0 in metadata due to rc=$? bug (fixed) |
| hdfs-a1/v1 | A1 (32 K vocab) | 32 768 | 2 | 2048 | 1058 (full) | 10 | 8 | **~0.173-0.198** (sentinel 1.0 in metadata; rc=137 OOM kill during measure script) | 0.0466 | 0.0457 | ~135 min | ~$1.80 | **FAIL — also worse than baseline** | diagnosed: `BATCH_SIZE=8` override (needed for OOM avoidance on 32K vocab softmax) inflated optimizer-step count 4× (15.6K → 62.5K) with the same 1e-4→1e-6 cosine schedule, so LR reached near-zero by epoch 4 → 60% of training was at near-zero LR. Also `measure_held_out_ratio.py` OOM'd — needs fp32→bf16 or per-segment cleanup. Next: A1b rerun with `EPOCHS=3` to put ~15K steps back under the schedule. |
| hdfs-a1b/v1 | A1b (32K vocab, correct steps) | 32 768 | 2 | 2048 | 1058 | 3 | 8 | **~0.164** (bits/token 4.3502 @ SPM bytes/token 3.31) | 0.0466 | 0.0457 | ~47 min | ~$0.63 | **FAIL** | schedule fix confirmed: eval bits/token at epoch 0 = 4.35, epoch 1 = 4.37, epoch 2 = 4.44 — LR working normally, but 32K vocab produces *shorter* val tokens than 16K (2.89 vs 3.02 bytes/token), so bits/byte went UP not down. HDFS's templated structure means 16K vocab already captures most useful merges; extra slots got wasted on training-only patterns and val falls back to short sub-words. **Vocabulary scaling is a dead end for HDFS; skip A2.** |

### Phase A verdict

Tokenizer scaling does not help on HDFS. Val bytes/token regresses
with larger vocab because the SPM allocates slots to training-only
rare patterns that don't appear in val, forcing val to fragment back
into short byte-fallback tokens. Net bits/byte goes up.

Moving directly to **Phase B** (model capacity scaling with 16K vocab).

## Infra fixes folded in for Spike 2

- Compression Dockerfile: `PYTHONUNBUFFERED=1` so the worker's
  progress prints land in CloudWatch live instead of after process
  exit.
- Compression Fargate scale-in: `minScalingCapacity: 1`, scale-in
  step changed to no-op at `ApproximateNumberOfMessagesVisible=0`.
  Prevents the scale-in-kills-active-task bug that hid Spike 1's
  verification for 15 min. Proper fix (MathExpression summing
  visible + in-flight) tracked in PRODUCTION_TODO.
- Training entrypoint now reads `VOCAB_SIZE`, `NUM_LAYERS`,
  `CONTEXT_LEN`, `EPOCHS`, `EPOCH_LENGTH`, `BATCH_SIZE`, `SAMPLE_MB`
  from env. Each experiment is a container-override change, no
  image rebuild required unless a Dockerfile change lands.
- Fargate worker task now uses 2 vCPU / 4 GB RAM (up from 1/2) so
  zstd-22 finishes in ~30% less walltime and leaves headroom for
  the l3tc-rust runtime when we wire it in.
- Metadata JSON now records every tunable hyperparameter so
  SPIKE_2_LOG can be regenerated from S3 if this file is lost.

## Decision gates

- If any experiment lands `held_out_ratio < 0.98 × zstd` → **WIN**,
  stop experiments, ship that config.
- If Phase A clears the gate → skip B/C/D.
- If Phase B (2 M model) fails but gets within 20% of gate → Phase C
  (hybrid template detection).
- If B1 is worse than expected (> 0.08) → jump straight to Phase C;
  capacity scaling isn't buying enough.
- Hard fail at Phase C: come back to the user, discuss whether to
  pivot off HDFS as the hard-mode benchmark.

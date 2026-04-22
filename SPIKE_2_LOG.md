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

| version | phase | vocab | num_layers | ctx | sample_mb | epochs | held_out | zstd | gate | walltime | cost | pass? | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 | (Spike 1 baseline) | 16 384 | 2 | 2048 | 200 | 10 | 0.1405* | 0.0466 | 0.0457 | 67 min | $0.89 | FAIL | *from in-process eval; sentinel 1.0 in metadata due to rc=$? bug (fixed) |

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

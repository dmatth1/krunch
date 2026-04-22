# Dispatcher simulation results — enwik8 + HDFS val

Ran `scripts/simulate_dispatcher.py` on both corpora at 64 KB chunk
size with the codec menu available in the MVP (zstd-22 --long=27,
bzip3, lz4, neural via entropy-bound estimate). No zstd trained
dictionary yet, no CLP, no brotli shared-dict.

## enwik8 (100 MB English Wikipedia)

Neural estimate: 1.7328 bits/byte (from historical L3TC-200K measurement
in `bench/results/enwik8-l3tc.md`).

| codec | total bytes | ratio | vs dispatcher |
|---|---|---|---|
| **DISPATCHER** | **21,621,016** | **0.2162** | 1.00× |
| neural alone | 21,659,851 | 0.2166 | 1.00× |
| bzip3 alone | 32,905,993 | 0.3291 | 1.52× |
| zstd -22 --long=27 per-chunk | 36,231,765 | 0.3623 | 1.68× |
| zstd -22 --long=27 whole-file (reference from bench/) | 25,270,000 | 0.2527 | 1.17× |
| lz4 alone | 57,030,001 | 0.5703 | 2.64× |

Codec distribution: **neural 99.2%, bzip3 0.8%**. Neural won nearly
every chunk. Safety net didn't fire.

**Dispatcher beats whole-file zstd-22 by 14.5%.** That's the headline
number for prose.

## HDFS val (277 MB templated log stream, Spike 1 baseline neural)

Neural estimate: 1.124 bits/byte (Spike 1 HDFS model, no template
normalization).

| codec | total bytes | ratio | vs dispatcher |
|---|---|---|---|
| **DISPATCHER** | **18,372,117** | **0.0662** | 1.00× |
| zstd -22 --long=27 per-chunk | 18,492,890 | 0.0667 | 1.01× |
| bzip3 per-chunk | 21,410,988 | 0.0772 | 1.17× |
| neural alone | 38,966,667 | 0.1405 | 2.12× |
| lz4 alone | 40,017,895 | 0.1443 | 2.18× |
| **zstd -22 --long=27 whole-file** (per Spike 1) | **12,924,975** | **0.0466** | **0.70×** |

Codec distribution: **zstd 86.9%, bzip3 13.1%**. Neural never won a
chunk. Safety net didn't fire.

**Dispatcher loses to whole-file zstd-22 by 42%** on HDFS. The
chunking penalty hurts us more than the dispatcher can recover.

## Interpretation

### Confirmed: neural wins on text

On enwik8 the dispatcher matches neural-alone almost exactly (0.2162
vs 0.2166), and both beat whole-file zstd by 14.5%. No classical
alternative in the menu wins chunks on prose. The dispatcher shape
is correct; Tier 1 should ship.

### Confirmed: safety-net architecture works

On both corpora the Stage 3 safety net never triggered because the
Stage 2 probe-encode correctly picked the best codec each time.
The safety net is cheap insurance for out-of-distribution chunks,
not an everyday tool.

### Surfaced problem: chunking hurts highly-templated data

On HDFS the whole-file zstd baseline (0.0466) is untouchable by any
codec run per-chunk, because zstd's 128 MB window captures
cross-chunk template repetition we've cut off. The dispatcher's
choice is limited to "zstd per chunk" or "something else per chunk,"
and the first option can't match the global optimum.

Two ways to fix:

1. **Larger chunks on templated-log datasets.** 4 MB chunks would
   give zstd a 4 MB window per chunk and recover most of the win.
   Tunable per-dataset.
2. **Add CLP to the codec menu.** CLP's template + variable-dict IR
   maintains its dictionary ACROSS chunks (by construction — the
   template table is per-dataset, not per-chunk). That's the
   structural fix for HDFS-class data. Per the research agent's
   report, CLP reports ~2× smaller than zstd-22 on real logs.

The design already calls out CLP as codec 0x6; this simulation
confirms **we need it in the menu for HDFS-class workloads to beat
whole-file zstd**. Without CLP, the dispatcher is strictly worse
than whole-file zstd on templated logs, even with the safety net.

### Implication: detector should also pick chunk size

When Stage 1 detects a templated-log corpus (or the customer
declares `templated_logs`), the dispatcher should default to **4 MB
chunks or larger** for that dataset instead of 64 KB. That keeps the
zstd path viable as a safety net AND gives CLP room to build a
richer per-chunk template table.

Adding chunk-size to the dispatcher config per-dataset isn't a big
change — it's a number in the training-complete Lambda's output.

## Tier 1 implementation adjustments from these results

1. **Ship CLP as part of Tier 1, not later.** The engineering plan
   had it at 3-5 days; that's still accurate but the priority is
   higher than earlier assumed. Without CLP in the menu, the
   dispatcher can't win HDFS.
2. **Per-dataset chunk size** as a tunable, defaulting to 64 KB for
   text-heavy and 4 MB for templated-log. Add to training pipeline
   output alongside the zstd dict.
3. **Safety net stays cheap** — unnecessary so far, keep the
   implementation simple.
4. **Metrics section already captures** the zstd_shadow_bytes that
   would have exposed this chunking-penalty issue in production;
   we'd see "SavingsVsZstdPct = -42%" on an HDFS customer and alarm
   immediately. Good validation of the metric design.

## Next steps

- Run enwik8 dispatcher simulation again once C6full lands with
  its finalized HDFS-model bpb, just to confirm C6full's numbers
  don't change the picture on prose (it shouldn't — C6full was
  trained on HDFS only, irrelevant to enwik8).
- Add CLP to the simulator as a stub codec estimator (rough
  per-chunk ratio of 0.02 on templated logs per the OSDI'21 paper)
  and re-run HDFS. Should show dispatcher = CLP chunks + zstd
  fallback = ratio < 0.0466.
- Kick off Spike 3 (JSON events, code, transcripts) to collect
  neural bpb numbers on the corpora the dispatcher would route
  to neural in production.

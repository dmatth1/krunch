# Spike 4 — log

Running trail of Spike 4 measurements + decisions. Plan lives in
`SPIKE_4_PLAN.md`.

## Status — 2026-04-22

- **Experiment 1 (WildChat multilingual, 200 MB)**: training complete,
  compression in flight on 16 vCPU Fargate Spot. held_out_ratio came
  in at **0.568** (neural worse than no compression on multilingual
  chat); dispatcher ratio pending.
- **Experiment 2 (WildChat English-only, 200 MB)**: queued. Waiting
  on image rebuild with ninja-leak fix + jobdef rev 19.
- **Experiment 3 (OASST2, diagnostic)**: deferred until experiments
  1 + 2 confirm the English-vs-multilingual hypothesis.

## Experiment 1 — WildChat multilingual

### Setup

- Corpus: 200 MB WildChat-4.8M subset, unfiltered (all 25+ languages)
- 35,218 conversations, 206,660 turns
- Schema: NDJSON with `{conversation_id, turn, role, content, timestamp, model}`
- Hardware: g6.xlarge on-demand (NVIDIA L40S)
- Model: L3TC-200K (2L × 96h), SPM unigram 16K vocab
- Training: 10 epochs × 1562 steps, batch=32, LR=1e-4, ctx=2048

### Training loss curve

| epoch | avg_loss | delta |
|---|---|---|
| 0 | 6.9226 | — |
| 1 | 5.3223 | −1.60 |
| 2 | 4.9842 | −0.34 |
| 3 | 4.7717 | −0.21 |
| 4 | 4.6285 | −0.14 |
| 5 | 4.5337 | −0.09 |
| 6 | 4.4733 | −0.06 |
| 7 | ~4.43 | — |
| 8 | ~4.42 | — |
| 9 | ~4.41 | — |

(Final epoch values estimated; post-training still dropped by ~0.03.)

### Metadata emitted

```json
{
  "codec": "hybrid",
  "held_out_ratio": 0.567952,
  "zstd_baseline_ratio": 0.185466,
  "bytes_per_token": 3.2178,
  "has_bin": true,
  "has_zstd_dict": true
}
```

**held_out_ratio = 0.568 is catastrophically bad** — 3× worse than
zstd alone. 200K params + 16K SPM vocab cannot cover 25+ languages
simultaneously. The `bytes_per_token = 3.22` (vs GH events' 5.03,
enwik8's ~3.5) confirms the tokenizer fragmented into many short
multilingual pieces, so the model has MORE tokens to predict per
byte but less context per token.

### Three training-entrypoint bugs surfaced + fixed

All three were latent — present since the Spike 3 training path
was added, but only triggered on this run because prior spikes
(Spikes 1–3 on English-heavy data) didn't fully exercise the
post-training section:

1. **`measure_held_out_ratio.py` call signature.** Vendor's
   `RWKV_TC_HIRA.forward()` routes `train=False` to a different
   4-arg signature (`input_token, output_token, output_types,
   criterion`) that expects a 3-D tensor. My first fix used
   `train=False` and crashed with `IndexError: Dimension out of
   range` at `flatten(1, 2)`.
   **Real fix**: call `model(inp, input_types, train=True)` under
   `model.eval()` — mirrors the trainer's call, avoids the
   forward_test path.

2. **`zstd --train` single-file error.** The entrypoint ran
   `zstd --train train.txt ...` which fails with `Error 14: nb of
   samples too low` — `zstd --train` wants many sample files.
   **Fix**: split the corpus into 64 KB chunks via `split -b 64K`,
   feed the directory to `--train`.

3. **Ninja build output leaking into shell capture.** The measure
   script's stdout includes torch cpp_extension.load output
   (`ninja: no work to do.`, `building RWKV_TC_HIRA: ...`) BEFORE
   the final ratio number. `held_out_ratio=$(python ...)` captured
   all of it, producing invalid JSON in metadata.json that crashed
   the training-complete Lambda.
   **Fix**: capture stdout to file, `grep '^[0-9]+\.[0-9]+$' | tail -1`
   to extract only the bare-float line.

Bug 3 specifically blocked the auto-compression handoff. Recovery
was manual: patched metadata.json in S3 (reconstructed from known
values), sent a compression SQS message directly.

### Compression run result

First production use of the **16 vCPU / 32 GB Fargate Spot Graviton2**
setup (compression-stack changes from 2026-04-22, `-C target-cpu=neoverse-n1`).

| metric | value |
|---|---|
| bytes in | 209,716,731 (200 MB) |
| bytes out | 41,646,633 (39.7 MB) |
| **dispatcher ratio** | **0.1986** |
| per-chunk zstd-22 shadow | 0.2256 |
| savings vs zstd shadow | +12.0% |
| **whole-file zstd-22** | **0.1723** |
| **dispatcher vs whole-file zstd** | **−15.2% (worse)** |
| chunks total | 201 |
| codec: bzip3 | 200 / 201 |
| codec: neural | 1 / 201 (433 bytes total — essentially zero) |
| safety-net substitutions | 0 |
| throughput | 0.17 MB/s |
| wall time | ~25 min |

Pass gate **missed by −15.2%** vs required. As predicted.

### Positive signals even in the miss

- **First production exercise of the optimized compression stack**:
  16 vCPU Fargate Spot on Graviton2 with the `neoverse-n1` binary.
  No SIGILL on Spot placement — the compat fix works.
- **Throughput: 0.17 MB/s = 5.5× the pre-optimization baseline**
  (0.0317 MB/s on 4 vCPU neoverse-v1). Still below the M1 170 KB/s
  ceiling (Graviton per-core is ~2.7× slower), but the
  compression-speed optimization work paid off.
- **Service pipeline robustness**: even with the three training
  bugs blowing up post-training, the compression side (manual SQS
  message → 16 vCPU Spot → task-protection → hybrid-compress → EMF
  + dashboard + S3 upload) worked cleanly. Infrastructure is at
  "handles degraded inputs gracefully" maturity.

## Experiment 2 — WildChat English-only (complete)

### Setup (actual)

- Corpus: 200 MB WildChat-4.8M filtered to `language == "English"`
  at stream time. ~35K conversations, ~207K turns.
- Hardware, model config, training recipe: identical to exp 1
  (200K RWKV, 2L × 96h, 10 epochs, batch 32)
- Training job: `426b3d6d` on jobdef rev 19 (image carrying the
  `input_types + train=True` and `zstd --train` multi-sample fixes
  from this session; held_out_ratio bug still latent, not yet
  fixed at time of run).
- Wall time: ~1h10m training + ~25 min compression

### Training loss curve (English vs multilingual)

| epoch | English | Multilingual (exp 1) | delta |
|---|---|---|---|
| 0 | 6.7444 | 6.9226 | −0.18 |
| 1 | 5.1864 | 5.3223 | −0.14 |
| 2 | 4.8519 | 4.9842 | −0.13 |
| 3 | 4.6493 | 4.7717 | −0.12 |
| 4 | 4.5165 | 4.6285 | −0.11 |
| 5 | 4.4290 | 4.5337 | −0.10 |
| 6 | 4.3731 | 4.4733 | −0.10 |
| 7–9 | (not captured) | (similar plateau) | |

Consistent ~0.1 nat improvement per epoch over multilingual.
Final loss (epoch 9) projected ~4.30 vs multilingual's ~4.41.

### Metadata emitted

```json
{
  "codec": "hybrid",
  "held_out_ratio": 0.54439,
  "zstd_baseline_ratio": 0.166851,
  "bytes_per_token": 3.4223,
  "has_bin": true,
  "has_zstd_dict": true
}
```

`held_out_ratio = 0.544` is WRONG — the measure script was still
loading the checkpoint under the wrong key (`"model_state_dict"` vs
actual `"model"`). This produced near-uniform next-token
predictions → entropy ≈ ln(16384) nats ≈ 9.7 nats/token → ratio
≈ 0.51, matching the 0.544 bogus output.

**Bug was diagnosed and fixed in commit `08d8802` after this run.**
All three prior training-job runs produced meaningless
`held_out_ratio` values. Real compression ratio (below) was always
the source of truth.

### Compression run result

| metric | value |
|---|---|
| bytes in | 209,716,337 (200 MB) |
| bytes out | 37,357,111 (35.6 MB) |
| **dispatcher ratio** | **0.1781** |
| per-chunk zstd-22 shadow | 0.2022 |
| savings vs zstd shadow | +11.9% |
| **whole-file zstd-22** | **0.1526** |
| **dispatcher vs whole-file zstd** | **−16.7% (worse)** |
| chunks total | 201 |
| codec: bzip3 | 200 / 201 |
| codec: neural | 1 / 201 (271 bytes — effectively zero) |
| safety-net substitutions | 0 |
| throughput | 0.13 MB/s |
| wall time | ~25 min |

### Pass gate outcome

| gate | target | actual | result |
|---|---|---|---|
| Required (≥15% better than zstd-22) | ≤ 0.1297 | 0.1781 | **MISS** (−16.7%) |
| Strong (≥25%) | ≤ 0.1145 | 0.1781 | miss |
| Stretch (≥35%) | ≤ 0.0992 | 0.1781 | miss |

### Interpretation

Neural won only 1 of 201 chunks (271 bytes). Bzip3 took
everything else. The 200K model produces an entropy bound
(inferred from training loss ~4.30 nats) around 0.22 at
bytes_per_token 3.42 — meaningfully worse than bzip3's ~0.18 on
this data.

**This is a capacity problem, not a training problem.** Multiple
angles confirm:
- Switching from 10 → 20 epochs would buy ~2% ratio (geometric
  loss decay), moving −16.7% to ~−14%. Still misses.
- English-only cleaned the multilingual tax but zstd also got
  more effective on English content (baseline 0.1723 → 0.1526).
  Relative gap widened slightly.
- Architecture-level fix (bigger model): Spike 5 (L3TC-12M)
  changes model capacity by ~60× and tests directly.

Product-level finding: **the 200K model is not enough for AI
chat archives**. Either we ship the 12M model (Spike 5) or
restrict the product's neural codec to prose-heavier content
types (documents, long-form legal correspondence, medical notes
that read more like articles than chat turns).

## Cross-experiment data table (updated)

| corpus | size | zstd-22 | dispatcher | savings | codec distribution |
|---|---|---|---|---|---|
| enwik8 prose | 5 MB | 0.2878 | 0.1876 | +34.8% | neural 5/5 |
| GH events JSON | 5 MB | 0.1436 | 0.1475 | −2.7% | bzip3 5/5 |
| **WildChat multilingual** | **200 MB** | **0.1723** | **0.1986** | **−15.3%** | **bzip3 200/201, neural 1/201** |
| **WildChat English** | **200 MB** | **0.1526** | **0.1781** | **−16.7%** | **bzip3 200/201, neural 1/201** |

Clear pattern: neural wins decisively on enwik8 prose, loses on
both JSON and chat at this model size. Spike 5 (12M) targets
the chat case directly.

### Setup

- Same WildChat-4.8M source, filtered to `language == "English"` at
  stream time → 200 MB subset (roughly half the conversations we'd
  otherwise get, since ~40-50% of WildChat-4.8M is non-English)
- S3 location: `spike4/wildchat-en/raw/wildchat_en_200mb.ndjson`
- zstd-22 baseline on the English subset: **0.1526** (tighter than
  multilingual's 0.1723 — English alone has more zstd-friendly
  repetition of model-response templates)
- Hardware, model, training config: identical to experiment 1
  (control experiment isolating the language variable)

### Pass gates

| gate | target | interpretation |
|---|---|---|
| Required | dispatcher ratio ≤ 0.1297 (−15%) | product thesis floor |
| Strong | ≤ 0.1145 (−25%) | matches LMCompress dialogue literature |
| Stretch | ≤ 0.0992 (−35%) | punches above 200K weight class |

### Pending

1. CodeBuild `ab2dede7` finishes with the ninja-leak fix (in flight)
2. Register jobdef rev 19 pointing at the new training image
3. Submit training via SQS — `{cid: spike4, dsid: wildchat-en, trigger: initial}`
4. Training completes → auto-enqueues compression → measure

### Expected outcome

Prediction based on LMCompress literature + enwik8 baseline + the
removal of the multilingual tax:
- held_out_ratio: ~0.12–0.15 (big improvement over multilingual's 0.57)
- dispatcher ratio: ~0.11–0.13 (at or slightly above the strong gate)
- codec distribution: neural wins majority, bzip3 wins the framing-heavy chunks

If this pans out, **we have the first empirical evidence the narrowed
product thesis holds**.

## Experiment 3 — OASST2 (deferred, diagnostic)

Queued behind experiment 2. If experiment 2 clears the strong gate,
OASST2 becomes a "curated vs production" sanity check. If experiment
2 barely clears required or misses, OASST2 becomes a debugging tool
(is it the corpus or the model?).

## Cross-spike data table so far

| corpus | size | zstd-22 | dispatcher | savings | codec distribution | source |
|---|---|---|---|---|---|---|
| enwik8 prose | 5 MB | 0.2878 | 0.1876 | +34.8% | neural 5/5 | RUST_DISPATCHER_BENCH |
| enwik8 prose | 100 MB (pre-dispatcher, neural alone) | 0.2527 | 0.2166 | +14.3% | neural only | bench/results/enwik8-l3tc |
| HDFS (templated logs) | 277 MB | 0.0466 | 0.0662 | **−42%** | zstd 87%, bzip3 13% | DISPATCHER_SIM_RESULTS (sim) |
| GH events JSON | 5 MB | 0.1436 | 0.1475 | **−2.7%** | bzip3 5/5 | SPIKE_3_LOG |
| WildChat multilingual | 5 MB sample | 0.17–0.18 | pending | | pending | this log |
| WildChat English-only | 5 MB sample | 0.15–0.16 | pending | | pending | this log |

Neural wins on prose at +34.8%. Classical wins on highly-structured
JSON (chunking penalty) and templated logs (needs CLP). Dialogue is
the open question this spike closes.

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

### Compression run (in flight at writeup time)

- First production use of the **16 vCPU / 32 GB Fargate Spot Graviton2**
  setup (compression-stack changes from 2026-04-22, `-C target-cpu=neoverse-n1`).
- Expected: dispatcher will route most chunks to classical codecs
  (bzip3 most likely, per Spike 3 experience) because the neural
  path is entropy-limited by the multilingual fragmentation.
- Throughput expectation: ~0.10–0.15 MB/s on 16 vCPU, ~25–35 min wall
  time for 200 MB.

**Results section to be filled in when the run completes.**

### Pass gate call

Experiment 1 **will miss the required gate** (≥ 15% below zstd-22 on the
val split = ≤ 0.1576). At best the dispatcher ties zstd via classical
fallbacks; at worst it's 1–3% behind due to the chunking penalty
observed in Spike 3 exp 1 on homogeneous data.

That's **expected and not a product failure** — multilingual isn't
our target deployment. Experiment 2 (English-only) is the real test.

## Experiment 2 — WildChat English-only (queued)

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

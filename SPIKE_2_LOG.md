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

| hdfs-b1/v1 | B1 (6L × 192H, 18.7 M params, ctx 2048) | 16 384 | 6 | 2048 | 1058 | 3 | 8 | **~0.144 at best epoch (0)** | 0.0466 | 0.0457 | ~55 min | ~$0.73 | **FAIL** | 18.7M total params (12.4M body). Final-epoch eval bits/token 3.97; best-epoch (0) was 3.76. **Eval loss climbed across epochs** — 2.61 → 2.72 → 2.75 nats — classic overfitting trajectory. Ratio even at best epoch (~0.144) is only marginally better than 200K baseline (0.1405). Capacity scaling didn't help on HDFS; same pattern as A: variable fields (block IDs, IPs) are genuinely hard regardless of capacity, more params just memorize training-set-specific values. Also hit WKV T_MAX=2048 assertion when trying ctx=4096 first — vendor kernel is hardcoded. Noted. |

### Phase B verdict

Capacity scaling alone does not help on HDFS. The pattern from A
repeats: bigger models memorize training-specific variable-field
values (block IDs, IPs) that don't appear in val, hurting generalization.
And after epoch 0 we see clear overfitting.

**Pivot to Phase C (template-absorbing tokenizer)** — let SPM's
`max_sentencepiece_length` find whole HDFS templates as single pieces
so the model doesn't have to predict template bytes at all.

| hdfs-c1/v1 | C1 (max_piece_len=256, baseline model) | 16 384 | 2 | 2048 | 1058 | 5 | 32 | **0.1010** (bits/token 2.198 / val bytes/token 2.720) | 0.0466 | 0.0457 | ~40 min | ~$0.53 | **FAIL but biggest improvement yet** | 28% better than Spike 1 (0.1405 → 0.1010). Still 2.17× worse than zstd. SPM did learn 196 pieces ≥ 30 chars, longest 63 chars — but they're training-specific Hadoop paths like `/user/root/sortrand2/_temporary/_task_200811101024_0003_r_00001`. Val has different task IDs → fragments these back to byte-fallback → val bytes/token 2.72, SHORTER than Spike 1's 3.02. Token density hurt, but per-token predictability improved much more (bits/token 3.39 → 2.20). Still the wrong compression axis — need templates that *generalize* across train/val. |

### Phase C1 diagnosis: template mining with unigram SPM overfits paths

Longest 10 pieces from C1 tokenizer, all `_task_200811101024_0003_r_XXXXX`
prefixes. SPM's unigram trainer can't distinguish "common template
with varying suffix" from "entire frequent substring"; with a
budget of 16 K vocab slots it burns slots on full training-path
variants. Val sees different tasks → byte-fallback.

Fix directions, in order of engineering cost:
- C2: bigger vocab (32K-128K) with max_piece_length=256, unigram.
  More slots may cover enough training-path variants that val hits
  fewer misses. Still training-specific, so bounded.
- C3: switch to BPE. BPE greedily merges character pairs, tends to
  produce template PREFIXES rather than full paths. Natural split
  point between "template" and "variable".
- C4 (proper fix): mask known variable fields
  (timestamps `\d{6} \d{6}`, block IDs `blk_-?\d+`, IPs
  `\d+\.\d+\.\d+\.\d+`, task IDs `_task_\d+_\d+`) with placeholders
  BEFORE SPM. Templates stabilize; SPM learns the real structure.
  Rust runtime will need the same preprocessing + postprocessing.

| hdfs-c2/v1 | C2 (64K vocab + max_piece=256) | 65 536 | 2 | 2048 | 1058 | 3 | 4 | **0.1013** | 0.0466 | 0.0457 | ~80 min | ~$1.07 | FAIL | 4× the vocab of C1; essentially no change in ratio. Confirmed vocab scaling is saturated on HDFS. |
| hdfs-c4b2/v1 | C4b2 (normalized SPM + 16K…1024) | 1 024 | 2 | 2048 | 1058 | 5 | 32 | **0.0816** (bits/token 1.22 / val bytes/token 1.87) | 0.0466 | 0.0457 | ~26 min | ~$0.35 | FAIL — but big improvement | Train SPM on normalized corpus, tokenize + train RWKV on ORIGINAL corpus. SPM learned only 1116 unique pieces on the normalized 2K-skeleton space. At encode time on original text, SPM falls back to byte-level for variable fields → val bytes/token drops to 1.87. **Bits/token drops dramatically (3.39 → 1.22, -64%)** because the model only needs to predict next byte/short-token given current context. Net ratio 0.0816 — 43% better than baseline, 1.75× worse than zstd. |

### Phase C verdict (so far)

Pattern across A/B/C: the bottleneck is **variable-field bytes**
(block IDs, IPs, timestamps). No amount of vocab or model capacity
makes a small learned model match zstd's 128 MB hash-dictionary on
a highly-repetitive templated corpus, because zstd's LZ references
are near-optimal for this data shape.

C4b2 got us closer than anything else (**ratio 0.0816 vs zstd
0.0466 = 1.75×** worse) by effectively byte-predicting with
normalized-template context. The remaining gap is variable-field
entropy — block IDs and timestamps the model can't predict without
explicit retrieval.

**To actually beat zstd on HDFS requires either:**
1. A **proper hybrid codec** with dictionary-lookup for
   high-repetition substrings + model for residual bytes. Weeks of
   engineering. Essentially reimplementing zstd's win on top of
   our model.
2. A **much larger long-context model** (100 M+, ctx 16 K+) that
   can retrieve previous block IDs from within its own context.
   Blows the inference envelope.
3. Switching corpora to one where LZ isn't near-optimal (JSON
   events, prose) — the user has explicitly ruled this out.

Running **C5 (1 M-param model, 10 epochs, same normalized setup)**
as the last budgeted within-envelope attempt before concluding
the spike and presenting options to the user.

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

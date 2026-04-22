# Spike 2 — beat zstd on HDFS within a production envelope

## Constraints (hard)

1. **Ratio gate:** `held_out_ratio < 0.98 × zstd_baseline_ratio` on HDFS.
2. **Training cheap + fast:** each experiment budget ≤ $10 and
   ≤ ~2 hr walltime; total spike budget target ≤ $50, walltime ≤ 1 day.
3. **Inference reasonable for a production service:**
   - Compression throughput **≥ 100 KB/s single-stream** on L4/A10G
     GPU (1 GB batch compresses in ≤ 3 hr, acceptable async).
   - Decompression throughput **≥ 1 MB/s single-stream** on L4/A10G
     (10 MB chunk decodes in ≤ 10 s, meets retrieval SLA).
   - GPU-batched decompression can later scale throughput linearly by
     batching independent blobs, so single-stream speed is the floor.

These constraints together mean **we want the smallest + simplest
model that clears the ratio gate**, not the biggest. The thing to
minimize is ratio-per-flop.

## Where Spike 1 landed

HDFS_v1 dataset, 1.39 GB NDJSON:

| metric | value |
|---|---|
| model entropy-bound ratio | **0.1405** (1.124 bits/byte) |
| `zstd --long=27 --ultra -22` ratio | **0.0466** (0.373 bits/byte) |
| model vs zstd | **3.01× worse** |

Model: 200K-param RWKV (2L × 96H), 16K SPM vocab, 2048 ctx.

## Why we lost (one-sentence recap)

HDFS lines are 90%+ template with a few variable fields; zstd with
128 MB window gets the template for ~5 bits per line, our model pays
full entropy cost every line because it lacks capacity + context to
reuse a template it saw 10 MB ago.

## Dials, ordered by ratio-per-flop (updated for cost + speed)

Cost/speed-aware order is very different from "just scale params":

| # | dial | predicted ratio on HDFS | training cost | inference cost vs 200K | unlock |
|---|---|---|---|---|---|
| 1 | **Larger SPM vocab (32K → 64K → 128K) + train tokenizer on whole corpus** | 0.10 → 0.08 | $1 / run | ~neutral (per-token cost scales with log vocab) | longer tokens → fewer predictions per byte |
| 2 | **Template-aware tokenization** (let SPM learn whole log templates as single tokens) | 0.07 | $1 | ~neutral | dramatic effect if SPM picks up template pieces ≥100 chars |
| 3 | **Mild capacity bump: 200K → 2M (6L × 192H)** | 0.06 | $3 | ~10× slower | enough params to actually learn the template distribution |
| 4 | **Longer context 2048 → 8192** | -5 to -10 % bits | +20% cost | +20% inference | more templates in history |
| 5 | **Hybrid: line-level dedup + RWKV on variable fields only** | 0.03-0.04 | $2 (tiny model) | small model → very fast | template detector does the LZ job, model does only the residual entropy — exactly what zstd does but better on variable fields |
| 6 | **Bigger model 2M → 20M** | 0.04 | $10 | ~100× slower — at the edge of prod envelope | only if #1-5 don't clear the gate |
| 7 | **100M+ params** | 0.03 | $30 | ~500× slower — **violates inference envelope** | rejected under the constraints unless absolutely necessary |

**Ordering rationale under the revised constraints:**

- #1 and #2 are one-training-run experiments that change the
  tokenizer, not the model. Cheapest possible wins. If HDFS templates
  get absorbed into SPM tokens, the ratio collapses without any
  capacity increase.
- #5 (hybrid) is promoted high because it targets the exact failure
  mode: zstd wins on templates, model wins on variable fields. A
  hybrid gets both. Implementation cost is moderate (detect repeated
  substrings via SA-LZ or similar, encode line as
  `template_id + variable_slot_tokens`) but still bounded to a few
  days of work.
- Capacity scaling (#3 → #6) is still in the plan but downstream of
  cheaper experiments. #7 is rejected by constraint #3 — a 100M
  model's inference is too slow for production even on GPU batching.
- #4 (context) is practically free because RWKV scales linearly.
  Fold it into the #3 experiment.

## Inference-speed sanity check

From `l3tc-rust/README.md`, a 200K RWKV runs 89 KB/s compress, 92
KB/s decompress on a laptop CPU. GPU (L4/A10G, ~80× faster for small
batch inference) should give ~7 MB/s compress, ~7 MB/s decompress
single-stream. Headroom of 70× on compress, 7× on decompress against
our gates.

Scaling to 2M params: RWKV inference cost is roughly linear in
params for models in this regime → 10× slower → ~700 KB/s decompress.
Still over the 1 MB/s floor **only if** we batch at least 2 blobs
concurrently on the GPU. Acceptable.

Scaling to 20M params: ~100× slower than 200K → ~70 KB/s
decompress single-stream. **Below the floor without batching.**
Needs ~15× batch-parallelism to recover. Still doable on a single
L4 (24 GB VRAM), but adds complexity. That's the soft ceiling.

Scaling to 100M+: single-stream ~7 KB/s decompress. Would need
100+ concurrent blobs in flight to meet the SLA. Kills product UX
and violates the envelope. **Rejected.**

## Experiment matrix

Each run: git commit → CodeBuild → new JobDef revision → `batch
submit-job` → read `metadata.json`. Runs are independent; I'll drive
them sequentially and short-circuit as soon as something clears the
gate.

### Phase A — tokenizer experiments (≤ 1 hr each, ≤ $2 each)

| exp | vocab | SPM sample | model | context | expected ratio | gate to pass |
|---|---|---|---|---|---|---|
| A1 | 32 K | full train (1.1 GB) | 200 K | 2048 | 0.11 | prove scaling trend |
| A2 | 65 K | full train | 200 K | 2048 | 0.09 | prove scaling trend |
| A3 | 131 K | full train | 200 K | 2048 | 0.08 | diminishing returns check |

Decision after Phase A:
- Any of A1-A3 < 0.046 → **done, ship that config.**
- Otherwise: pick the best-ratio config as the tokenizer for Phase B.

### Phase B — mild capacity + context bump (≤ 2 hr each, ≤ $5 each)

| exp | vocab | model | context | steps | expected ratio |
|---|---|---|---|---|---|
| B1 | best from A | 2 M (6L × 192H) | 8192 | 100 K | 0.05-0.06 |

Decision after Phase B:
- B1 < 0.046 → **done, ship.**
- B1 between 0.046 and 0.08 → go to Phase C (hybrid) — capacity is
  within reach but the 2M model is already 10× more expensive at
  inference. Hybrid gets us further ratio cheaply.
- B1 > 0.08 → capacity helped less than expected; **skip straight to
  hybrid (Phase C)**.

### Phase C — hybrid: template dedup + model on residuals

This is the serious engineering lift, so it runs only if Phase A + B
fail. Scoped tightly:

1. **Template detection** — at SPM training time, build a suffix-array
   or rolling-hash index over the train corpus; identify the top N
   repeated substrings of length ≥ 32 bytes with frequency ≥ 10.
   These become N special tokens in the SPM vocab.
2. **Encoding** — at inference, greedy-match these long tokens first
   (like a minimal LZ pass), then fall through to regular SPM.
3. **Model** — same 200 K or 2 M RWKV, trained on the long-token-ified
   stream.

Expected ratio on HDFS: **0.02 - 0.03**. Model stays tiny, tokens
are ~100× denser on template material, variable fields still go
through the model. This is basically "zstd's behavior done right."

Implementation budget: ≤ 2 days end-to-end, including getting the
template-token encoder wired into the training container.

### Phase D — capacity last resort (≤ $10, ≤ 3 hr)

Only reached if Phase C has engineering blockers OR somehow fails
to clear the gate. Last-resort single experiment:

| exp | vocab | model | context | expected ratio |
|---|---|---|---|---|
| D1 | best tokenizer | 20 M (12L × 384H) | 8192 | 0.04 |

If D1 still loses, the spike fails. Escalate to user: thesis requires
rethinking — HDFS may be genuinely not worth competing on.

## Training-time budget management

From Spike 1 timings:
- 200 K model, 500 K steps, ctx 2048: ~67 min, ~$0.89 on g6.xlarge.
- 2 M model, same steps, ctx 8192 (Phase B1): estimated ~2-3 hr,
  ~$2-3. Walltime scales ~4× (10× params ≈ ~4× flops for small
  models; 2× context is linear in RWKV).
- 20 M model (Phase D): ~5-8 hr, ~$5-8.

Total if all phases run: **~$25-30, ~15-20 hr** of Batch compute.
Well within the $50 / 1-day envelope.

Retries for infrastructure flakes (Spike 1 burned 8 retries today)
don't count because we're now past the plumbing issues.

## Experiment instrumentation

Every run writes to S3 as model version `v{N}`. Each
`v{N}.metadata.json` gets extended from Spike 1's shape with:
- `vocab_size`
- `num_layers`, `hidden_size`, `intermediate_size`, `rwkv_rank`
- `context_len`
- `train_samples`, `train_steps`
- `train_wall_seconds`
- `tokenizer_bytes_per_token`
- `held_out_ratio` (real, not sentinel — Spike 1 rc=$? bug fixed)

I'll log every experiment into `SPIKE_2_LOG.md` with this table
shape + the decision made after each.

## Out of scope (same as Spike 1)

- Throughput beyond the "reasonable for production" envelope
  (single-stream ≥ 100 KB/s compress, ≥ 1 MB/s decompress on GPU).
- Non-HDFS corpora. Win HDFS first.
- Architecture changes off RWKV-v4 + HiRA.
- Production hardening (PRODUCTION_TODO remains open).

## Stopping criteria

- **WIN** at any phase: ratio < 0.98 × zstd ratio, training walltime
  < 4 hr, and predicted decompression > 1 MB/s single-stream on
  L4 GPU → stop, write up, ship.
- **HARD FAIL** at Phase D: user decision needed. Options:
  a) stretch training budget to 100 M params and accept decompression
     batching complexity,
  b) invest deeper in hybrid (Phase C with more templates, better
     detector),
  c) abandon HDFS-specific goal and pivot the product to
     variable-content corpora only.

## Execution order (once Spike 1 compression verification lands)

1. Phase A1 (32 K vocab) — smallest change, fastest feedback.
2. Decide Phase A next step based on A1's ratio.
3. Run A2 / A3 as needed.
4. Phase B1 with the best Phase A tokenizer.
5. Decide C vs D based on B1.
6. Final `SPIKE_2_LOG.md` writeup + product recommendation.

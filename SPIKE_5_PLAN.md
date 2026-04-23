# Spike 5 — bigger model on GPU

## Why this spike

Two signals that land together after Spikes 3 and 4:

1. **200K params is near its capacity ceiling on chat data.** Spike 4
   exp 1 on WildChat multilingual hit `held_out_ratio = 0.568` (worse
   than no compression). Exp 2 on English-only is in flight; even if
   it clears the required gate, it's expected to land near 0.13 —
   ~15% headroom over zstd, not the decisive 25–40% the narrowed
   product pitch leans on.

2. **The GPU path is open at 1M+ params.** Per the math check in
   `COMPRESSION_OPTIMIZATION.md` + our session review, the
   CPU-vs-GPU crossover for our workload is around **200–500K
   params**. Above that, a well-tuned GPU backend beats Graviton
   CPU. At 10M params the gap is 30× in GPU's favor, and the model
   compresses dialogue meaningfully better (LMCompress literature
   + transformer scaling laws → ~20–30% additional ratio
   improvement vs 200K).

Spike 5 tests both at once: **does a 10M model, run on GPU,
decisively beat zstd on English LLM chat AND hit the ≥500 KB/s
throughput envelope?**

Pass = the "Track 3" product thesis is live: per-customer 10M
models, GPU-backed compression, $/GB and ratio both defensibly
better than any general compression on this data. That's the
pitch to Abridge / Harvey / Hebbia at closing stage.

Fail = we either stay at 200K (smaller margins but shippable), or
the GPU throughput doesn't materialize and we need a different
approach (CPU batching, different model family, or give up on
speed SLO).

## Dependencies

- **Spike 4 exp 2 complete** (English-only 200K baseline). Spike 5
  needs that number to measure the 10M uplift against.
- **CUDA port of the neural forward pass** in `l3tc-rust/`, OR an
  ONNX Runtime CUDA path that plugs into the dispatcher. Details
  in the engineering-steps section below.
- **ECS-on-GPU compression compute env** — one-time CDK addition;
  existing Fargate path stays for small-model / small-corpus
  customers.

## Corpus

**WildChat-English, 200 MB.** Same exact corpus as Spike 4 exp 2
so the 200K vs 10M comparison is clean (same tokenizer, same val
split, same zstd baseline 0.1526).

No second corpus for this spike — the question here is **model
size + inference compute**, not corpus variety. OASST2 / other
dialogue corpora are noise relative to that variable.

## Model spec

| param | Spike 4 (200K) | **Spike 5 (10M)** | delta |
|---|---|---|---|
| layers | 2 | **4** | 2× depth |
| hidden dim | 96 | **256** | 2.7× width |
| ctx_len | 2048 | 2048 | — |
| vocab (SPM unigram) | 16K | 16K | — |
| HiRA rank | 4× base | 4× base | — |
| params | ~200K | **~10M** | 50× |

50× parameter count is deliberate — below that the GPU advantage
is marginal, above it training cost starts to bite per customer.
10M sits in the literature sweet spot for small LM compressors
(NNCP's 1.7M model beats zstd by ~55% on enwik8; scaling to 10M
should saturate the compression curve for this model family).

Training stays per-customer (our design differentiator):
- 10 epochs, batch=32, LR=1e-4 (same as Spike 4 — fair comparison)
- Expected training cost: ~$5–15 per dataset on g6.xlarge
  (~3–4 hr wall clock vs 1.5 hr for 200K)

## GPU inference path — pick one

Two paths to get the 10M model running on GPU. Picked before
spike kicks off.

### Option A: ONNX Runtime + CUDA (lower engineering cost)

- Export trained RWKV to ONNX via `torch.onnx.export` after training
- `compression_worker.py` loads ONNX model via `onnxruntime-gpu`
  instead of calling `l3tc` binary
- Arithmetic coder stays in Rust/Python — ONNX only handles the
  logits forward pass
- **Effort**: ~1 week (ONNX export + worker integration)
- **Pros**: fast to ship; ONNX Runtime has mature CUDA optimization
  (graph fusion, kernel selection, persistent command buffers)
- **Cons**: introduces a new runtime dependency; throughput probably
  10–20% below hand-tuned CUDA; determinism story is ORT-version-dependent

### Option B: CUDA port of l3tc-rust neural backend (higher engineering cost)

- Add `backend::cuda` module to `l3tc-rust/` parallel to the existing
  `backend::metal`
- Implement the head matvec + cum_freqs + RWKV forward in CUDA via
  `cuda-rs` or `wgpu`
- Keep the single-binary deployment model
- **Effort**: ~2–3 weeks
- **Pros**: maximum throughput; deterministic per-backend (compatible
  with the existing `FLAG_GPU_ENCODED` blob-format convention);
  no new runtime dependency
- **Cons**: CUDA kernel work is real; longer to validation

**Recommendation: Option A first** for Spike 5. We want to learn
*whether* GPU delivers the claimed throughput + ratio improvement
quickly. If yes, Option B becomes the production-path work. If no,
Option B would have been premature anyway.

## Pass gates

Two gates, both must pass (this is the AND of ratio AND throughput):

### Gate 1 — ratio (stricter than Spike 4)

Compared to zstd-22 whole-file baseline 0.1526 on WildChat-English:

| gate | dispatcher ratio | vs zstd-22 |
|---|---|---|
| Required | ≤ 0.1145 | ≥ 25% smaller |
| Strong | ≤ 0.0992 | ≥ 35% smaller |
| Stretch | ≤ 0.0840 | ≥ 45% smaller |

Required is **25% better than zstd** (not 15% as Spike 4) — the
bigger model should open this margin up, and 25% is where the
product's sales pitch starts to feel genuinely differentiated
against compliance-archive incumbents.

### Gate 2 — throughput (on GPU)

Measured end-to-end compression throughput on the ECS-on-g5.xlarge
(A10G) or g6.xlarge (L40S) path:

| gate | throughput | notes |
|---|---|---|
| Required | ≥ **300 KB/s** | matches our Fargate 16 vCPU baseline; minimum bar to not regress on speed |
| Strong | ≥ **500 KB/s** | hits the Spike 4 speed-optimization stretch; $/GB comparable to Fargate-classical |
| Stretch | ≥ **1000 KB/s** | the Track 3 thesis number from compression_optimization.md |

Both gates must pass. 45% ratio with 50 KB/s is useless — customer
won't wait 11 hours to compress a day's logs. 300 KB/s at 15%
ratio isn't enough to justify the GPU infrastructure cost over
Fargate-CPU.

## Service infrastructure changes

### Phase A (spike) — single g5.xlarge, manual

Stand up one g5.xlarge EC2 via ECS RunTask. Pull the spike4-en
trained model + run hybrid-compress (with ONNX neural path) on the
5 MB sample. Measure. Not production-grade — no auto-scaling,
manual teardown.

### Phase B (if spike passes) — production ECS-on-GPU

- New `compression-stack.ts` capacity provider with `g5.xlarge`
  + `g6.xlarge` instance types
- Keep Fargate-CPU as a second capacity provider (small-model
  customers stay there via metadata-driven routing)
- ECS service-level strategy: `FARGATE_SPOT` primary for 200K
  datasets, `EC2` on g5/g6 for ≥ 1M-param datasets
- Task-protection + min=0 scale-to-zero stays (already working
  for Fargate Spot)

Phase B is engineering work outside the spike's scope. Spike 5
is only Phase A — the minimum to answer "does the approach
work."

## Execution plan

| step | effort | $ | gate |
|---|---|---|---|
| 1. Wait on Spike 4 exp 2 completion | — | — | baseline number for comparison |
| 2. Train 10M model on WildChat-English (Batch, env override NUM_LAYERS=4 HIDDEN_SIZE=256) | ~4 hr | ~$5 | loss curve healthy + convert_checkpoint survives on 4L × 256h |
| 3. ONNX export + local sanity check (Mac, CPU-ORT) | ~2 days | $0 | round-trip equivalence vs PyTorch |
| 4. Spin up one-off g5.xlarge EC2, pull model + run ONNX hybrid-compress on 5 MB English sample | ~2 days | ~$5 | compiles + round-trips |
| 5. Measure ratio + throughput | ~0.5 day | — | **gate 1 + 2** |
| 6. Write SPIKE_5_LOG.md; commit-or-no-commit call | ~0.5 day | — | decision |

**Total to reach decision: ~1 week wall clock, ~$10-15 compute.**

## Cost model after commit

If gates pass + we commit to Phase B:

| path | per-GB-compressed cost projection |
|---|---|
| Fargate 16 vCPU Spot (current, 200K) | ~$1.32/GB at 0.17 MB/s |
| ECS g6.xlarge on-demand (10M, spike-realistic) | ~$1.20/GB at 300 KB/s |
| ECS g6.xlarge on-demand (10M, spike-stretch) | ~$0.36/GB at 1000 KB/s |

The $/GB target from `COMPRESSION_OPTIMIZATION.md` was ≤$0.05/GB.
Realistic with 10M + current GPU economics is **$0.30–0.50/GB**
— 3–10× what the doc set but still better than Fargate-CPU at
200K. Hit the aggressive target requires either bigger Spot
discount on GPU instances (Spot A10G is available but bursty) or
further throughput tuning (maybe CUDA graphs). Note in the customer
pitch.

## Decision tree

| ratio | throughput | what it means | next |
|---|---|---|---|
| **STRONG (≥35%)** | **STRONG (≥500 KB/s)** | Track 3 thesis live. Product story at full power. | Commit to Phase B: ECS-on-GPU prod infra + first customer outreach at this quality level |
| **STRONG** | REQUIRED (300 KB/s) | Ratio wins but speed is tight. Pitch "comparable speed to Fargate + 35% better ratio". | Ship anyway; follow-up Spike 6 on throughput (CUDA graphs, batch=512) |
| REQUIRED (25%) | STRONG | Speed works but ratio gain modest. 25% is still defensibly better than Fargate-CPU + 200K. | Ship Phase B; smaller pitch than ideal but still valid |
| REQUIRED | REQUIRED | Both gates cleared at the floor. Marginal but valid. | Ship Phase B; close monitoring of first-customer metrics; revisit if gains are too thin |
| **miss ratio** | pass throughput | 10M didn't meaningfully improve ratio over 200K. Likely model-architecture ceiling (RWKV-v4 might be the wrong backbone for dialogue at this size). | Revisit architecture (Mamba-2, transformer, different tokenizer) before committing more engineering |
| pass ratio | **miss throughput** | ONNX Runtime isn't fast enough. CUDA port (Option B) might close gap; or batching strategy needs work. | Do Option B (hand-tuned CUDA) before committing infra |
| **miss both** | | Both levers failed. Track 3 is dead; either stay 200K forever or look at fundamentally different model family. | Hard pause, rethink |

## Risks

1. **Training cost scales with customer count.** 50 regulated-AI
   customers × $10 per training run × retrainings/year = meaningful.
   Negotiable: (a) shared-model warm-start + fine-tune per customer,
   (b) cheaper Spot GPU for training, (c) smaller model for customers
   who don't need the top-tier ratio.

2. **GPU Spot capacity.** We hit `UnfulfillableCapacity` on
   g5.xlarge during Spike 1 training. A10G / L40S Spot pools are
   smaller than general-purpose. Phase B will need on-demand
   fallback (same pattern as Batch training).

3. **Blob format versioning.** Bigger model + GPU backend → more
   FP-determinism boundaries. Existing `FLAG_GPU_ENCODED` covers
   Metal vs CPU; we'd need to extend to include CUDA-vs-Metal-vs-CPU
   identification so a blob encoded on one backend isn't decoded
   on another. Two lines of Rust + a blob-version bump.

4. **ONNX export fidelity on RWKV.** RWKV's WKV kernel is a
   custom CUDA extension at training time; ONNX export may need
   a manual replacement op. Worth confirming before committing
   engineering.

5. **Per-conversation deletion still unsolved.** Not a Spike 5
   issue but reminder that it's a prerequisite for the
   regulated-vertical pitch (`CUSTOMER_PROFILE.md` open item).

## Out of scope (defer to later spikes / work)

- Decompression throughput on GPU (important but separate SLO)
- Tier 2 field-aware columnar (still TBD)
- CLP port (Tier 1+ deferred; orthogonal to model-size work)
- Compliance certifications (BAA, SOC 2, HITRUST) — ops timeline
- Multi-customer shared-base-model + fine-tune (training cost
  mitigation — future engineering, not needed for spike validation)

## Success criterion

**Spike 5 succeeds if both gates pass at the REQUIRED level:
dispatcher ratio ≤ 0.1145 (25% smaller than zstd-22) AND
end-to-end compression throughput ≥ 300 KB/s on the GPU path.**

That's the floor for committing to the ECS-on-GPU production
infrastructure and pitching the Track 3 product story to
first-customer candidates.

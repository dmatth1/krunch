# Optimizations

Speed and ratio levers identified via `l3tc profile` and code survey.
Covers both compress throughput and bits-per-byte ratio. Items with
`[NEW]` came from the 2026-04 survey; unmarked items predate.

## Current bottleneck breakdown

From `l3tc profile` on fiction.txt (152 KB, single-thread, 6L × 96H × 32K):

| phase | µs/step | % of time |
|-------|---------|-----------|
| forward pass (RWKV + head matvec) | 541 | 84.5% |
| cum_freqs (softmax + prefix sum) | 99 | 15.4% |
| AC encode | 0.26 | 0.04% |

Decompress is ~2× slower than compress on the 6L model due
to a linear search in the AC decoder.

For comparison, the 2L × 32K model:

| phase | µs/step | % of time |
|-------|---------|-----------|
| forward pass | 234 | 82.2% |
| cum_freqs | 50 | 17.7% |
| AC encode | 0.17 | 0.06% |

Forward pass dominates at every parameter budget. Inside it,
the vocab-scale head matvec (`32768 × 96` INT8 AXPY) is the
single largest sub-cost; attention state update is the second.
Phase 4e confirmed the speed ceiling at current vocab/arch is
vocab-bound, not depth-bound — halving layers only bought 1.12×.

---

## Speed — Compress path

### 1. Head matvec: pre-widen INT8 columns

**File:** `l3tc-rust/src/tensor.rs:656`
**Impact:** ~30% reduction in head matvec time

The i8→f32 widening (`sxtl` + `scvtf`) happens inside the
inner 32K-iteration loop, limiting NEON autovectorization.
Pre-widening one column of i8 to f32 in a temp buffer before
the AXPY multiply would let the inner loop be a pure `fmla`
broadcast.

```
Current:  for j in 0..96 { for i in 0..32768 { out[i] += xs * (col[i] as f32) } }
Proposed: for j in 0..96 { widen(col, tmp); for i in 0..32768 { out[i] += xs * tmp[i] } }
```

### 2. Top-K cum_freqs truncation

**File:** `l3tc-rust/src/codec.rs:1388`
**Impact:** cum_freqs from 99 µs → ~10 µs (15% → 2% of total)

Most probability mass is in <1000 tokens. Zero out the tail,
compute cum_freqs only over the top-K. Cuts the 32K loop to
~1K per token. Requires matching truncation on the decode side
(set all non-top-K freqs to the minimum floor value).

Ratio cost: bounded by the mass given up to the floor. Needs
validation on enwik6 that the drift is <0.0005 bpb.

### 3. Vectorize cum_freqs prefix sum

**File:** `l3tc-rust/src/codec.rs:1450`
**Impact:** moderate (diminishing if top-K done first)

The scalar serial loop over 32K elements has a carried data
dependency (`cum[i+1] = cum[i] + freqs[i]`). A SIMD parallel
prefix scan (tree reduction) could help but is complex. Top-K
(#2) is simpler and more impactful — do that first.

### 4. Fuse time_mix exp loops [NEW]

**File:** `l3tc-rust/src/rwkv.rs:626-658`
**Impact:** ~20-25% forward-pass reduction

time_mix has ~8 separate scalar `exp()` calls per layer per token
(state update `exp(state_p - p)`, `exp(ww - p)`, etc.). The NEON
exp already exists (`exp_f32x4_neon`, tensor.rs:314-361) and
backs softmax. Fuse the state-update exp chains into a single
4-wide NEON pass over the 96-element state vectors.

Ratio-neutral (algebraically identical). Second-biggest compress
win after head pre-widen.

### 5. Precompute -exp(time_decay) at session reset [NEW]

**File:** `l3tc-rust/src/rwkv.rs:645`
**Impact:** 1 scalar exp loop per layer per token saved

`time_decay[i].exp()` is recomputed every token even though
`time_decay` is fixed per-block. Store `-exp(decay)` in
`LayerState` or alongside the weights at load time and read
directly. Trivial; does not affect #4's fusion scope.

### 6. INT8 attention + FFN projections [NEW]

**File:** `l3tc-rust/src/rwkv.rs` block matmuls, `tensor.rs` kernels
**Impact:** ~10-15% forward-pass reduction

Currently only the head is INT8. The 8 attention/FFN projections
per layer (key/value/receptance/output for time_mix, receptance/
key/value for channel_mix) are all f32 96×96. 4× memory-traffic
reduction on ~96 KB of weights. Phase 4a validated INT8 head is
ratio-neutral with per-column scales; extend the same scheme.

Needs diff-harness validation — block quantization error
compounds across depth, unlike the head.

### 7. Hand-tuned NEON for non-96 shapes [NEW]

**File:** `l3tc-rust/src/tensor.rs` dispatcher
**Impact:** 3.2M tier 26 KB/s → targeting 40+ KB/s

The 3.2M opt-in tier falls through to `matrixmultiply::sgemm` for
its 256×256 attention projections and 512×256 / 256×512 FFN.
Custom NEON matvec kernels matching the `matvec_96x96_neon`
hand-tuning would close most of the 200K/3.2M speed gap. Phase 4d
flagged this explicitly, never shipped.

---

## Speed — Decompress path

### 8. AC decode: binary search over cum_freqs

**File:** `l3tc-rust/src/arithmetic.rs:228`
**Impact:** decompress speed roughly doubles

Currently a linear O(V) scan per decoded token with a comment:
"Linear search is fine for small vocabularies; we can switch
to binary search later if the vocab gets large."

At 32K vocab: ~16K comparisons/token → ~15 with binary search.
The cum_freqs array is already sorted (it's a prefix sum), so
`partition_point` or manual binary search is a drop-in fix.

This is the single biggest decompress optimization.

---

## Speed — Both paths

### 9. Larger segment size

**Current:** 4096 bytes per segment.
**Impact:** better ratio + less per-segment overhead

Larger segments (8K–16K) mean:
- Fewer segments → less rayon dispatch overhead
- Fewer session resets → less wasted context
- Longer context window → better predictions → better ratio
- Tradeoff: less rayon parallelism on small files

Should benchmark 4K vs 8K vs 16K on enwik6 to quantify the
ratio improvement vs parallelism tradeoff.

### 10. Entropy-driven segment boundaries [NEW]

**File:** `l3tc-rust/src/codec.rs`
**Impact:** ratio tighten + throughput via fewer segments

Complementary to #9. Close segments when predicted entropy spikes
rather than on fixed byte counts. Runs of predictable content
(repetitive prose, config files, log boilerplate) stay in a single
segment → less framing overhead. Hard-to-predict regions get
boundary resets where they actually help prediction. Framing cost
is ~5 bytes/segment (v4 varint headers), so this translates
directly to bpb.

### 11. Per-core multi-stream SIMD batching [NEW]

**File:** `l3tc-rust/src/rwkv.rs`, `tensor.rs`
**Impact:** single-core throughput uplift on few-core machines

Interleave state updates for 2–4 segments on one core. Model fits
L1 (~150 KB for 200K INT8). Orthogonal to the existing rayon
cross-core parallelism — stacks on top of it. Helps on 2-4 core
targets and on serial audit/entropy-bound runs.

---

## Ratio — Model-level levers

### 12. INT4 head [NEW]

**File:** `l3tc-rust/src/tensor.rs`
**Impact:** 2× cum_freqs memory reduction, ~10-20% speedup; ratio TBD

Per-column INT8 scaling extends naturally to INT4. Phase 4a did
not validate INT4 accuracy — needs diff-harness run on the
entropy bound before committing. Speed bet, ratio-risk, and the
quantization error at 4 bits is non-trivial on the head because
the logit range is wide.

### 13. Ratio-preserving distillation [NEW]

**File:** `scripts/distill_l3tc.py` (Phase 4e infra retained)
**Impact:** tighten actual ratio at fixed forward cost

Phase 4e asked "can a smaller student match the bigger teacher's
ratio?" and failed — speed ceiling is vocab-bound, not
depth-bound. The flip question: train the production 2L/6L student
against a much larger teacher's soft distributions (CE to teacher
probs, not hard tokens). Same inference cost, tighter cross-
entropy, so tighter ratio. Uses existing distillation infra.

Biggest ratio lever that doesn't cost speed.

### 14. Byte-level tail model for unk payloads [NEW]

**File:** `l3tc-rust/src/codec.rs` unk-extraction path
**Impact:** potentially halve the 0.36 pp unk slice

Unk payloads are non-tokenizable bytes stored verbatim after the
Phase 4b2 extraction (~3,623 B on enwik6 = 0.36 pp of the 0.61 pp
gap-to-entropy-bound). A tiny byte-level n-gram or logistic
predictor over the last few bytes could compress these inside
an AC stream. This is the last reachable slice of gap-closing
at 200K capacity.

### 15. Smaller vocab with balanced unigram tokenizer [NEW]

**File:** `tokenizer_balanced_32k/` (Phase 11 retrain in progress)
**Impact:** head matvec and cum_freqs both scale linearly with vocab

If the Phase 11 tokenizer retrain holds ratio at 32K → 16K (or
further), both head and cum_freqs shrink proportionally. This is
a direct speed win that's already in flight — just needs the
ratio on enwik6 validated when the retrain finishes. Stacks with
#1, #2, #6, #12.

---

## Ratio — Architectural bets

### 16. Hybrid dispatch (Phase 8) [NEW]

**File:** new — per-segment dispatcher
**Impact:** rescues degenerate inputs; ratio-neutral on target content

Route near-random content (already-compressed files, encrypted,
raw binary) to zstd. Gate per-segment on entropy estimated from
the first N tokens: if predicted bpb > zstd's expected output,
fall back. Ratio-neutral on text (never triggers); graceful
degradation on binary. Already on the product roadmap as Phase 8.

### 17. Domain dispatch with specialized checkpoints [NEW]

**File:** new — tiny classifier + multiple checkpoints
**Impact:** solves the Phase 11 capacity ceiling without a bigger model

Phase 11 retraining 200K on a 10 GB diverse corpus traded enwik6
ratio for domain breadth. Lesson: 200K params cannot be universal.
Ship 3–5 specialized checkpoints (Wikipedia, code, logs, JSON,
generic fallback), auto-detect corpus from the first KB via a
tiny classifier (logistic over byte-n-gram features), swap at
load. Each checkpoint is ~200K params; swap cost is one
`Model::load_file` call.

Weeks of training commitment, but this is the path to "best
ratio on everything" at 200K inference cost.

### 18. Two-tier cheap/full predictor [NEW]

**File:** new — cheap predictor alongside RWKV Session
**Impact:** 2–3× throughput if ~60% of tokens are "easy"

Cheap predictor (bigram/trigram, or a tiny linear head over the
last RWKV state) handles high-confidence tokens. Full RWKV
forward pass fires only when the cheap predictor's top-K misses
the next token. Skips the forward pass entirely on predictable
content (common whitespace, XML structure, log boilerplate).

Ratio cost is bounded by how often the cheap predictor is wrong
about its own confidence, tunable via threshold. Architecturally
the biggest speed unlock left, orthogonal to everything above.

---

## Implementation order

Ordered by ratio-of-impact-to-effort, with ratio-neutral items
first and validation requirements flagged.

1. **Binary search decode** (#8) — trivial fix, biggest decompress win
2. **Top-K cum_freqs** (#2) — 15% compress reduction, needs ratio validation
3. **Head pre-widen** (#1) — biggest single compress win, ratio-neutral
4. **Precompute -exp(time_decay)** (#5) — trivial, small win, ratio-neutral
5. **Fuse time_mix exp** (#4) — ~20–25% forward, ratio-neutral
6. **Larger segments** (#9) + **entropy-driven boundaries** (#10) — config + bench
7. **INT8 blocks** (#6) — another 10–15% forward, needs diff-harness validation
8. **256×256 NEON** (#7) — unlocks 3.2M tier to match 200K scaling
9. **Vocab-aware tokenizer retrain** (#15) — already in flight; measure when done
10. **Ratio-preserving distillation** (#13) — biggest ratio lever at fixed speed
11. **Two-tier predictor** (#18) — architectural, biggest remaining speed unlock
12. **Domain dispatch** (#17) — product-direction dependent, weeks of training
13. **Hybrid dispatch / Phase 8** (#16) — product polish, not a ratio win on target
14. **Byte-level unk model** (#14) — last reachable slice of entropy-bound gap
15. **INT4 head** (#12) — ratio-risky, only if speed still wanted after the above
16. **Prefix sum SIMD** (#3) — complex, only if cum_freqs still matters after #2
17. **Per-core multi-stream** (#11) — niche (few-core / serial paths only)

# Phase 12 — CPU optimization sweep (NEON, fused kernels, kernel polish)

**Goal:** drive single-thread and multi-thread CPU compress/decompress
throughput as high as possible without regressing ratio. Built on
top of Phase 4-11. Concluded April 2026.

**Status:** complete. Eight shipped commits (12a-h), several reverts
documented inline. Tactical CPU lever set is exhausted on the 200K /
16K-vocab default model. Remaining gains are architectural — see the
"big-picture" section at the end. GPU work continues as Phase 13.

---

## Results (clean system, 1MB enwik6)

| state | compress | decompress | single-thread |
|---|---:|---:|---:|
| CLAUDE.md baseline (pre-Phase-12) | 131 KB/s | 128 KB/s | ~8 KB/s |
| **Phase 12 cumulative (a–h)** | **172 KB/s** | **180 KB/s** | **22.7 KB/s** |
| **Speedup** | **+31%** | **+41%** | **+184%** |

3.2M opt-in tier: **40 KB/s** compress (+58% over Phase 4d baseline).

Memory-bandwidth bound from 10 threads up. Ratio held at 0.1699,
entropy bound 0.163723 — exact match with Python L3TC reference.

Phase 12 work is portable to any ARM64 (Linux, Android, embedded)
since the NEON intrinsics are stable Rust. On x86_64 the kernels
fall back to scalar paths and would need parallel AVX2/FMA ports
to recover similar gains.

## Phase 12 — what got done

✅ shipped, ❌ tried and reverted, ⏭ deliberately skipped.

| | item | one-liner |
|---|---|---|
| ✅ 12a | NEON `sub_exp` in `time_mix` | replaces 4 scalar exp loops/layer; ratio-neutral |
| ✅ 12a | binary-search AC decode | +5–10% decompress (was linear O(V) scan) |
| ✅ 12a | precompute `-exp(time_decay)` at load | one saved exp/layer/token; trivial |
| ✅ 12b | NEON `max_f32` reduction (cum_freqs Pass 1) | cum_freqs 80→25 µs/step |
| ✅ 12c | NEON `sigmoid` (safe `-|x|` form via `vbslq`) | +3–5% throughput, entropy bound exact |
| ✅ 12d | hand-tuned NEON 16-wide INT8 head matvec | +11% ST, +4–5% MT |
| ✅ 12e | fused `time_mix` step1+step2 NEON kernels | +0.5% ST, +3–4% MT (cuts L1 round-trips) |
| ✅ 12f | NEON `layer_norm` (3-pass NEON reduction) | −1 µs/step on forward |
| ✅ 12g | NEON 256×256 + 256×512 + 512×256 matvecs (3.2M tier) | 3.2M: 25→40 KB/s (+55-63%) |
| ✅ 12h | drop `saturating_add` on cum_freqs prefix sum | +3% compress (overflow provably impossible) |
| ❌ — | head matvec pre-widen (simple variant) | regressed compress −12% — 128KB scratch + 128KB out blow M1 L1 |
| ❌ — | chunk-skip in softmax Pass 2 (4-wide and 16-wide) | sub-noise; NEON exp polynomial too cheap to gate |
| ❌ — | `lto = "fat"` | equivalent throughput on memory-bound workload, breaks PGO |
| ❌ — | larger default segment (4096 → 8192) | monotonic ratio gain at monotonic compress speed cost (~6% per 2×); kept at 4096 |
| ❌ — | `time_mix3` fused triple-blend helper | regressed ST forward +2-3% (LLVM autovec broken by larger arg list, L1 already cached originals) |
| ⏭ — | INT8 attention/FFN block projections | matvec_96x96_neon already ~3.2 µs/token (~2% of budget); ratio risk not worth it |
| ⏭ — | INT8 embedding row lookup | one row × 384 bytes/token at L2 latency ≈ 50 ns; INT8 saves ~30 ns (0.07%) |
| ⏭ — | top-K cum_freqs via quickselect | finding K-th largest costs 50–150 µs vs 12 µs polynomial savings; net loss |
| ⏭ — | rANS replacing Nayuki AC | ST decompress (23.6) > ST compress (22.5) — AC decode already free after 12a |
| ⏭ — | PGO (profile-guided optimization) | rustc 1.94.1 LLVM value-profile crashes intermittently; one run measured +5-6%, defer to future rustc |

## Open tactical leftovers (small)

One-liner each — kept as a list for future-reference, but none are
high-leverage on the current 200K/16K default model.

- **NEON prefix-sum on cum_freqs Pass 3b** — capped at ~3% throughput; complex (carried u64 dependency, u32→u64 widening at AC boundary).
- **Larger / entropy-driven segment boundaries** — ratio gain available but pure-speed cost; opt-in via `--segment-bytes 8192` already works.
- **Per-core multi-stream SIMD batching** — niche (helps few-core machines and serial audit paths only).
- **INT4 head** — ratio-risky per arXiv 2301.12017; final projection is the most fragile layer.
- **Byte-level tail model for unk payloads** — caps at 0.18 pp ratio improvement (half of the 0.36 pp unk slice).

## Big-picture / architectural levers (multi-day to multi-week)

These are where the genuine remaining gains live. All require
training infrastructure, not just inference-side code changes.

### Ratio-preserving distillation

**File:** `scripts/distill_l3tc.py` (Phase 4e infra retained)
**Time:** ~1-2 weeks GPU, no inference-side code changes
**Impact:** **biggest ratio lever at fixed inference cost**

Phase 4e asked "can a smaller student match the bigger teacher's
ratio?" and failed — speed ceiling is vocab-bound, not depth-bound.
The flip: train the production student against a much larger
teacher's soft distributions (cross-entropy to teacher probs, not
hard tokens). Same forward pass cost, tighter cross-entropy →
tighter ratio. Plausibly closes most of the 0.17 → 0.13 gap (200K
→ 3.2M) while keeping 200K-tier inference speed.

### Two-tier cheap/full predictor

**File:** new — cheap predictor alongside RWKV `Session`
**Time:** ~2-3 weeks (train cheap predictor, plumb through codec)
**Impact:** **biggest speed unlock left** — speculative 2-3× compress

Cheap predictor (bigram/trigram, or a tiny linear head over the
last RWKV state) handles high-confidence tokens. Full RWKV forward
pass fires only when the cheap predictor's top-K misses the next
token. Skips the forward pass entirely on predictable content
(whitespace, XML structure, log boilerplate). Ratio cost is bounded
by how often the cheap predictor is wrong about its own confidence,
tunable via threshold.

### Smaller vocab via tokenizer retrain

**File:** `tokenizer_balanced_32k/` (Phase 11 retrain in progress)
**Time:** in flight as parallel work
**Impact:** head matvec and cum_freqs both scale linearly with vocab

If the Phase 11 tokenizer retrain holds ratio at 32K → 16K (or
further), both head and cum_freqs shrink proportionally. Direct
speed win that's already in flight — measure when the retrain
finishes. Stacks multiplicatively with everything in Phase 12.

### Domain dispatch with specialized checkpoints

**File:** new — tiny classifier + multiple checkpoints
**Time:** weeks of training (one checkpoint per domain)
**Impact:** solves the Phase 11 capacity ceiling without a bigger model

Phase 11 retraining 200K on a 10 GB diverse corpus traded enwik6
ratio for domain breadth. Lesson: 200K params cannot be universal.
Ship 3–5 specialized checkpoints (Wikipedia, code, logs, JSON,
generic), auto-detect corpus from the first KB via a tiny classifier
(logistic over byte-n-gram features), swap at load. Each checkpoint
is ~200K params; swap cost is one `Model::load_file` call. The path
to "best ratio on everything" at 200K inference cost.

### Hybrid dispatch (Phase 8)

**File:** new — per-segment dispatcher
**Time:** ~1 week (segment classifier + zstd plumbing)
**Impact:** rescues degenerate inputs; ratio-neutral on text

Route near-random content (already-compressed files, encrypted,
raw binary) to zstd. Gate per-segment on entropy estimated from the
first N tokens: if predicted bpb > zstd's expected output, fall back.
Already on the product roadmap. Not a ratio win on target content
(text), but turns a hard failure (compressing already-compressed
data) into graceful degradation.

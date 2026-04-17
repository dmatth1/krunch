# Optimizations

Speed and ratio levers for the L3TC Rust runtime. Status legend:
✅ shipped, ❌ tried and reverted, ⏭ deliberately skipped,
🟢 open / un-attempted.

## Phase 12 results (2026-04, clean system, 1MB enwik6)

| state | compress | decompress | single-thread |
|---|---:|---:|---:|
| CLAUDE.md baseline (pre-Phase-12) | 131 KB/s | 128 KB/s | ~8 KB/s |
| **Phase 12 cumulative (a–f)** | **168 KB/s** | **178 KB/s** | **22.5 KB/s** |
| **Speedup** | **+29%** | **+39%** | **+180%** |

Memory-bandwidth bound from 10 threads up — `RAYON_NUM_THREADS`
10/12/16/20 all measure ~151–170 KB/s (variance ±10) on a clean
idle system. Tactical CPU work for the 200K/16K default model is
exhausted; remaining headroom is architectural.

**Ratio:** 0.1699 held throughout. Entropy bound 0.163723 — exact
match with Python L3TC reference (no model-side regression).

The Phase 12 work is portable to any ARM64 (Linux/Android/etc.)
since the NEON intrinsics are stable Rust; on x86_64 the kernels
fall back to scalar paths and would need parallel AVX2/FMA ports
to recover similar gains.

## Done in Phase 12 — shipped

| commit | item | result |
|---|---|---|
| 12a | NEON `sub_exp` in `time_mix` (replaces 4 scalar exp loops/layer) | ratio-neutral |
| 12a | Binary-search AC decode (was linear O(V) scan) | +5–10% decompress |
| 12a | Precompute `-exp(time_decay)` at load | trivial; one saved exp/layer/token |
| 12b | NEON `max_f32` reduction on cum_freqs Pass 1 | cum_freqs 80→25 µs/step |
| 12c | NEON `sigmoid` (safe `-|x|` form via `vbslq`) | +3–5% throughput, entropy bound exact |
| 12d | Hand-tuned NEON 16-wide INT8 head matvec | +11% ST, +4–5% MT |
| 12e | Fused `time_mix` step1+step2 NEON kernels (11 passes → 2) | +0.5% ST, +3–4% MT |
| 12f | NEON `layer_norm` (3-pass NEON reduction) | −1 µs/step on forward |
| 12g | NEON 256×256 + 256×512 + 512×256 matvecs (3.2M tier) | 3.2M: +55% compress, +63% decompress (25→40 KB/s) |

## Done in Phase 12 — reverted

| item | reason |
|---|---|
| Head matvec pre-widen (simple full-column variant) | regressed compress −12% on 32K vocab; 128 KB widen scratch + 128 KB output buffer blow M1 L1. Tiled rework still possible — needed for future 32K-vocab models. |
| Chunk-skip in softmax Pass 2 (4-wide + 16-wide attempts) | sub-noise gain; NEON exp polynomial (~30 cycles/chunk) too cheap to gate on horizontal-max-then-branch (~10 cycles overhead per chunk). |
| `lto = "fat"` in `Cargo.toml` | equivalent throughput to thin LTO on this memory-bound workload, and incompatible with PGO instrumentation when revisited. |
| Larger default segment size (4096 → 8192) | monotonic ratio improvement at monotonic compress speed cost (~6% per 2×). Speed is non-negotiable per CLAUDE.md, so default kept at 4096. Empirical sweep table now in codec.rs docstring. |

## Done in Phase 12 — skipped with justification

| item | reason |
|---|---|
| **INT8 attention/FFN block projections** | `matvec_96x96_neon` already ~3.2 µs/token total across all 16 calls (~2% of budget). 4× memory savings on 9 KB matrices that already fit L1 isn't worth the per-layer-compounding ratio risk. |
| **INT8 embedding row lookup** | one row per token × 96 f32 = 384 bytes at L2 latency ≈ 50 ns total. INT8 saves ~30 ns (~0.07% throughput) at the cost of dequant compute. |
| **Top-K cum_freqs via quickselect** | finding K-th largest costs ~50–150 µs (heap or quickselect); Pass-2 polynomial cost is ~12 µs. Net loss on this hardware. |
| **rANS replacing Nayuki AC** | measured single-thread decompress (23.6 KB/s) > single-thread compress (22.5 KB/s) — AC decode is essentially free after Phase 12a binary search. rANS would save <1% throughput. |
| **PGO (profile-guided optimization)** | rustc 1.94.1 + aarch64-apple-darwin: LLVM value-profile instrumentation segfaults the instrumented binary intermittently (same binary, EXIT 0 once then 139 next; crash in `__llvm_profile_instrument_target`). One successful end-to-end run measured +5–6% multi-core. Defer to a future rustc that fixes the regression. |

---

## Open work

### Speed — compress path

#### NEON prefix sum on cum_freqs (Pass 3b)

**File:** `l3tc-rust/src/codec.rs:1450`
**Impact:** small (cum_freqs is now ~14% of total; the prefix-sum
walk is ~1/4 of that — capped at ~3% throughput)

Scalar serial loop over 16K u32 freqs with a carried u64
dependency: `cum[i+1] = cum[i] + freqs[i]`. A SIMD parallel
prefix scan via shift-and-add (Hillis–Steele) could halve the
loop, but the u32→u64 widening at the AC boundary complicates
the kernel. Complex to write correctly.

### Speed — both paths

#### Larger segment size

**Current:** 4096 bytes per segment.
**Impact:** better ratio + less per-segment overhead

Larger segments (8K–16K) mean fewer segments → less rayon
dispatch overhead, fewer session resets → less wasted context,
longer context window → better predictions → better ratio.
Tradeoff: less rayon parallelism on small files. Should
benchmark 4K vs 8K vs 16K on enwik6 to quantify.

#### Entropy-driven segment boundaries

**File:** `l3tc-rust/src/codec.rs`
**Impact:** ratio tighten + throughput via fewer segments

Complementary to the above. Close segments when predicted
entropy spikes rather than on fixed byte counts. Predictable
content (config files, log boilerplate) stays in a single
segment → less framing overhead. Hard regions get boundary
resets where they actually help. Framing cost is ~5 bytes/
segment (v4 varint headers), so this translates directly to
bpb.

#### Per-core multi-stream SIMD batching

**File:** `l3tc-rust/src/rwkv.rs`, `tensor.rs`
**Impact:** single-core throughput uplift on few-core machines

Interleave state updates for 2–4 segments on one core. Model
fits L1 (~150 KB for 200K INT8). Orthogonal to the existing
rayon cross-core parallelism. Helps on 2-4 core targets and
on serial audit/entropy-bound runs.

### Ratio — model-level

#### INT4 head

**File:** `l3tc-rust/src/tensor.rs`
**Impact:** 2× cum_freqs memory reduction; ratio TBD

Per-column INT8 scaling extends naturally to INT4. Phase 4a
did not validate INT4 accuracy — needs a diff-harness run on
the entropy bound before committing. Ratio-risky: the
quantization error at 4 bits is non-trivial on the head
because the logit range is wide. Per *Understanding INT4
Quantization* (arXiv 2301.12017), the final projection is
the most fragile layer — INT4 should target inner blocks
first, not the head.

#### Byte-level tail model for unk payloads

**File:** `l3tc-rust/src/codec.rs` (unk-extraction path)
**Impact:** potentially halve the 0.36 pp unk slice

Unk payloads are non-tokenizable bytes stored verbatim after
the Phase 4b2 extraction (~3,623 B on enwik6 = 0.36 pp of the
0.61 pp gap-to-entropy-bound). A tiny byte-level n-gram or
logistic predictor over the last few bytes could compress
these inside an AC stream. The last reachable slice of
gap-closing at 200K capacity.

### Ratio — architectural bets (multi-day undertakings)

#### Smaller vocab via tokenizer retrain

**File:** `tokenizer_balanced_32k/` (Phase 11 retrain in progress)
**Impact:** head matvec and cum_freqs both scale linearly with vocab

If the Phase 11 tokenizer retrain holds ratio at 32K → 16K (or
further), both head and cum_freqs shrink proportionally. Direct
speed win that's already in flight — measure when the retrain
finishes.

#### Ratio-preserving distillation

**File:** `scripts/distill_l3tc.py` (Phase 4e infra retained)
**Impact:** tighten actual ratio at fixed forward cost

Phase 4e asked "can a smaller student match the bigger teacher's
ratio?" and failed — speed ceiling is vocab-bound, not depth-
bound. The flip question: train the production student against
a much larger teacher's soft distributions (CE to teacher
probs, not hard tokens). Same inference cost, tighter cross-
entropy → tighter ratio. Uses existing distillation infra.
**Biggest ratio lever that doesn't cost speed.**

#### Hybrid dispatch (Phase 8)

**File:** new — per-segment dispatcher
**Impact:** rescues degenerate inputs; ratio-neutral on text

Route near-random content (already-compressed files,
encrypted, raw binary) to zstd. Gate per-segment on entropy
estimated from the first N tokens: if predicted bpb > zstd's
expected output, fall back. Already on the product roadmap.

#### Domain dispatch with specialized checkpoints

**File:** new — tiny classifier + multiple checkpoints
**Impact:** solves the Phase 11 capacity ceiling without a bigger model

Phase 11 retraining 200K on a 10 GB diverse corpus traded
enwik6 ratio for domain breadth. Lesson: 200K params cannot
be universal. Ship 3–5 specialized checkpoints (Wikipedia,
code, logs, JSON, generic), auto-detect corpus from the first
KB via a tiny classifier (logistic over byte-n-gram features),
swap at load. Each checkpoint is ~200K params; swap cost is
one `Model::load_file` call. Weeks of training, but the path
to "best ratio on everything" at 200K inference cost.

#### Two-tier cheap/full predictor

**File:** new — cheap predictor alongside RWKV `Session`
**Impact:** 2–3× throughput if ~60% of tokens are "easy"

Cheap predictor (bigram/trigram, or a tiny linear head over
the last RWKV state) handles high-confidence tokens. Full
RWKV forward pass fires only when the cheap predictor's top-K
misses the next token. Skips the forward pass entirely on
predictable content (whitespace, XML structure, log
boilerplate). Ratio cost is bounded by how often the cheap
predictor is wrong about its own confidence, tunable via
threshold. **Architecturally the biggest speed unlock left.**

---

## Implementation order (open items)

Ratio-of-impact-to-effort, ratio-neutral first.

1. **Vocab tokenizer retrain** — already in flight; measure when done
2. **Ratio-preserving distillation** — biggest ratio lever at fixed speed
3. **Two-tier predictor** — biggest speed unlock left
4. **Domain dispatch** — product-direction dependent, weeks of training
5. **Hybrid dispatch / Phase 8** — product polish, not a ratio win on target
6. **Byte-level unk model** — last reachable slice of entropy-bound gap
7. **INT4 head** — ratio-risky; only if speed still wanted after the above
8. **Prefix-sum SIMD** — small win, complex
9. **Per-core multi-stream** — niche (few-core / serial paths)

# Phase 2.5 — Aggressive speed optimizations

**Starting point (end of Phase 2):**
- 89 KB/s compress, 92 KB/s decompress on enwik6 (1 MB)
- Ratio 0.2060
- 6.9× faster than Python L3TC-200K
- Byte-identical round trip, 35 unit tests + 4 integration tests passing

**Phase 2.5 targets:**
- **Speed: ≥150 KB/s compress on enwik6** (≥10× Python, stretch 200 KB/s)
- **Ratio: maintain or improve** 0.2060 — not the focus this phase, but
  must not regress
- **Correctness: same byte-identical round trip guarantee**

---

## Why another speed phase before Phase 3

The per-token profile from Phase 2 shows where time is going (on a
single thread, no segment parallelism):

| Stage | us/token | % | Notes |
|---|---:|---:|---|
| Tokenize | ~1 | 0.4% | negligible |
| Forward pass | ~190 | 70% | |
|   — Head matvec | ~100 | ~37% | col-major AXPY, serial |
|   — 16 block matvecs | ~50 | ~18% | scalar with 4-acc unroll |
|   — Layer norms + time mix + element-wise | ~40 | ~15% | |
| cum_freqs (softmax → freq table) | ~80 | 29% | |
| AC encode | ~4 | 1% | negligible |
| **Total per-token per-core** | **~275** | | |

At 8 cores via segment parallelism, per-file throughput is roughly
`8 * 1_000_000 bytes / (275e-6 * token_count)`. With ~275,000 tokens
per 1 MB (~3.6 bytes/token) that gives ~106 KB/s ideal, ~90 KB/s
measured (about 85% parallel efficiency because of load imbalance
and serial tail).

To hit 150 KB/s we need per-token time ~165 us, or 110 us savings
per token. To hit 200 KB/s we need ~125 us/token, or 150 us savings.
The three big levers and their estimated savings:

1. **INT8 head quantization**: head ~100 us → ~35-50 us. Savings ~50-65 us.
2. **Hand-tuned NEON block matvecs**: blocks ~50 us → ~10 us. Savings ~40 us.
3. **Vectorized cum_freqs**: cum_freqs ~80 us → ~25-40 us. Savings ~40-55 us.

Any two of these stacked hit the 150 KB/s target. All three put us
close to 200 KB/s.

---

## 2.5a — Hand-tuned NEON for 96x96 block matvecs

**Why this first:** lowest risk, no weight changes needed, scoped
entirely to `src/tensor.rs`. Works on any CPU with ARM NEON (all
modern ARM and Apple Silicon).

**The target op:** row-major `(96, 96)` matvec called 16 times per
token in `Session::forward` (8 per block × 2 blocks for L3TC-200K).
Current implementation is a 4-accumulator unrolled scalar loop
which LLVM auto-vectorizes to ~3 us per call. Peak NEON throughput
for this shape should be ~0.4 us per call.

**Approach:**
1. Add a specialized `matvec_96x96_neon` function in `tensor.rs`
   using `std::arch::aarch64` intrinsics.
2. Preload x into 6 `float32x4_t` registers (6 × 4 = 24 × 4 = 96
   elements) — stays in registers for all 96 rows.
3. Inner loop: load 2 × `float32x4_t` from the current row, do 2
   FMAs against the corresponding x registers, accumulate into
   2 `float32x4_t` accumulators.
4. After the inner loop, horizontal-sum the 2 accumulators via
   `vaddvq_f32`.
5. Guard the function with `#[target_feature(enable = "neon")]`
   and call it only when `target_arch = "aarch64"`. Fall back to
   the generic scalar path otherwise.
6. Hook it into `src/rwkv.rs` for the block projections (K, V, R,
   output, ffn_k, ffn_v, ffn_r, short).

**Expected impact:**
- 96x96 matvec: 3 us → 0.4-0.6 us
- 16 calls per token: 48 us → 6-10 us
- Total forward pass savings: ~40 us/token
- End-to-end throughput improvement: ~90 → ~115-125 KB/s

**Risks:**
- Unsafe code (intrinsics require `unsafe { }`)
- Platform-specific (only helps aarch64; x86 gets no benefit)
- Must verify the round-trip still works (the SIMD operations
  should be bit-identical with the scalar path because f32 FMA
  is IEEE 754 compliant, but worth confirming)

**Success criteria:**
- All 35 unit tests still pass
- All 4 end-to-end integration tests still pass
- `iter.sh` shows ≥110 KB/s compress on enwik6
- Ratio unchanged (0.2060 ± 0.0005)

---

## 2.5b — INT8 head quantization

**Why this second:** biggest single win but highest implementation
risk. Requires modifying the checkpoint converter, the model
structure, and the head matvec. Best done after 2.5a has locked in
a faster baseline so we can measure the INT8 gain cleanly.

**The target tensor:** `head.weight` shape `(16384, 96)`, currently
stored as 1.57 M f32 (6.3 MB). Access pattern: load one column per
token, multiply by a broadcast scalar from x, accumulate into
output. Memory-bound on most CPUs because the column doesn't fit
in L1.

**Approach:**
1. Extend the Python checkpoint converter (`scripts/convert_checkpoint.py`)
   to compute an INT8 quantization of the head weight:
   - Per-column scales (96 f32 values) + INT8 data (1.57 M bytes)
   - Simple symmetric quantization: `q = round(w / scale)`, with
     `scale = max(|col|) / 127`
2. Write the quantized head to the binary format as a new tensor
   type (`dtype = 1` for INT8 with per-column scales).
3. Extend the Rust checkpoint reader to load the new tensor type.
4. Add an INT8 matvec function in `tensor.rs`:
   - Load column as 24 × `int8x16_t` SIMD registers (16 INT8
     lanes each, 24 × 16 = 384 values — wait, 16384 is too many
     for register-resident. Load in chunks of, say, 128 rows at a
     time.)
   - Widen to INT16, multiply by the broadcast x scalar (converted
     to INT16 via fixed-point), accumulate into INT32 accumulators.
   - At end of each chunk, convert INT32 → f32 and scale by the
     column's f32 scale factor, store to `out`.
5. Hook it into `Session::forward` for the head projection only.

**Quantization error analysis:**
- Symmetric INT8 quantization has ~0.4% relative error on uniform
  f32 inputs.
- The head projection's output feeds directly into softmax + freq
  table conversion. Small relative errors in logits translate to
  small relative errors in probabilities, which (through the
  arithmetic coder's log2) translate to small per-symbol coding
  overhead.
- Expected ratio impact: ~0.2-0.5 percentage points worse (e.g.
  0.2060 → 0.2080-0.2110 on enwik6).
- Acceptable for the 3× head speedup. Worst case: we offer INT8 as
  an opt-in flag and keep f32 as the default.

**Expected impact:**
- Head memory traffic: 6.3 MB → 1.6 MB per token (4× reduction)
- Head matvec: ~100 us → ~35-50 us
- End-to-end throughput improvement: ~125 → ~170-200 KB/s
  (stacked with 2.5a)

**Risks:**
- Numerical precision: the arithmetic coder is sensitive to
  probability changes. If the INT8 head shifts the argmax token,
  we lose compression on the first token of every sequence.
  Mitigation: test on enwik6 and measure ratio change.
- More complexity in the checkpoint format (new tensor type).
- INT8 kernels have more edge cases than f32 (sign handling,
  saturation, endianness).

**Success criteria:**
- Ratio regression ≤ 1 percentage point (0.2060 → ≤0.2160)
- Compress speed ≥ 150 KB/s on enwik6 (1.5× Phase 2 baseline)
- All existing tests still pass with the f32 path
- New test: INT8 round trip matches f32 round trip within epsilon

---

## 2.5c — Vectorized cum_freqs

**Why this last:** the most architecturally uncertain optimization.
It might be easy, it might be hard — depends on whether we can
find a split-precision scheme that doesn't hurt the ratio.

**The target function:** `logits_to_cum_freqs_scratch` in
`src/codec.rs`. Currently ~80 us/token. Three sub-passes:
1. Find max (~3 us, vectorizes)
2. Compute fast_exp_neg + sum (~30 us, partially vectorizes)
3. Scale + floor + clamp + cum accumulate (~45 us, scalar due to
   `f64 → u64` cast on ARM)

**Approach 1: split precision**
- Scale down the target_total from `MAX_TOTAL` (2^62) to something
  that fits in u32 precision (e.g., 2^30).
- Use an additive baseline scheme so low-probability tokens don't
  get squished to 1 (this was the bug in an earlier Phase 2 attempt).
- Specifically: each token gets `freq = 1 + floor(e * scale_32)`
  where `scale_32` is chosen so `1 + scaled_sum = CUM_TOTAL_U32`.
- Cast f32 → u32 vectorizes (4 lanes at once) on NEON via `fcvtzu`.

**Approach 2: direct logit-to-freq via bit manipulation**
- Compute `freq[i] = 2^((logit[i] - max) * scale)` using the same
  bit-cast trick as `fast_exp_neg`.
- Skip the softmax entirely.
- Needs careful normalization so the sum matches the target total.

**Approach 3: top-K truncation**
- Find the top-K logits (K=256 or 512) via partial sort.
- Compute proper softmax for top-K only.
- All other tokens get `freq = 1` uniformly.
- Cost of finding top-K: O(n log K) ≈ 120 K ops ≈ 30 us.
- Savings from skipping softmax on 16128 tokens: ~70 us.
- Net: ~40 us savings.

**Expected impact per approach:**
- Approach 1: ~45 us savings, possible 0.5-1 pp ratio cost
- Approach 2: ~50 us savings, similar ratio cost
- Approach 3: ~40 us savings, smaller ratio cost (top-K is
  information-theoretically sound)

**Best guess:** Approach 3 (top-K) because its ratio impact is
bounded and well-understood. Start there.

**Success criteria:**
- cum_freqs drops from ~80 us to ≤40 us
- Ratio regression ≤ 0.5 pp
- End-to-end throughput ≥ 160 KB/s on enwik6 (stacked with 2.5a+b)

---

## Execution order

1. **2.5a NEON block matvecs** (lowest risk, quickest win)
   - Implement `matvec_96x96_neon`
   - Integrate into `Session::forward`
   - Measure, commit
2. **2.5b INT8 head quantization** (biggest win, highest risk)
   - Extend converter
   - Extend Rust reader
   - Implement INT8 matvec
   - Integrate, measure, commit
3. **2.5c Vectorized cum_freqs** (if still needed for 150 KB/s target)
   - Start with Approach 3 (top-K)
   - Measure ratio impact carefully
   - Commit only if net positive

After each substep: run `iter.sh` for a quick round-trip +
throughput check, run full `cargo test --release`, run
`end_to_end` integration tests against the full enwik6. If any
step regresses correctness or drops ratio below 0.212, revert.

---

## Non-goals

- Multi-threading within a single forward pass (already tried,
  D15: segment-level parallelism is the correct level).
- Replacing the arithmetic coder (it's already 1% of per-token
  cost, not worth touching).
- Retraining the model (Phase 4 concern).
- Changing the file format (Phase 3 concern).
- Changing the segment size default again (Phase 2 tuning was
  already thorough).

## Success criteria (Phase 2.5 exit)

Phase 2.5 is done when:
- Compress speed ≥ 150 KB/s on enwik6 (stretch: 200 KB/s)
- Ratio ≤ 0.215 on enwik6 (≤ 0.5 pp regression)
- All 35 unit tests pass
- All 4 end-to-end integration tests pass
- Full 1 MB enwik6 round-trip is byte-identical
- Final commit documents which of 2.5a/b/c actually shipped and
  why the others (if any) were deferred

After Phase 2.5 we move to Phase 3 (file format + stream API +
Silesia + full enwik8/9).

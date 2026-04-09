# Phase 4c — CPU speed polish  ⚠️ LARGELY DONE (+7%/+8%)

**Final state as of the last commit:** enwik6 compress
**126.8 KB/s**, decompress **128.2 KB/s**, ratio 0.1699
(unchanged). Cumulative gain over Phase 4b2's 119/119 baseline
is roughly 7% compress and 8% decompress — below the original
170-200 KB/s target but above the 99 KB/s speed floor.

**Honest takeaway.** Apple Silicon + LLVM's aarch64 autovectorizer
is aggressive enough that most hand-written NEON on short
(≤96-element) f32 vectors is a wash. The clear wins came from
pointing the code at existing NEON kernels that the compiler
couldn't reach on its own (4c2: FFN matvecs to NEON 96×96) and
from a fused vectorized quantize pass (4c3). The "replace libm
exp with a polynomial" idea (4c1) is numerically correct but
didn't move the needle because Apple's libm is already fast.

**Items backburnered for later:**

- **4c5 INT4 head quantization** — the only remaining CPU speed
  item with a plausible 5-10% win (halving head memory traffic
  on the biggest matmul). Not implemented in this phase because
  (a) the engineering cost is 2-3 days with real ratio risk and
  (b) the "different model" path (port L3TC-3.2M, then decide on
  distillation) now looks like a higher-impact direction. If
  that path stalls or we decide to come back for CPU polish,
  4c5 is the next thing to try.



**Goal:** extend our lead in single-stream CPU neural compression
by reclaiming the per-token hot-loop time that Phase 4b's ratio
work didn't touch. Target: enwik6 compress **119 KB/s → 170-200
KB/s** with ratio unchanged (within 0.0005 of 0.1699).

**Starting point (end of Phase 4b2, enwik6):**
- 119 KB/s compress / 119 KB/s decompress
- ratio 0.1699 actual coded bytes, entropy bound 0.1632
- forward pass bit-identical to Python L3TC (4a)
- gap to entropy bound 0.61 pp (structurally irreducible at
  this architecture + corpus)
- 34 unit tests passing
- aarch64-apple-darwin native build with NEON 96×96 matvec
  (Phase 2.5a) and INT8 head (Phase 2.5b)

**Non-goals (explicit):**
- GPU inference (separate track — see STORAGE_SERVICE_VISION.md)
- Ratio improvements (Phase 4b maxed that at this architecture;
  Phase 5 / 8 / 11 are where more ratio lives)
- Cross-platform numeric contract (Phase 7)
- New classical / neural compressor competitors in the bench
  (not a speed change)

## Per-token profile (rough, pre-4c)

Not precisely re-measured since Phase 2.5 + 4b changes; this is
the mental model that drives the priority order below.

| stage | us/tok | % | notes |
|---|---:|---:|---|
| cum_freqs (softmax → freq table) | ~60 | ~27% | dominated by libm `f32::exp` since 4a removed fast_exp_neg |
| Head matvec (INT8 16384×96 col-major AXPY) | ~40 | ~18% | already INT8 in 2.5b |
| Layer norms ×4 + element-wise time/channel mix | ~40 | ~18% | scalar f32, NOT NEON'd |
| FFN K (96→384) + FFN V (384→96) matvecs | ~35 | ~15% | plain matvec, NOT NEON'd |
| 12 block matvecs (K/V/R/out + FFN R + short, NEON 96×96) | ~10 | ~5% | already optimized in 2.5a |
| Time-mix WKV element-wise math (exp/div/etc) | ~15 | ~7% | scalar |
| Tokenizer (amortized, parallel in 4b2) | ~5 | ~2% | SPM is C++ |
| AC encode | ~4 | ~2% | effectively free |
| Scratch / memory / everything else | ~15 | ~7% | background |
| **total per token** | **~225 us** | 100% | |

## The five items

### 4c1 — NEON `exp_f32x4` in cum_freqs

**What:** replace the scalar `(logits[i] - max).exp()` loop with a
hand-rolled NEON `float32x4_t` exp built from the standard
`exp(x) = 2^(x * log2e) = 2^k * P(r)` decomposition. Use a
degree-6 minimax polynomial for 2^r on [0, 1] (max relative
error < 5e-7, comfortably tighter than the old `fast_exp_neg`
polynomial's 1% tolerance that we deleted in Phase 4a).

**Why it's the biggest single win:** the softmax exp loop runs
16384 times per token after the max-subtract. libm `expf` is
~25-40 ns per call on aarch64. We spend ~60 us per token here.
Vectorizing 4-wide with an in-register polynomial drops that
to ~12-18 us per token — 3-4× faster on this stage alone.

**Regression gate:** the entropy bound on enwik6 at segment
4096 must stay at 0.1632 ± 0.0001. If the polynomial's
accuracy shifts logit probabilities enough to change the
entropy bound by more than 1e-4, reject and go to degree-7 or
revert. Round trip must stay byte-identical.

**Expected gain:** ~30-45 us/token saved → ~+15-20% end-to-end
throughput. **Target: 119 → ~140 KB/s.**

### 4c2 — Switch FFN matvecs to the existing NEON 96×96 kernel

**What happened:** the original plan called this "fused K/V/R
288×96 matvec" in time_mix — but investigating the code for
4c3 surfaced a bigger and simpler finding: the FFN key and
value matvecs in `channel_mix` were still using the scalar
`tensor::matvec` fallback, not the NEON `matvec_96x96` kernel
from Phase 2.5a. Phase 2.5a converted the attention K/V/R/out
matvecs but missed the two FFN matvecs of the same shape. A
two-line fix.

**Why it's 96×96 and not 96×384:** the original 4c3 plan
assumed `intermediate_size > hidden_size` (the standard
transformer FFN expansion ratio). But L3TC-200K's config sets
`intermediate_size == hidden_size == 96`, so the FFN key and
value are *also* 96×96. No new kernel needed — just call the
existing one. Bigger L3TC variants (800K / 3.2M / 12M) have
different shapes and would need proper expansion-ratio
kernels, but we don't ship those today.

**Result on enwik6:**
  compress: 119 -> 124.3 KB/s  (+4% over 4b2)
  decompress: 119 -> 125.6 KB/s  (+5%)
  ratio: 0.1699 -> 0.1699  (169869 -> 169876, +7 bytes from
    4c1's NEON exp slightly shifting a few freq-quantization
    boundaries; well within the ±0.0005 gate)

### ~~4c3 — NEON FFN matvecs (96×384 and 384×96)~~  OBSOLETE

Originally scoped assuming `intermediate_size > hidden_size`.
For L3TC-200K that's false: FFN matvecs are 96×96 and already
covered by 4c2 above. This item becomes relevant only if we
ever target the bigger L3TC variants, at which point we'd
write the proper expansion-ratio NEON kernels. For the 200K
model ship path, 4c3 is a no-op.

### 4c4 — NEON layer norm + fused element-wise time mix

**What:**
- NEON layer norm: f32x4 reduction for mean, f32x4 reduction for
  variance, f32x4 element-wise normalize+scale+shift. Replaces
  the scalar `tensor::layer_norm` with `layer_norm_neon` when
  `target_arch = "aarch64"`.
- NEON `time_mix` (`x * mix + state * (1 - mix)`) and related
  element-wise ops (`add_inplace`, `sigmoid_inplace`,
  `relu_inplace`, `square_inplace`). All trivially f32x4
  vectorizable.

**Why:** four layer norms per token + a dozen element-wise
passes = ~40 us today. NEON'd, ~15-20 us. Trivial engineering,
each op is 10-20 lines.

**Regression gate:** ratio unchanged, round trip byte-identical,
numerical deltas within f32 ULP on synthetic tests.

**Expected gain:** ~15-20 us/token → ~+7% throughput. **Target:
160 → ~170-175 KB/s.**

### 4c5 — INT4 head quantization (riskiest, last)

**What:** replace the INT8 per-column head quantization (Phase
2.5b) with INT4 per-group (group size 32 or 64) quantization.
Halves memory traffic on the head from ~1.57 MB to ~780 KB.

Storage layout: `Vec<u8>` holding two INT4 nibbles per byte,
plus a per-group `Vec<f32>` of scales. Matvec kernel unpacks
nibbles to i8, widens to i32 accumulator, then dequantizes by
the group scale.

**Why:** head matvec is memory-bound at the current throughput.
Halving the bytes touched per token gets close to halving the
head's contribution to per-token latency. ~40 us → ~20-25 us.

**Risk:** Phase 4b2 closed the entropy bound gap to 0.61 pp.
INT4 head will add some quantization error back into the
logits. The question is whether that error is smaller than
the ~5e-4 wiggle room the ratio has before we'd notice.

**Regression gate (strict):** the entropy bound on enwik6 must
stay at 0.1632 ± 0.0005. The actual coded ratio must stay at
0.1699 ± 0.0005. If either drifts more, revert and accept the
INT8 head as permanent.

**Expected gain:** ~15-20 us/token → ~+8-12% throughput.
**Target: 175 → ~190-200 KB/s.**

## Execution order and rationale

1. **4c1 first** because it's the biggest isolated win, has a
   clean regression gate (entropy bound), and lives entirely
   in cum_freqs without touching the forward pass.
2. **4c2** next because it's a mechanical refactor with minimal
   risk and unlocks a measurable win.
3. **4c3** after 4c2 because the FFN matvec rewrite is the
   most engineering-heavy but has the clearest performance
   ceiling (copy-paste from NEON 96×96).
4. **4c4** before 4c5 because it's risk-free and its win is
   additive — we want it locked in before we risk the riskier
   INT4 work.
5. **4c5 last** because it's the only item that can threaten
   the ratio. If we need to revert, we only lose this item's
   win, not the whole phase.

Re-measure between every item with `./iter.sh`, `l3tc audit`,
and `l3tc entropy-bound`. Commit each item as its own commit
so `git bisect` can isolate a regression.

## Success criteria

Phase 4c is done when:

- enwik6 compress ≥ **170 KB/s** (stretch: ≥ 185 KB/s)
- enwik6 ratio ≤ 0.1705 (within 0.0005 of current 0.1699)
- enwik6 entropy bound stays at 0.1632 ± 0.0005
- enwik8 compress ≥ **160 KB/s** with ratio ≤ 0.180
- All 34+ unit tests pass
- Byte-identical round trip on enwik6 and enwik8
- `docs/phase_4c_findings.md` documents per-item deltas with
  before/after audit outputs

## What happens if we miss the target

If we land at, say, 150 KB/s instead of 170+:

- Identify which of 4c1-5 underperformed the estimate.
- Don't chase the last 20 KB/s by sacrificing ratio. The speed
  floor from CLAUDE.md is 99 KB/s; 150 KB/s is well above it
  and still a 26% improvement over Phase 4b2's 119.
- Defer any remaining items to Phase 4d (speculation territory:
  prefetch tuning, AC decode latency, inter-segment pipelining).
- Move on to Phase 5 (v7 architecture) or Phase 11 (broader
  corpus) — both are higher impact than another round of
  micro-optimizations.

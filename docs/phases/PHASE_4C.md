# Phase 4c — CPU speed polish  ✅ LARGELY DONE (+7%/+8%)

Enwik6 compress **126.8 KB/s**, decompress **128.2 KB/s**, ratio 0.1699 (unchanged). Cumulative gain over Phase 4b2's 119/119 baseline is roughly 7% compress and 8% decompress.

Apple Silicon + LLVM's aarch64 autovectorizer is aggressive enough that most hand-written NEON on short (<=96-element) f32 vectors is a wash. Clear wins came from pointing code at existing NEON kernels the compiler couldn't reach on its own.

## What shipped

- **4c1** — NEON `exp_f32x4` in cum_freqs. Replaced scalar `f32::exp` loop with hand-rolled NEON polynomial. Numerically correct but Apple's libm is already fast, so minimal throughput gain.
- **4c2** — FFN matvecs switched to existing NEON 96x96 kernel (2-line fix; Phase 2.5a had missed these). Compress 119 -> 124.3 KB/s (+4%), decompress 119 -> 125.6 KB/s (+5%).
- **4c3** — Obsolete (FFN matvecs are 96x96 for 200K, not 96x384 as assumed).
- **4c4** — Fused vectorized quantize pass, contributing the remaining gain to 126.8/128.2.

## Backburnered

- **4c5 INT4 head quantization** — plausible 5-10% win but 2-3 days engineering with real ratio risk. The "different model" path (Phase 4d/4e) looked higher impact.

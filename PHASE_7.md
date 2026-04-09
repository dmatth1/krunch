# Phase 7 — Cross-platform numeric contract (byte-identical forward pass)

**Status:** back-burner. Prerequisite for Phase 6 (multi-platform
release builds) but not urgent until we actually ship to non-Apple
platforms or start a service pilot that needs multi-region
replication.

**The problem.** Arithmetic coding diverges on a single-bit float
drift. Our current forward pass uses libm `f32::exp`, f32
layer-norm, f32 time-mix, and scalar f32 accumulation — none of
which is guaranteed to produce bit-identical results across:

- `aarch64-apple-darwin` (current default)
- `aarch64-unknown-linux-gnu`
- `x86_64-apple-darwin`
- `x86_64-unknown-linux-gnu`
- Windows, iOS, Android if we ever ship there

A 1-ULP difference in any logit between two targets means the
cum_freqs table differs, which means the arithmetic coder's
`(low, high)` bracket differs, which means the output bytes
differ from that token forward. The file is non-portable.

This is the single largest blocker between "research runtime on
one machine" and "production compressor with a stable file
format". In [CLAUDE.md](CLAUDE.md) terms: goal #1 is meaningless
if the bytes aren't portable, because the entropy bound is a
property of a specific forward pass.

## Approaches

**Option A — Fixed-point / integer forward pass (preferred).**
Convert the forward pass to integer math with carefully chosen
scales. Bit-exact by construction. Related prior art in
`vendor/L3TC/models/RWKV_V4/ptq/` — the L3TC training code
includes post-training quantization infrastructure. We'd extend
it from "INT8 for the head" (already done in 2.5b) to "INT8 (or
INT16) everywhere". Tradeoff: ratio may degrade by 0.5-1 pp from
compounded quantization error across two layers. Validation
required before shipping.

**Option B — Locked-down f32 with portable libm and deterministic
intrinsics.** Replace `f32::exp` with a hand-rolled polynomial
that produces the same bit pattern on every target. Compile with
`-ffp-contract=off` to disable FMA fusion. Ship explicit scalar
fallbacks for non-aarch64 platforms that byte-match the NEON
intrinsic path. Smaller ratio impact than Option A but much more
fragile: every LLVM version, every libm change, every compiler
flag can break it.

**Option C — Ship the service only (punt on client distribution).**
If the long-term pitch is a managed service (see
`STORAGE_SERVICE_VISION.md`), cross-platform clients don't
matter — everything runs on our hardware. Phase 7 becomes "pin
our server fleet to one instance type" which is hours of Docker
work, not weeks of numerics engineering.

**Recommendation:** A if we ever want a standalone CLI; C if the
product becomes the managed service.

## Validation

The test is simple to describe and somewhat hard to set up:

1. Take a 1 MB enwik6 sample.
2. Compress it on every supported target via CI runners.
3. `shasum` the compressed outputs.
4. All shasums must match.

If a shasum differs, diff the dumped per-token logits (Phase 4a
`dump-logits` subcommand already does this) between the two
targets to find the first divergence. Fix the op responsible.
Iterate until convergence.

Needs a CI matrix with real runners for each target — cross-
compilation alone isn't sufficient because we want to validate
that the produced binary actually runs and produces matching
output on the target hardware.

## Success criteria

- Compressed bytes for the same input are bit-identical across
  `{aarch64-apple-darwin, aarch64-linux-gnu, x86_64-darwin,
   x86_64-linux-gnu}`.
- enwik6 ratio stays within 0.5 pp of the current 0.1699
  (INT8 throughout may cost us some of the entropy gap we just
  closed; measure and decide).
- Decompress speed regression ≤ 20% from the current ~119 KB/s.
- All 34+ unit tests still pass; 4 end-to-end tests still pass.
- A new cross-target CI job runs on every PR and fails on byte
  mismatch.

## Non-goals

- Windows / iOS / Android support (Phase 10 maybe).
- x86_64 SIMD intrinsics (scalar fallback is fine for correctness;
  speed optimization is a separate question).
- Bit-exact match to the Python reference implementation — we
  already verified that for the algorithmic forward pass in
  Phase 4a; Phase 7 only cares about Rust-to-Rust portability.

# Phase 7 — Cross-platform numeric contract (byte-identical forward pass)

**Status:** back-burner. Prerequisite for Phase 6 (multi-platform release builds). Not urgent until we ship to non-Apple platforms or start a service pilot needing multi-region replication.

**The problem.** Arithmetic coding diverges on a single-bit float drift. Our forward pass uses libm `f32::exp`, f32 layer-norm, f32 time-mix, and scalar f32 accumulation -- none of which is guaranteed bit-identical across `{aarch64-apple-darwin, aarch64-linux-gnu, x86_64-darwin, x86_64-linux-gnu}`. A 1-ULP difference in any logit means the cum_freqs table differs, the AC brackets diverge, and output bytes differ from that token forward. The file is non-portable.

This is the single largest blocker between "research runtime on one machine" and "production compressor with a stable file format".

## Approaches

- **Option A — Fixed-point / integer forward pass (preferred).** Convert to integer math with carefully chosen scales. Bit-exact by construction. Prior art in `vendor/L3TC/models/RWKV_V4/ptq/`. Extends our INT8 head (2.5b) to INT8/INT16 everywhere. Risk: 0.5-1 pp ratio degradation from compounded quantization error.
- **Option B — Locked-down f32 with portable libm.** Hand-rolled polynomial exp, `-ffp-contract=off`, explicit scalar fallbacks that byte-match NEON. Fragile: every LLVM/libm change can break it.
- **Option C — Service-only (punt on client distribution).** If the product is a managed service, pin to one instance type. Hours of Docker work, not weeks of numerics.

**Recommendation:** A if we want a standalone CLI shipping to multiple platforms; C if the product is the managed service.

## Validation

1. Compress 1 MB enwik6 on every target via CI runners.
2. `shasum` compressed outputs; all must match.
3. On mismatch, diff per-token logits via Phase 4a's `dump-logits` to find the first divergent op.

## Success criteria

- Compressed bytes bit-identical across all four target triples
- enwik6 ratio within 0.5 pp of current 0.1699
- Decompress speed regression <= 20%
- Cross-target CI job on every PR, fails on byte mismatch

# Phase 5 — RWKV-v7 architecture upgrade

**Status:** not started. On the roadmap but not the next priority.

**Goal:** replace the RWKV-v4-with-HiRA forward pass with RWKV-v7 at the same ~200K parameter budget, retrained from scratch on enwik8. Target a measurable ratio improvement over the current v4-based entropy bound (0.1632 at segment 4096) without giving up speed.

## Why v7

Phase 4 proved the ratio lives in the model, not the codec. Our AC body is within 3 bytes of the entropy bound -- we can't squeeze more ratio from the coder. RWKV-v7 is ~3 years newer than v4, with known wins on language-modeling benchmarks at comparable parameter counts. Still linear-time / constant-memory per token, so the speed profile doesn't degrade.

## What the upgrade involves

The forward pass in `src/rwkv.rs` is essentially rewritten:
- Different time-mix semantics (data-dependent decays instead of learned constants)
- New channel mixing (possibly squared-relu or GLU variants)
- New state layout per layer
- No HiRA (v7 has its own parameterization)
- New tensor ops may be needed in `src/tensor.rs`
- Checkpoint converter needs a v7 path

Phase 4a's diff harness (`l3tc dump-logits`, `scripts/diff_logits.py`) carries over directly for validating the v7 Rust forward pass against a Python v7 reference.

## Two-step approach

1. **5a — Train RWKV-v7-200K on enwik8.** If entropy bound isn't at least 0.5 pp better than v4's 0.1632, stop the phase. Compute estimate: a few hours on a single A10G/L4 (~$20-50).
2. **5b — Port the v7 forward pass to Rust.** Only if 5a shows ratio improvement. Validate via dump-logits diff harness; commit only if ratio improves without regressing speed >15%.

## Relationship to the shipping track

This is an apples-to-apples architecture comparison on enwik8 -- same data, same parameter count, different forward pass. Broader corpora are Phase 11; specialist dispatch is Phase 8. Those compose cleanly: train specialists on v7 if v7 wins.

For universal text compression, v7's improved predictions benefit all text types, not just enwik. Any ratio gain here propagates to every specialist model trained later.

## Success criteria

- RWKV-v7-200K entropy bound <= 0.158 on enwik6 (>= 0.5 pp better than v4's 0.1632)
- Rust v7 forward pass bit-identical to Python v7 (max L_inf < 1e-4)
- enwik6 actual coded ratio <= 0.165
- Compress speed >= 99 KB/s on enwik6
- All unit tests pass; new tests for v7 forward pass

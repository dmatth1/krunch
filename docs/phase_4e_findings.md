# Phase 4e findings — distillation for compression speed

**TL;DR:** The 1-layer 96-hidden student experiment runs cleanly
end-to-end (teacher dump → distillation training on MPS →
checkpoint conversion → Rust runtime compress/verify) but
**fails both decision criteria**. Result: 0.2871 ratio at 146
KB/s on enwik6 — worse than bzip2 on ratio, only 1.12× faster
than the shipped 200K default. **Phase 4e is closed; the 200K
model stays the default tier and the 3.2M stays the opt-in
high-ratio tier.** The speed lever at fixed vocab × hidden is
structurally too small to reach the "≥260 KB/s at ≤0.195 ratio"
target that motivated this phase.

## What we measured

| tier | ratio | entropy bound | compress KB/s | decompress KB/s | params |
|---|---:|---:|---:|---:|---:|
| 200K default (Phase 4c) | 0.1699 | 0.1637 | 131 | 132 | ~200K |
| 3.2M opt-in (Phase 4d) | 0.1337 | 0.1275 | 25.95 | 23.43 | 3.2M |
| **4e3 1L96H student** | **0.2871** | **0.2808** | **146** | **150** | 3.66M |
| Phase 4e target | ≤0.195 | — | ≥260 | — | — |

Note that the "1-layer 96-hidden" student has 3.66M parameters,
not the ~100K quoted in PHASE_4E.md. That estimate was wrong:
at hidden 96 × vocab 16384, the embedding table (1.57M) + head
(1.57M) dominate the param count regardless of depth. The
L3TC-200K's "200K" label counts non-embedding parameters only;
the shipped .bin is ~10 MB and ours is ~13 MB.

## Training setup

- **Teacher:** L3TC-3.2M (`checkpoints/l3tc_3m2.bin`)
- **Teacher dump corpus:** first 5 MB of enwik8
  (`bench/corpora/enwik8_5mb`)
- **Teacher dump:** `l3tc dump-teacher`, top-K=64,
  segment-bytes=2048 → 748 MB output, 1.44M prediction steps,
  3m 11s wall (25.45 KB/s parallel)
- **Student architecture:** 1 layer, 96 hidden, 96 intermediate,
  rwkv_rank=4, random init (no warm start)
- **Training loop:** `scripts/distill_l3tc.py` on MPS (Apple
  Silicon), pure-PyTorch WKV replacing the CUDA-only kernel,
  fp32
- **Loss:** Hinton distillation
  `(1-α) CE + α T² KL(teacher || student)`, α=0.5, T=1.5
- **Optimizer:** Adam, lr=2e-4, betas=(0.9, 0.99), grad-clip 1.0
- **Schedule:** 2 epochs over 2442 segments, batch size 1
- **Wall time:** ~27 minutes on MPS

## Training trajectory

| stage | loss | CE | KL |
|---|---:|---:|---:|
| epoch 0 seg 40 | 7.97 | 9.72 | 6.22 |
| epoch 0 seg 500 | ~6.4 | ~7.8 | ~5.0 |
| epoch 0 end | ~5.8 | ~7.2 | ~4.4 |
| epoch 1 seg 400 | 4.43 | 5.98 | 2.87 |
| epoch 1 end | **4.35** | **5.91** | **2.79** |

Loss decreased monotonically and plateaued mid-epoch-1. More
epochs would shave a bit more off but epoch 1's flattening
suggests we're near a local minimum given the architecture,
lr, and data. The remaining gap from the teacher isn't a
"needs more training" problem — the entropy bound of 0.2808
on enwik6 is what this architecture can do.

## Why the speed lever was smaller than predicted

PHASE_4E.md hypothesized a ~2× speedup for halving the layer
count. The Rust runtime profile on the shipped 200K model
(Phase 4c):

| stage | 200K (2L) | 1L96H student | change |
|---|---:|---:|---:|
| forward pass (total) | 154 µs | 135 µs | −12% |
| cum_freqs | 39 µs | 36 µs | −8% |
| AC encode | 0.1 µs | 0.1 µs | — |

Per-token end-to-end speed is ~1.14× at best. cum_freqs and
the head matvec both scale with **vocab (16384)**, not with
layer count, so halving layers only touches the block-compute
chunk — which was already a minority of the per-token cost
after the Phase 4c NEON polish. The architecture-level
speedup this phase relied on doesn't exist at the current
vocab × hidden ratio.

The corollary: any future "smaller student for speed"
experiment needs to also shrink **vocab or hidden** (or do
both), since those are the terms that actually dominate the
per-token cost. A 1-layer student without that change hits a
structural speed ceiling of ~1.15×.

## Why the ratio ceiling is so high

The student's entropy bound on enwik6 is 0.2808 vs the 200K
teacher's 0.1637 (Phase 4a). That's 71% worse — a large gap
that the distillation signal couldn't close in 2 epochs over
5 MB. Two things contribute:

1. **Capacity.** A 1-layer RWKV-v4-HiRA at 96 hidden has
   meaningfully less capacity than 2 layers, even before
   counting interactions across depth. The 200K teacher had
   more room and was trained on orders of magnitude more
   data.
2. **Training budget.** 5 MB × 2 epochs = 10 MB of effective
   text, vs the published L3TC-200K which trained on the full
   enwik8 for 20+ epochs. Distillation helps with sample
   efficiency but it doesn't invent capacity or data that
   isn't there.

These are separable — we could rerun with 10 MB × 5 epochs and
the ratio would probably improve — but the *speed* result
above is what kills the phase regardless. Even a
better-trained 1L96H student couldn't clear the 260 KB/s bar.

## Artifacts

- `bench/corpora/enwik8_5mb` — 5 MB slice of enwik8 used as
  training corpus (not committed, regenerable via
  `head -c 5000000 bench/corpora/enwik8 > bench/corpora/enwik8_5mb`)
- `/tmp/teacher_3m2_enwik8_5mb.bin` — 748 MB teacher dump
  (not committed, regenerable via `l3tc dump-teacher`)
- `l3tc-rust/checkpoints/l3tc_1l96h_distilled.pth` — trained
  student (gitignored)
- `l3tc-rust/checkpoints/l3tc_1l96h_distilled.epoch0.pth`,
  `.epoch1.pth` — per-epoch checkpoints (gitignored)
- `l3tc-rust/checkpoints/l3tc_1l96h.bin` — converted Rust
  binary (gitignored)

## Verdict and follow-up

**Phase 4e is closed as failed.** The PHASE_4E.md decision
criteria were:

- ratio ≤ 0.195 at ≥ 260 KB/s → ship as fast tier ❌
- ratio 0.195–0.22 → keep experimenting ❌
- **ratio > 0.22 → architecture too small; document and move on** ✅

We're in the third bucket at 0.2871. The honest follow-up is
*not* to try a bigger student or longer training; it's to
recognize that **the speed bottleneck on this architecture at
vocab 16384 is no longer layer-depth.** To get a meaningful
speedup (say ≥1.5×) without rewriting the compressor, we would
need to either:

1. **Shrink the head matvec** — smaller vocab (would need a
   new tokenizer) or factorized head (complicates the codec)
2. **Replace cum_freqs with something sub-linear in vocab** —
   hierarchical softmax, top-K writing
3. **Ship the existing Phase 4c 200K as the default** and
   invest engineering effort elsewhere

The project moves to option 3. Phase 5 (RWKV-v7 upgrade),
Phase 6/7 (release builds + numeric determinism), and
Phase 9/10 (production hardening + distribution) are higher
EV than further distillation work.

## Retained infrastructure

Even though the phase didn't ship a new tier, several pieces
of infrastructure are worth keeping for future runs:

- `l3tc dump-teacher` CLI subcommand, rayon-parallelized,
  v2 format with self-contained input/target token IDs
- `scripts/distill_l3tc.py` with configurable student shape,
  pure-PyTorch WKV for MPS/CPU, epoch checkpoints
- The vendor-side one-line patch for `n_layer == 1` in
  `rwkv_tc_hira_train.py` (would re-apply trivially on any
  future re-clone)

These stay in the repo so any future distillation experiment
(say against a new architecture in Phase 5) can start from a
working pipeline instead of re-doing the plumbing.

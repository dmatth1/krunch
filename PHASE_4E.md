# Phase 4e — Distillation for compression speed

**Status:** Closed (failed). Phase 4d delivered L3TC-3.2M as an
opt-in "max ratio" tier (0.1337 ratio, 25.95 KB/s on enwik6).
Phase 4e was the speed-oriented follow-on: **train a smaller
student architecture via distillation from the 3.2M teacher so
we ship a faster default while keeping ratio acceptable**.
The 4e3 experiment ran end-to-end cleanly but missed both
targets (0.2871 ratio at 1.12× speedup vs the ≤0.195 / ≥2×
bar). See [`docs/phase_4e_findings.md`](docs/phase_4e_findings.md)
for the full writeup. The 200K stays as default, 3.2M as
opt-in, and the project moves on to Phase 5/6/7 tooling work.

**Goal:** produce a new default checkpoint that compresses
**≥ 2× faster than L3TC-200K** (≥ 260 KB/s on enwik6) with a
ratio penalty of **≤ 2.5 pp** (≤ 0.195 on enwik6). If we can
hit those numbers, the default tier moves from
"131 KB/s at 0.170" to something like "260+ KB/s at 0.195" —
a 2× speed win for a modest ratio trade. If the smaller student
can't reach a ratio anywhere near 0.195, we fall back to the
shipped 200K and learn that 200K is already near the capacity
floor for this task.

**Starting reference points:**

| tier | actual ratio | compress KB/s | params |
|---|---:|---:|---:|
| l3tc-rust 200K (default) | 0.1699 | 131 | 200K |
| l3tc-rust 3.2M (opt-in) | 0.1337 | 25.95 | 3.2M |
| **Phase 4e target** | **≤ 0.195** | **≥ 260** | **≤ 100K** |

---

## Why distillation, and what it actually does for us

The key insight, corrected from my earlier framing in this doc:

**Speed comes entirely from the architecture.** A smaller
architecture (fewer layers, smaller hidden, or both) does fewer
FLOPs per token and therefore runs faster. This is a compute
fact, independent of training quality. Halving the number of
layers ≈ halves per-token compute ≈ 2× faster inference. Full
stop.

**Ratio comes from training quality — within the architecture's
capacity ceiling.** Every architecture has a floor it can't
beat regardless of training (set by param count and model
shape) and a ceiling it hits if undertrained (essentially
infinity — a random model outputs uniform). Training moves the
model from the ceiling toward the floor.

**Distillation's role** is to make training more sample-efficient
so a smaller student can reach its floor with less data and less
compute than training from scratch would require. The teacher's
soft probability distributions carry more information per
example than hard ground-truth labels — they tell the student
not just "the next token was X" but "X was 40% likely, Y was
30%, Z was 15%, the rest were ~0". That richer signal lets the
student learn the shape of the probability landscape faster.

**The Phase 4e thesis in one sentence:** a 1-layer
96-hidden student (half the compute of the shipped 200K)
distilled from L3TC-3.2M can plausibly reach a ratio close
enough to 200K's 0.1699 that the 2× speedup is a worthwhile
default.

**The risk:** the student architecture's capacity floor might
be significantly worse than 0.1699 no matter how well we train
it. That's the empirical question Phase 4e answers.

## cmix as the existence proof

Why we have *any* faith that a smaller architecture can approach
200K's ratio: **cmix uses essentially zero neural network and
gets 0.117 on enwik9**. Its whole "model" is dozens of n-gram
tables + a tiny neural blender (~1 KB of weights). It runs at
~1-2 KB/s because it's serial and unoptimized, but the ratio it
achieves proves the text-compression task has significant
headroom at small model sizes. The capacity requirement is not
"you need 200K params to hit 0.17" — it's "we don't yet know the
smallest architecture that can hit 0.17 with good training".

Distillation is the shortest path to finding out.

---

## Progress so far

### 4e1 — Teacher dump infrastructure ✅ DONE

Shipped in commit `4aac554` and extended in commit `1e2e412`:

- `l3tc dump-teacher` CLI subcommand that runs a model over an
  input file and writes per-step top-K softmax distributions
  to a binary file
- `codec::dump_teacher` library function, parallelized across
  segments via rayon (same parallelism as compress)
- Top-K selection via `select_nth_unstable_by` + partial sort,
  O(N + K log K) per step
- File format v2: self-contained (includes per-segment step
  counts + per-step input + target token ids) so the Python
  consumer doesn't need to re-tokenize the corpus

Measured throughput: **22 KB/s for L3TC-3.2M on enwik6**.
Full enwik6 (1 MB) dumps in 44 seconds → ~147 MB output at
top_k=64. enwik8 extrapolation: ~75 min parallel.

### 4e2 — Training pipeline ✅ INFRASTRUCTURE DONE

Shipped in commit `1e2e412`:

- `scripts/distill_l3tc.py`: PyTorch training loop
- Uses L3TC's `RWKV_TC_HIRA` training model class from
  `vendor/L3TC/models/RWKV_V4/rwkv_tc_hira_train.py` (the one
  with HiRA branches as separate Linear layers)
- Monkey-patches `RUN_CUDA` with a pure-PyTorch CPU/MPS WKV
  implementation (~30 lines, autograd-friendly), replacing
  L3TC's CUDA-only WKV kernel
- Forces `RWKV_FLOAT_MODE=fp32` at import time
- Standard Hinton distillation loss: `(1-α) CE + α T² KL`
- Loads teacher dump v2, reads input/target tokens directly
  from the dump (tokenization-free Python side)

Validated on MPS: load → train → save → convert → compress
→ round-trip OK. Throughput ~1 segment/sec at batch=1 on MPS
= ~8 min per epoch on enwik6.

### 4e2a — Same-shape distillation (SANITY CHECK, NOT THE GOAL)

Two runs on enwik6 with the 200K-shape student:

| run | α | T | lr | max_seg | epochs | result ratio |
|---|---:|---:|---:|---:|---:|---:|
| original 200K | — | — | — | — | — | 0.1699 (baseline) |
| 4e2a-1 | 0.7 | 2.0 | 5e-5 | 50 | 3 | 0.2287 (overfit) |
| 4e2a-2 | 0.3 | 1.5 | 1e-5 | 489 | 1 | 0.1767 (regression) |

Both runs regressed the ratio. The retuned 4e2a-2 was stable
(CE flat ~3.58 across the epoch instead of rising), but
fine-tuning the 200K checkpoint on 1 MB of enwik6 is a
fundamentally weird experiment: we're training on the test set
with less data and fewer epochs than the original teacher got,
and asking it to generalize better. It can't.

**Critical finding: the distilled model's *speed* is identical
to the original 200K** (~123-131 KB/s compress, within noise).
That's because the student inherits the 200K architecture
exactly — same 2 layers, same 96 hidden, same NEON kernel
dispatch. Inference cost is a function of architecture, not
weights.

**Same-shape distillation cannot improve speed.** This is the
key realization that course-corrects Phase 4e toward 4e3.

---

## 4e3 — Smaller student distillation (the actual Phase 4e work)

The speed gain comes from making the student SMALLER than
L3TC-200K. Options in rough order of expected speedup per
engineering cost:

| change | est speed | est ratio risk | engineering cost |
|---|---:|---|---|
| 1 layer instead of 2 (baseline 4e3) | **~2×** | moderate | 1-line config |
| 64 hidden instead of 96 | ~1.5× | moderate | 1-line config |
| 1 layer + 64 hidden | ~3× | high | 1-line config |
| Drop HiRA entirely | ~1.2× | moderate | 10 lines |
| Custom simplified time-mix | ~1.3× | low | medium |

**First experiment: 1 layer, 96 hidden, 96 intermediate.**
Simplest path, halves the per-token compute of the current
2-layer 200K, keeps the existing NEON 96×96 matvec kernels
hot, doesn't require new tensor shapes or kernels.

### 4e3 experiment plan

1. **Generate teacher dump on enwik8.** The current teacher
   dump is on enwik6, which is the test set. For an honest
   measurement we need to train on enwik8 (the teacher's
   actual training corpus) and evaluate on enwik6 (unseen
   during student training). Full enwik8 dump is ~75 min on
   our Rust runtime; the **first 10 MB of enwik8** is ~7.5 min
   and gives a big enough training set for a directional
   answer.

2. **Define a 1-layer student config.** Instantiate
   `RWKV_TC_HIRA` from L3TC's training code with
   `num_hidden_layers=1` instead of 2. Everything else stays
   the same: 96 hidden, 96 intermediate, rank 4, vocab 16384.
   Parameters: ~100K (roughly half the 200K model).

3. **Training from scratch with distillation.** Can't
   fine-tune from the 200K checkpoint because layer count
   differs. Initialize the student with random weights (or
   optionally copy the single remaining layer from 200K's
   layer 0 as a warm start). Train for several epochs with:
   - `α = 0.5` — balanced CE + KL
   - `T = 1.5` — moderate softening
   - `lr = 2e-4` — higher than 4e2a because we're training
     from scratch, not fine-tuning
   - 3-5 epochs on 10 MB
   - Adam optimizer, same betas as L3TC's training

4. **Periodic validation on enwik6.** Every N training steps,
   save a checkpoint and (externally) run the Rust runtime's
   `entropy-bound` and `iter.sh` on enwik6 to measure actual
   ratio. Ideally also an in-training eval to early-stop when
   ratio stops improving. For v1, save a checkpoint per epoch
   and measure after the run.

5. **Decision criteria:**
   - **enwik6 ratio ≤ 0.195 at ≥ 260 KB/s** — ship it as a
     fast tier alongside 200K default and 3.2M opt-in
   - **ratio 0.195-0.22** — keep experimenting; try bigger
     student (2 layer 64 hidden), longer training, or full
     enwik8 corpus
   - **ratio > 0.22** — the architecture is too small;
     document the finding and move on. Ratio floor of 1-layer
     96-hidden appears to be above 0.22, which means we can't
     win on speed without a bigger speed lever (like GPU batch
     inference or architectural tricks out of scope for 4e).

### 4e3 compute budget

- Teacher dump on first 10 MB of enwik8: ~7.5 min
- Training: 10 MB / 2048 bytes/segment ≈ 5000 segments
- Per-segment step: ~1 sec on MPS at batch 1
- 1 epoch: ~85 min on MPS
- 5 epochs: ~7 hours
- **Total first-experiment time: ~7 hours on MPS**, most of
  which is training

Too long for a quick iteration. Options to shorten:
- First pass on 5 MB instead of 10 MB: ~3.5 hours
- Batch size > 1 (if MPS supports it well for this model):
  could cut 2-4×
- Start with 2 epochs and check convergence, only keep going
  if ratio is improving

For the first directional experiment, target **first 5 MB of
enwik8, 2 epochs, batch size 1, MPS**: about **2-3 hours**.
Long but tractable as a background run.

### 4e3 risks and what could go wrong

- **1-layer capacity floor might be too high.** If a 1-layer
  96-hidden architecture can't reach ratio ≤ 0.22 on enwik6
  no matter the training, the speed-vs-ratio trade isn't
  worth it. We learn this after the first run and try a
  different shape (2-layer 64-hidden, etc.)
- **MPS numeric stability.** L3TC's training stack was built
  for CUDA. Our CPU WKV replacement runs on MPS in fp32, but
  there may be subtle issues with MPS's handling of the
  `scan`-like recurrence. If training diverges or produces
  NaN, we fall back to CPU (slower but solid).
- **Random init convergence.** Training a 100K model from
  scratch on 5 MB is a small data/compute budget. If it
  underfits, we need to either train longer or warm-start
  from the 200K checkpoint's layer 0 weights.
- **Weird interactions with HiRA.** The training model class
  has HiRA branches; a 1-layer student has one set of HiRA
  branches instead of two. Should work, but untested.

### What Phase 4e3 does NOT try

- Training beyond ~10 MB (full enwik8 is Phase 4e4 if 4e3 is
  promising)
- Multiple student sizes in parallel (start with one, iterate)
- A new custom architecture (use L3TC's existing
  `RWKV_TC_HIRA` training class, just with different config)
- GPU training (MPS is what we have; if MPS chokes, fall
  back to CPU)
- Production training pipeline — this is a research
  experiment, not a trainable product

---

## Success criteria

Phase 4e is done when:

- A distilled student checkpoint exists and loads cleanly in
  our Rust runtime (`l3tc compress --model checkpoints/
  l3tc_distilled.bin`)
- Byte-identical round trip on enwik6 (1 MB)
- **enwik6 compress speed ≥ 260 KB/s** (or the nearest
  achievable; document the actual number)
- **enwik6 actual coded ratio ≤ 0.195** (or the nearest
  achievable; document the actual number)
- enwik8 (100 MB) also round-trips without regression
- All 36+ Rust unit tests pass; no 200K regression
- `docs/phase_4e_findings.md` documents the training setup,
  hyperparameters, compute cost, and the final number on the
  speed × ratio curve

If the speed × ratio trade isn't on a useful Pareto point,
Phase 4e fails gracefully: we keep the original 200K as
default + 3.2M as opt-in, and document the ratio-floor
finding so we know what to try next (bigger student, different
architecture, GPU batch inference, or back to pure Phase 5
v7 work).

## Non-goals

- Training on a broader-than-enwik8 corpus (Phase 11)
- RWKV-v7 architecture upgrade (Phase 5)
- Training a completely new architecture from scratch with
  no distillation teacher
- Production training pipeline (ship-quality training infra
  is a separate question)
- CPU-only training (MPS is the target; CPU is a fallback)

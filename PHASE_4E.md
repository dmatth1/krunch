# Phase 4e — Distillation from L3TC-3.2M into a fast student

**Status:** Starting now. Phase 4d delivered L3TC-3.2M as an
opt-in "max ratio" tier at ratio **0.1337** / compress
**25.95 KB/s** on enwik6. Phase 4e asks: can we get most of
that ratio win at most of 200K's speed?

**Goal:** train a student model via knowledge distillation from
the L3TC-3.2M teacher, targeting an actual-coded enwik6 ratio
**≤ 0.155** (closing ≥60% of the 200K→3.2M ratio gap) while
staying at **≥ 99 KB/s compress** (the CLAUDE.md speed floor).

**Starting reference points:**

| | actual ratio | compress KB/s | params |
|---|---:|---:|---:|
| l3tc-rust 200K | 0.1699 | 131 | 200K |
| l3tc-rust 3.2M | **0.1337** | 25.95 | 3.2M |
| **Phase 4e target** | **≤ 0.155** | **≥ 99** | ~100-200K |

If this works: Phase 4e is the biggest combined ratio+speed
improvement of the entire project.

If it doesn't: we learn something about whether small neural
compression students can approach big-teacher quality, and the
shipping story stays at 200K default + 3.2M opt-in.

---

## Approach: fine-tune the 200K checkpoint toward 3.2M's outputs

The simplest form of distillation that can plausibly work:

1. **Start from the shipped L3TC-200K checkpoint** (not random
   init). This gives the student a strong starting point and
   dramatically reduces the number of training steps needed.
2. **Use L3TC-3.2M as a frozen teacher.** Run it over enwik8,
   capture per-token softmax probability distributions.
3. **Fine-tune the 200K student** with a loss that blends
   standard cross-entropy against ground-truth tokens with
   KL divergence against the teacher's soft distributions:

   ```
   loss = (1 - α) * CE(student_logits, target) +
          α * T^2 * KL(student_soft / T, teacher_soft / T)
   ```

   where `T` is a temperature (usually 2-4) and `α` is the
   distillation weight (usually 0.5-0.7).

4. **Convert the fine-tuned checkpoint** via our existing
   `scripts/convert_checkpoint.py` — no format changes needed,
   just new weights.

5. **Measure** entropy bound + actual coded ratio on enwik6
   with our Rust runtime.

The student architecture is **unchanged** from the shipped
L3TC-200K: 2 layers, 96 hidden, 96 intermediate, HiRA rank 4,
SPM BPE 16384. Inference cost is byte-identical to the current
200K. The only thing that changes is the weight values.

**If 200K-architecture distillation works well**, a Phase 4e2
could try a smaller student (e.g., 100K params: 1 layer or
64 hidden) to trade some ratio for more speed.

**If 200K distillation doesn't meaningfully improve ratio**, we
learn that 200K-architecture's ratio is near the ceiling at
this parameter budget and we need a bigger student or a
different architecture.

---

## Concrete tasks

### 4e1 — Dump teacher distributions from L3TC-3.2M on enwik8

Add a Rust CLI subcommand `dump-teacher` that takes:
- A model checkpoint (3.2M)
- An input file (enwik8)
- Segment boundaries matching the training setup
- Outputs: a binary file containing per-position soft
  probability distributions

Format: for each segment, for each prediction step, write the
16384 f32 softmax probabilities. Plus a header with segment
boundaries and step counts. File size for enwik8 at segment
2048 with ~275k predict steps: 275000 × 16384 × 4 bytes ≈
18 GB. Too big.

**Smarter format: top-K sparse.** For each prediction step,
store only the top K (e.g., K=64) probabilities plus their
token ids. The student's KL loss only needs to match the
teacher on the tokens the teacher thinks are likely; for
the long tail, a floor value (teacher_tail = 1e-6 say) is
indistinguishable from the real tiny probabilities.

Format per step: `u32 n_active + n_active × (u32 token_id +
f32 prob)`. For K=64: 64 × 8 bytes = 512 bytes/step + 4 byte
header. For ~275k steps: 512 × 275000 = ~140 MB. Acceptable.

Even tighter: quantize the probs to u16 or u8, halve or
quarter the storage again. Defer unless 140 MB hurts.

**Alternative:** skip the file dump entirely and compute
teacher distributions on-the-fly during training. Only viable
if the training loop can call our Rust runtime via FFI, which
is complex. File dump is simpler.

### 4e2 — Build the PyTorch training loop

Either reuse `vendor/L3TC/models/` training infrastructure
with a modified loss, or write a minimal from-scratch loop.

Minimal approach:
1. Load L3TC-200K checkpoint in PyTorch via the existing
   `RWKV_TC_HIRA_Infer_For_Script` class (or the training
   variant)
2. Freeze nothing (all params trainable)
3. Load enwik8 in 2048-byte segments with SPM tokenization
4. Load teacher distributions from the dump file
5. For each batch of segments:
   a. Forward pass student → logits
   b. Compute standard CE loss on ground-truth tokens
   c. Compute KL loss on teacher distributions
   d. Blend: `loss = α * KL + (1-α) * CE`
   e. Backward pass + optimizer step
6. Every N steps: eval on enwik6 via the Rust runtime (or
   recompute in PyTorch)
7. Save checkpoint when entropy bound on enwik6 improves

Compute estimate:
- 200K params × enwik8 forward+backward
- On M-series CPU: ~8-12 KB/s effective training throughput
- enwik8 one epoch: 100 MB / 10 KB/s ≈ 10000 s ≈ 2.8 hours
- 5-10 epochs: 14-28 hours
- On M-series with PyTorch MPS: ~5-10× faster if supported
- With GPU (not on this machine): minutes

**Reality check:** we probably want a GPU for this, or at
least a lot of patience. Running on the Apple Silicon CPU is
feasible but slow. Running a first pass on a small subset of
enwik8 (say 10 MB) for sanity-check first.

### 4e3 — Convert + measure

The fine-tuned checkpoint has the same shape as the original
L3TC-200K, so our existing converter should work unchanged.
Drop the new `.bin` next to the old one, run:

- `l3tc entropy-bound --input enwik6 --model checkpoints/
  l3tc_200k_distilled.bin` — did the entropy bound improve?
- `iter.sh` with the distilled checkpoint — did the actual
  coded ratio improve without speed regression?
- `l3tc compress --verify` on enwik8 — does byte-identical
  round trip still work?
- `docs/phase_4e_findings.md` — document the training setup,
  hyperparameters, final numbers, and the comparison against
  Phase 4d's baselines

Success criterion: enwik6 actual coded ratio ≤ 0.155 at
≥ 99 KB/s compress, with byte-identical round trip preserved.

### 4e4 — Optional: smaller student

If 4e1-3 produce a meaningfully-better 200K, try a 100K
student with a reduced architecture (e.g., 64 hidden, 2 layers).
Same distillation pipeline, different model shape. Target: ratio
≤ 0.17 at ≥ 200 KB/s compress. This is the "2× speed win" path.

Out of scope for the initial Phase 4e ship — add as a
sub-phase only if the full-size distillation works.

---

## Compute budget and realism

The single biggest risk is compute. Training ~200K params on
~100 MB of text for a few epochs takes:

| hardware | time estimate |
|---|---|
| Apple Silicon CPU | ~15-30 hours |
| Apple Silicon MPS (PyTorch) | ~2-5 hours |
| consumer NVIDIA GPU (RTX 3060+) | ~30 minutes |
| cloud A10G / L4 spot instance | ~15 minutes, ~$0.20 |

If we have local GPU access: ship this in a day.
If we don't: either run overnight on CPU, try MPS, or rent
cloud compute briefly. The pipeline is the same either way.

**Intermediate milestone:** run the whole pipeline on a
*tiny subset* (e.g., the first 1 MB of enwik8, ~5 minutes to
train end-to-end on CPU) first to debug the loss function and
format. Only commit to a full training run once the pipeline
is known-good.

---

## Success criteria

Phase 4e is done when:

- `l3tc_200k_distilled.bin` exists in the repo, produced by
  fine-tuning the original 200K checkpoint
- enwik6 actual coded ratio ≤ 0.155 on the distilled checkpoint
  (closing ≥60% of the 200K→3.2M gap)
- enwik6 compress ≥ 99 KB/s (CLAUDE.md speed floor)
- enwik6 byte-identical round trip
- All 36 unit tests pass
- `docs/phase_4e_findings.md` with the full training setup,
  hyperparameters, and measurement comparison

## Non-goals

- Training a completely new architecture from scratch (Phase 5
  or later if at all)
- Training on a broader corpus (Phase 11)
- Retraining the tokenizer (locked to the shipped SPM)
- Running inference on anything but CPU for benchmarking
  (training may use GPU, but shipped artifact is CPU-runnable)
- A production training pipeline (this phase is a one-off
  research experiment — if distillation works we productize
  it in a later phase)

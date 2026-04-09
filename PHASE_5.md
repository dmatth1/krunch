# Phase 5 — RWKV-v7 architecture upgrade (still enwik8)

**Goal:** replace the RWKV-v4-with-HiRA forward pass with RWKV-v7
at the same ~200K parameter budget, retrained from scratch on
enwik8. Target a measurable ratio improvement over the current
v4-based entropy bound (0.1632 at segment 4096) without giving
up speed.

**Starting point (end of Phase 4b2):**
- enwik6 actual ratio **0.1699** at segment 4096
- enwik6 entropy bound **0.1632**
- 119 KB/s compress, 119 KB/s decompress
- Gap to entropy bound 0.61 pp (86% closed)
- RWKV-v4 200K + HiRA, trained on enwik8
- Forward pass bit-identical to Python reference
- File format v4 (varint segments, CRC32 trailer)
- 34 unit tests + 4 end-to-end integration tests passing

**Why v7 now.** The Phase 4 diff harness proved the ratio lives
in the *model*, not the codec. Our AC body is within 3 bytes of
the entropy bound — we can't squeeze more ratio out of the coder
side. The remaining lever on enwik6 is "can the same 200K
parameter budget predict next tokens more accurately with a
better architecture". RWKV-v7 is the natural candidate:

- ~3 years newer than RWKV-v4 with several rounds of published
  improvements
- Known wins on standard language-modeling benchmarks at
  comparable parameter counts
- Still linear-time / constant-memory per token (no
  quadratic attention), so the speed profile doesn't blow up
- Training infrastructure exists in the upstream RWKV repo and
  is relatively approachable

Important scope constraint from the user in the Phase 4b planning
discussion: **stick with enwik8 as the training corpus for now.**
Broader corpora (The Pile, RedPajama, customer-specific) are a
separate question, deferred to the generalization phases (Phase 8
specialist dispatch, or a future broader-training phase if the
need arises). Phase 5 is an apples-to-apples architecture
comparison: same data, same parameter count, different forward
pass.

---

## What "upgrading to v7" actually involves

RWKV-v7 is a meaningfully different forward pass than v4. The
specifics have to be read from the upstream repo and the v7
paper, but at a high level the changes we'll encounter:

1. **Different time-mix semantics.** v4's
   `x * mix_k + state_x * (1 - mix_k)` becomes a more elaborate
   recurrence with per-head learned mixing and data-dependent
   decays. The Rust `time_mix` op gets replaced with whatever v7
   specifies; our `src/rwkv.rs::time_mix` needs a full rewrite.

2. **Data-dependent decay** (`w_t` in v6/v7). v4's time_decay is
   a learned per-channel constant. v7 makes decay a function of
   the current input, which means more matmuls per token and
   tighter integration between state update and the current
   input.

3. **Channel mixing changes.** v7's FFN path is different from
   v4's — possibly squared-relu or GLU variants, different
   linear shapes.

4. **New state layout.** v4's `(state_a, state_b, state_p,
   state_x, state_ffn)` shape is specific to v4. v7 will have
   its own state shape per layer.

5. **No HiRA.** L3TC's HiRA reparameterization is a v4-specific
   trick. v7 has its own parameterization story; we'd either
   skip HiRA entirely or port it, depending on what the training
   script demands.

The net effect: the forward pass in `src/rwkv.rs` is essentially
rewritten. `src/tensor.rs` may need additional ops (new element-
wise functions, possibly new matvec shapes). `src/checkpoint.rs`
needs a new loader for v7-shaped weights. `scripts/convert_check
point.py` needs to handle the v7 checkpoint format.

Phase 4's diff harness (`l3tc dump-logits`, `scripts/dump_python
_logits.py`, `scripts/diff_logits.py`) carries over directly —
we re-use it to validate the v7 Rust forward pass against a
Python v7 reference implementation the same way we validated v4.
That's the biggest leverage point: we already have the
verification tooling.

---

## 5a — Train RWKV-v7-200K on enwik8

Before any Rust work, we need a trained checkpoint to port.

Tasks:

1. Clone upstream RWKV-LM (or the v7 training fork, whichever
   is the canonical source).
2. Configure a 200K-parameter model — match the shape of the
   current L3TC-200K as closely as possible (2 layers, 96 hidden,
   SPM tokenizer with vocab 16384).
3. Point it at enwik8 + the existing SPM tokenizer model
   (`vendor/L3TC/dictionary/vocab_enwik8_bpe_16384_0.999/...`).
4. Train from scratch for comparable compute to the original
   L3TC-200K training run (~20 epochs, batch 32, seq 2048 — or
   whatever the v7 training script recommends).
5. Evaluate at each epoch: measure training cross-entropy, and
   every few epochs run the PyTorch forward pass on enwik6 and
   compute the entropy bound. We want the v7 entropy bound to be
   measurably lower than v4's 0.1632 before we bother porting.

**Checkpoint:** if after 20 epochs v7-200K's entropy bound isn't
at least 0.5 pp better than v4 on enwik6, **stop the phase**.
Architecture choice didn't help at this parameter count; no
point porting. (This is a real risk — small models don't always
benefit from the architectural improvements that help big ones.)

**Compute estimate:** 2 layers × 96 hidden × 200K params is
tiny. Training on enwik8 for 20 epochs probably takes a few
hours on a single A10G / L4. Call it $20-50 of cloud compute.
Cheap enough to not worry about.

## 5b — Port the v7 forward pass to Rust

Only starts if 5a shows a ratio improvement.

Tasks:

1. **Convert the v7 checkpoint** to our Rust binary format.
   Extend `scripts/convert_checkpoint.py` with a v7 path. May
   need new tensor types if v7 has weights we don't currently
   handle.

2. **Rewrite `src/rwkv.rs::time_mix`** to match v7's attention /
   WKV recurrence. Unit-test against a tiny reference on known
   inputs before hooking up the full pipeline.

3. **Rewrite `channel_mix`** if it changed (it did in v6, likely
   did in v7 too).

4. **Add any missing tensor ops** to `src/tensor.rs` — anything
   new that v7 needs (e.g., data-dependent sigmoid or squared-
   relu variants we haven't used).

5. **Update `src/rwkv.rs::LayerState`** to match the v7 state
   shape.

6. **Validate via the diff harness.** Run
   `l3tc dump-logits` against a Python v7 reference dump. Max
   L_inf should be in the 1e-5 range (f32 ULP). Fix any
   divergence before moving on.

7. **Re-measure entropy bound and actual coded ratio** on enwik6
   and enwik8. Commit only if both improve without regressing
   speed >15%.

---

## Why NOT do this now

- Phase 4b2 got the v4 architecture within 0.61 pp of its
  entropy bound. Further ratio improvement from architecture
  (v7) is maybe 0.5-1 pp at this parameter count, which is
  smaller than the ~3 pp we just won from Phase 4b. Diminishing
  returns.
- v7 port is a multi-week engineering lift (rewrite forward
  pass, retrain, validate) for a modest ratio win.
- The bigger structural problems (OOD, cross-platform
  determinism, distribution) aren't addressed by v7. A v7 model
  would still crash on webster and still be non-portable.
- If the eventual direction is the storage service vision, the
  training-side work converges with v7 naturally: we'd be
  training per-customer models anyway. Phase 5 becomes "start
  training on v7 when we ship the service".

Phase 5 is on the roadmap because the ratio lever exists and is
worth documenting; it's not the next thing to work on.

---

## Success criteria

- RWKV-v7-200K trained on enwik8, checkpoint saved
- Python reference reports entropy bound ≤ 0.158 on enwik6
  (at least 0.5 pp better than v4's 0.1632)
- Rust v7 forward pass bit-identical to Python v7 (max L_inf
  < 1e-4 on first 256 tokens of enwik6)
- enwik6 actual coded ratio ≤ 0.165 (closing most of the v7
  entropy gap)
- Compress speed ≥ 99 KB/s on enwik6 (speed floor from
  CLAUDE.md)
- All unit tests pass; new tests for the v7 forward pass
- `docs/phase_5_findings.md` documents the architecture delta,
  ratio delta, speed delta, and any gotchas

## Non-goals

- Broader training corpora (enwik8 only — out of scope for
  Phase 5)
- Multi-model dispatch (Phase 8)
- RWKV-7 at non-200K sizes (optionally measure but don't ship
  as default — the speed budget rules out 3.2M+)
- Training infrastructure productization (Phase 5 is a one-time
  training run for research, not a pipeline)
- Porting other modern architectures (Mamba, xLSTM, RetNet) —
  could be a Phase 5b if v7 disappoints

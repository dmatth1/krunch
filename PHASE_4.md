# Phase 4 — Hybrid with classical + close the ratio gap to Python L3TC

Phase 4 has two complementary deliverables:

- **4a — Hybrid fallback to a classical compressor.** Stops
  l3tc-rust from ever making a file *larger* than its input on
  out-of-distribution corpora. Shipped as default behaviour.
- **4b — Close the ratio gap to Python L3TC.** Match the Python
  reference within ±0.5 pp on enwik6 without losing more than 15%
  on compress speed. The headline goal of CLAUDE.md.

4a is days of work and removes a visible regression today; 4b is
~1-2 weeks and closes the largest single quality debt in the
project. Ship 4a first so user-visible behaviour stops being
embarrassing on Silesia/binary inputs, then take the time to do
4b properly.

---

## 4a — Hybrid fallback to a classical compressor

**Why:** Phase 3's Silesia run uncovered the OOD failure mode in
its loudest form: webster (a 41 MB English dictionary) compresses
to **52 MB** through l3tc-rust's text path — 27% *larger* than
the original. The L3TC-200K model is trained on enwik8 prose, so
the dictionary's idiosyncratic format is wildly out of
distribution; the AC needs more bits per token than raw bytes
have. This isn't a bug — it's the fundamental tradeoff of
model-based compression — but a "compressor" that makes files
larger is a non-starter for any default workflow.

Shipping with classical fallback gives a strict lower bound:
`final_size = min(l3tc_size, classical_size)`. Worst case becomes
"as good as zstd"; best case stays "best ratio in the suite".

**Approach:**

1. Add `zstd-rs` as a dependency (the de facto Rust binding for
   the zstd C library; Apache-2.0; widely audited).
2. New `compress_with_fallback(...)` path:
   - Run the LM-based encoder.
   - Run zstd at level 19 in parallel on the same input.
   - Pick whichever output is smaller.
   - Record which one was used in a new header flag bit
     (`FLAG_CLASSICAL_FALLBACK`).
3. Decoder branches on the flag: if set, decompress via zstd; if
   clear, take the existing tokenized path.
4. CLI gets a `--no-fallback` switch for users who want the pure
   LM path (e.g. for ratio research or to reproduce old numbers).

**Sub-tasks rolled in (UTF-8 robustness gaps from Phase 3):**

The 4a work also fixes two related Silesia gaps:
- **Bug A** "stray byte poisons whole file" (dickens — 8 stray
  high bytes in 10 MB ASCII send the entire file to raw-store).
  Fix: per-segment UTF-8 detection, with invalid-byte regions
  routed through the existing `needs_raw_fallback` segment path
  rather than escalating to whole-file raw-store.
- **Bug B** "mid-stream UTF-8 failure crashes encode_reader"
  (reymont, xml — first batch passes UTF-8 but later bytes fail,
  encode errors out partway with a half-written output file).
  Fix: catch the mid-stream error inside encode_reader and either
  fall back to the classical-compressor path or restart in
  raw-store mode. Should never crash.

**Success criteria:**

- l3tc-rust output is never larger than the input + 28 bytes
  on any corpus we test (enwik6/8/9, all of Silesia, Canterbury,
  binary blobs).
- enwik6 ratio is unchanged (0.2061) — the LM path wins on text.
- Webster: 1.2613 → ≤ zstd's ratio (~0.21).
- Reymont/xml: encode_reader no longer crashes; either uses LM
  path with per-segment fallback or classical fallback, never
  both errors.
- All 36 unit tests pass + a new "fallback never makes things
  bigger" property test that runs on a fuzzed input.

---

## 4b — Close the ratio gap to Python L3TC

**Starting point (end of Phase 3):**
- Ratio 0.2061 on enwik6, 0.2166 on enwik8
- Python L3TC-200K reference: **0.1665** on enwik6
- **Gap: ~4 percentage points** (24% worse than the reference)
- 116 KB/s compress, 121 KB/s decompress on enwik6
- 7.4× faster than the Python reference, byte-identical round trip,
  v3 file format with CRC, streaming encode + decode, binary-input
  raw-store mode

**Phase 4 goal (the project's headline goal #1 from CLAUDE.md):**

> Match the Python L3TC-200K compression ratio within ±0.5 pp on
> enwik6 (target ≤0.1715) **without losing more than 15% on
> compress speed**.

The 4 pp gap is the largest single quality debt in the project.
Closing it would also flow through to enwik8 (currently 0.2166 →
target ≤0.18) and any other corpus we benchmark, putting
l3tc-rust ahead of every classical compressor by an even larger
margin.

---

## Where the gap actually comes from

We have several theories from earlier phases. None have been
isolated empirically yet. Phase 4's first job is to **measure**.

**Suspect 1: cum_freqs precision and quantization**
- The current code computes per-token freqs as
  `(((e as f64) * scale).floor() as u64).max(1)` with
  `scale = (target_total - n) / sum`. The `.max(1)` clamp gives
  every long-tail token a freq of 1 even when its true probability
  is much smaller than `1/target_total`.
- The Python reference uses a slightly different rounding rule
  (round-to-nearest with banker's tie-breaking) and a different
  fixup pass.
- Empirical test: log per-token freqs from both implementations on
  the first 1000 tokens of enwik6 and diff. If they differ
  systematically the gap is here.

**Suspect 2: model state precision (f32 vs f16)**
- Python L3TC runs the forward pass in f32 throughout. So do we.
  But the INT8 head quantization in 2.5b introduces ~0.4% relative
  error on logits before softmax. We measured the ratio impact as
  ~+0.0001 on enwik6, which is small — but on enwik8 the impact
  could be larger.
- Test: revert the head to f32 temporarily and measure. If the
  ratio improves measurably, INT8 is contributing to the gap and
  we need a smarter quantization scheme (per-row scales? blocked?).

**Suspect 3: tokenizer fallback path**
- The raw-fallback mechanism stores certain segments as plain
  bytes alongside the AC body. Python doesn't have this fallback
  — it relies on SPM round-tripping cleanly. On corpora without
  Persian/Arabic/CJK content the fallback should never trigger.
- Test: count fallback segments per corpus. If enwik6 has none
  this isn't the cause.

**Suspect 4: segment boundary leakage**
- Each segment resets the model state, which costs ~1-2 tokens of
  context for the AC's first predictions. With 4096-byte segments
  on enwik6 there are ~250 segments, so the cost is ~250-500
  tokens × ~0.5 bits = ~150 bytes. That's 0.015% of the 0.2 ratio
  — too small to explain the gap on its own, but could compound.
- Test: try segment_bytes = 32768 and measure the ratio. If it
  drops materially, segment overhead is real.

**Suspect 5: HiRA merge precision**
- Phase 1's checkpoint converter merges HiRA branches into the
  base weights at the f32 level (`W' = W + B @ A`). On the
  L3TC-200K model the BA term is ~10% the magnitude of W, so the
  merged W' has reduced effective precision.
- Test: verify the merged W' against a forward pass that keeps
  HiRA separate. If they differ beyond f32 epsilon the merge is
  lossy.

---

## Phase 4 plan

### 4a — Measurement: pin down the actual gap source

Before changing any code, write a measurement harness that runs
the Python reference and l3tc-rust on the same input and dumps:
- Per-token freqs at every position (both implementations)
- Per-token logits (both)
- Per-token AC state (both)

Diff the dumps. The first place they diverge is the cause.

This is the only way to isolate which suspect is real. Estimated
~1-2 days. **No optimization commits before this is done.**

### 4b — Fix the dominant cause first

Whichever suspect 4a fingers, fix that one cleanly. Each fix is
its own commit with before/after ratios on enwik6 and enwik8.

Order of likely fixes (most impactful first):
1. Replace cum_freqs's `floor + max(1)` with banker's-rounding +
   the Python fixup pass — only if 4a shows the gap is here.
2. Replace INT8 head with INT8-blocked head (per-block scales)
   or revert to f32 head — only if 4a shows the head is the cost.
3. Re-derive the HiRA merge in f64 and convert to f32 only at the
   end — only if 4a shows HiRA precision is the cost.

### 4c — Re-measure on enwik8 and Silesia

After each fix in 4b, run:
- enwik6 (1 MB) — quick gate
- enwik8 (100 MB) — production reference
- Silesia text files — confirm the win generalizes

Commit only if the ratio improves on at least enwik6 + enwik8.

### 4d — Defend the speed budget

Phase 4 has a strict speed floor: any ratio improvement that
costs >15% compress throughput must be opt-in (CLI flag, not
default). The default binary stays under the speed budget. The
flag exists for users who want maximum ratio at the cost of speed
(roughly the Python L3TC use case).

---

## Success criteria (Phase 4 exit)

Phase 4 is done when:
- enwik6 ratio ≤ 0.1715 (≤0.5 pp from Python L3TC-200K)
- enwik8 ratio ≤ 0.18 (corresponding improvement)
- Compress speed ≥ 99 KB/s on enwik6 (≤15% drop from Phase 3 116)
- All 36+ unit tests pass
- All end-to-end integration tests pass
- `docs/phase_4_findings.md` documents which suspect was the
  actual cause and what the fix was

## Non-goals (deferred to later phases)

- Multi-platform release builds (Phase 6)
- Replacing the model architecture (Phase 5 or later)
- Retraining the model on a larger or cleaner corpus (Phase 5)
- Adding new compressor variants (3.2M, 12M) — once 200K is
  matched, the bigger models come essentially for free

## Why this is the highest-value work

The CLAUDE.md goals are ratio (#1) and speed (#2). Phase 0-3
moved speed by 7.4× (huge) but only moved ratio by 0.0033 (from
0.2094 → 0.2061). The ratio is where we have the most room to
improve and the most direct payoff for users — every percentage
point of ratio improvement saves a percentage point of disk on
every file forever, and l3tc-rust is currently the
best-compressing tool in the bench suite *despite* this gap.
Closing it makes the lead structural rather than marginal.

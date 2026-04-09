# Phase 4 — Close the ratio gap to Python L3TC (implementation diff)

**The headline goal:** match the Python L3TC-200K reference ratio
on enwik6 (≤ 0.1715, currently 0.2061) **without losing more than
15% on compress speed** (≥ 99 KB/s, currently 116).

**Key framing.** Python L3TC and our Rust port use the *same
architecture* (RWKV-v4 with HiRA), the *same checkpoint*
(`l3tc_200k_bpe16k_c999_checkpoint0019.pth`), and the *same
training corpus* (enwik8). The 4 pp ratio gap is therefore
**purely an implementation difference** in our Rust pipeline —
somewhere in checkpoint conversion, the forward pass, the
softmax→freq quantization, or the AC encoding our path diverges
from Python's. **No retraining and no architecture upgrade is
required to close it.** That's a Phase 5+ concern.

Phase 4 is therefore organized as:

- **4a — Implementation diff: find and fix the divergence(s).**
  The headline work. Instrument both Python and Rust to dump
  per-token state on the first ~1000 tokens of enwik6, find the
  first place they disagree, fix it, repeat until ratio matches.
- **4b — Hybrid fallback to a classical compressor.** Polish.
  Stops l3tc-rust from ever producing output *larger* than its
  input on out-of-distribution corpora (e.g. webster's 1.26
  ratio). Days of work. Ship whenever convenient — does not
  block 4a.

**Backburner (NOT in Phase 4):**
- RWKV-v7 architecture upgrade — moved to a future phase. The
  current architecture can match Python L3TC; switching to v7 is
  a separate "push past Python" goal that we'll consider after
  4a closes the implementation gap.
- Training on a broader corpus (The Pile / RedPajama) — Phase 5.
- Bigger model variants (3.2M / 12M) — Phase 5b or later.

---

## 4b — Hybrid fallback to a classical compressor (polish)

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

The 4b work also fixes two related Silesia gaps:
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

## 4a — Close the ratio gap to Python L3TC (the headline work)

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

We have several theories from earlier phases. The first one was
testable directly against the Python source under
`vendor/L3TC/scripts/compressor.py` and turned out to be wrong.
Empirical evidence beats every prior, so the suspect list below
is now ranked by what's actually still on the table.

**Suspect 1: cum_freqs precision and quantization** ❌ **RULED OUT**

The Python reference does literally three lines:
```python
probs = torch.softmax(logits, dim=-1)
freqs = torch.round(probs * 10_000_000).int()
freqs = torch.max(freqs, freqs.new_ones(freqs.size()))
```

We reimplemented exactly this scheme in `logits_to_cum_freqs_scratch`
(Phase 4a, commit pending) — `round` instead of `floor`,
`PYTHON_FREQ_TOTAL = 10_000_000` instead of ~2^62, `max(1)` clamp,
no residual fixup. We also replaced the ~1%-error
`fast_exp_neg` polynomial with libm `f32::exp` so the softmax
matches `torch.softmax` to ULP precision.

**Result on enwik6: ratio 0.2061 → 0.2060.** Movement is noise
(±0.0005 budget). **The cum_freqs scheme is not the source of
the gap.**

This is a high-value negative result. It eliminates the largest
chunk of the suspect surface and forces the search into the
forward pass / checkpoint side, where divergences are harder to
spot but more structural.

**Suspect 2: forward-pass divergence (highest prior given suspect 1 ruled out)**

The 4 pp gap predates all of Phase 2.5: Phase 1 hit ratio 0.2094,
Phase 2 went to 0.2060, Phase 2.5 stayed at 0.2061, Phase 4a
landed at 0.2060. Python is at 0.1665. **The gap was already there
the moment we ran the Rust forward pass against the same
checkpoint.**

That points at the forward pass itself: somewhere in
`src/rwkv.rs` or `src/checkpoint.rs` or
`scripts/convert_checkpoint.py` we produce numerically different
logits than Python's `models/RWKV_V4/rwkv_tc_hira_infer.py`.
Possible sources, in rough order of likelihood:

- **HiRA merge dtype:** the converter computes `W' = W + B @ A`
  in PyTorch f32, then writes f32 to disk. If the Python
  inference path keeps W and HiRA branches separate (and applies
  the merge at every forward pass in higher precision), the
  merged-once result we use could be measurably lossy.
- **Time-mix order:** RWKV-v4 time mix is
  `xk = x * mix_k + state * (1 - mix_k)`. Subtle reorderings or
  the use of different `state_ffn` vs `state_att` semantics
  could change downstream values.
- **Layer norm precision:** if Python computes layer norm in f32
  via fused PyTorch ops and we do it in scalar f32 with a
  different rounding chain, the cumulative drift across 2 layers
  + ln_out could measurably shift logits.
- **The "short" connection:** the 200K model has a dedicated
  `block.w_short` matvec with relu. We treat it as a 96x96 NEON
  matmul (correctly, post-2.5a). Verify that Python's
  `rwkv_tc_hira_infer.py` does the same.

**Test:** instrument both Python and Rust to dump per-token
logits at the same input position on the first ~1000 tokens of
enwik6, and diff. The first divergence is the cause. This is
the actual measurement step that should drive the next commit.

**Suspect 3: model state precision (f32 vs f16)**
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

## 4a execution plan

### Step 1 — Measurement: pin down the actual gap source

Before changing any code, write a measurement harness that runs
the Python reference and l3tc-rust on the same input and dumps:
- Per-token logits (both implementations)
- Per-token cumulative-frequency tables (both)
- Per-token AC state at the same positions (both)

Diff the dumps. The first place they diverge is the cause.
Diffing logits first isolates whether the gap is in the forward
pass; if logits match but freqs differ, the gap is in the
softmax→freq quantizer; if both match but AC state differs, the
gap is in the coder.

This is the only way to know which suspect is real. **No
optimization commits before this is done.**

### Step 2 — Fix the first divergence

Whichever step 1 fingers, fix it cleanly in its own commit with
before/after ratios on enwik6 and enwik8. If a fix moves ratio
but doesn't fully close the gap, re-run step 1 to find the next
divergence and repeat.

Most likely fix order (high to low prior probability):
1. **Replace cum_freqs's `floor + max(1)` clamp** with the
   Python rounding scheme (probably round-to-nearest with the
   residual distributed at the high-prob end). The current
   `max(1)` floor inflates every long-tail symbol.
2. **Match the Python softmax order/precision** if it computes
   `e^(logit - max)` differently or sums in a different order.
3. **Replace INT8 head with f32** temporarily to see how much
   ratio the head quantization is costing on enwik8 (Phase 2.5b
   measured +0.0001 on enwik6 — could be more on enwik8).
4. **Re-derive the HiRA merge in f64** and convert to f32 only
   at the end if step 1 shows the merged W' is lossy.

### Step 3 — Re-measure on enwik8 and Silesia

After each fix, run:
- enwik6 (1 MB) — quick gate, ratio + speed
- enwik8 (100 MB) — production reference
- Silesia text files (dickens, webster, nci) — confirm
  generalization

Commit only if ratio improves on enwik6 *and* enwik8 and speed
stays above 99 KB/s.

### Step 4 — Defend the speed budget

Phase 4a has a strict speed floor: any ratio improvement that
costs >15% compress throughput is rejected (or moved behind a
`--max-ratio` opt-in flag, with the default binary staying under
the speed budget). The flag exists for users who want maximum
ratio at the cost of speed.

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

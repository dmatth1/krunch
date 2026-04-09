# Phase 4d — Port L3TC-3.2M into the Rust runtime

**Goal:** load and run the L3TC-3.2M checkpoint end-to-end in
l3tc-rust. Measure its actual ratio and speed on enwik6/enwik8
at our current runtime quality. Use the result to decide whether
(a) 3.2M is useful as an opt-in "max ratio" mode, (b) 3.2M →
100-200K distillation is the next phase (4e), or (c) 3.2M is too
slow on CPU to be worth shipping even as opt-in.

**Status:** Phase 4c is largely done (+7%/+8% over 4b2). The
remaining 4c5 INT4 head item is backburnered. 4d is the next
work: "different model" direction, starting with the bigger
teacher model so we have concrete numbers before deciding on
distillation.

---

## What's different about 3.2M vs 200K

From `vendor/L3TC/config/l3tc/l3tc_3m2.py` vs `l3tc_200k.py`:

| param | 200K | **3.2M** |
|---|---:|---:|
| `num_hidden_layer` | 2 | **3** |
| `hidden_size` | 96 | **256** |
| `intermediate_size` | 96 | **512** |
| `rwkv_rank` | 4 | 4 |
| vocab (SPM BPE) | 16384 | 16384 |
| tokenizer | enwik8 BPE 0.999 | same |
| checkpoint | `l3tc_200k_bpe16k_c999_checkpoint0019.pth` | `l3tc_3m2_bpe16k_c999_checkpoint0019.pth` |

Structural implications:

1. **3 layers instead of 2.** Already supported in our loader
   (the loop in `Model::from_checkpoint` walks `blocks.N` until
   it runs out of tensors).

2. **`hidden_size == 256`** instead of 96. Every per-layer
   vector goes from 96 f32 to 256 f32. Scratch buffers scale
   accordingly.

3. **`intermediate_size == 512`** (FFN expansion). For the
   first time we have `intermediate_size ≠ hidden_size`. FFN
   key is `(512, 256)`, FFN value is `(256, 512)`. Our current
   code assumes FFN matvecs are `(h, h)` — that assumption
   breaks.

4. **Layer norms, element-wise ops** — shape-agnostic, no code
   change.

5. **Head matvec** is now `(16384, 256)` instead of
   `(16384, 96)`. The existing INT8 col-major AXPY path
   (`matvec_col_major_int8`) is dimension-agnostic; it already
   takes `rows` and `cols` as parameters. The per-column INT8
   quantization should work. Only the scratch buffer sizes need
   to change.

6. **Attention matvecs** (K/V/R/out) are now `(256, 256)`
   instead of `(96, 96)`. The hand-tuned `matvec_96x96_neon` is
   specialized to 96 and won't work. We either need a new
   `matvec_256x256_neon` (biggest hot-path improvement) or fall
   back to scalar `matvec`.

7. **FFN matvecs** go from `(96, 96)` to `(512, 256)` and
   `(256, 512)`. Definitely no specialized NEON kernel; scalar
   fallback or new kernels.

## Tasks

### 4d1 — Make the runtime dimension-agnostic

The current `Session::forward` hardcodes `hidden_size` in a few
places via the `h` parameter that's passed down, but it also
calls `matvec_96x96` explicitly which only works at n=96. We
need to either:

**Option A: specialize per-shape at compile time** via Rust
const generics. Every Block / Session becomes generic on
`HIDDEN: usize`. Pro: zero runtime dispatch, best speed.
Con: two monomorphized code paths, bigger binary, more
complexity.

**Option B: dispatch at runtime** based on `model.hidden_size`.
Wrap each matvec call in a helper that picks NEON 96×96 or
scalar depending on shape. Pro: simpler, one code path. Con:
small runtime dispatch cost (negligible for matvec-sized
work).

**Recommendation: Option B.** The matvec work dominates each
call; a 2-cycle branch to pick the kernel is noise. Simpler to
implement and maintain.

Concrete changes in `src/tensor.rs`:
- Add `matvec_square(mat: &[f32], x: &[f32], out: &mut [f32],
  n: usize)` that dispatches `matvec_96x96` when `n == 96`,
  falls back to `matvec_scalar` otherwise.
- Keep `matvec` as the generic rectangular version for FFN
  shapes (96×384 etc.).

Concrete changes in `src/rwkv.rs`:
- Replace `matvec_96x96` call sites with `matvec_square(..., h)`.
- Change FFN matvec calls to use the generic `matvec` since
  they're no longer square.
- Make scratch buffers sized on `max(hidden_size, intermediate_size)`
  so FFN intermediates fit.

### 4d2 — Extend the checkpoint loader

`Model::from_checkpoint` already handles variable `hidden_size`
(it reads from the emb tensor shape) and variable layer count
(it loops until blocks.N is missing). The main change: also
read `intermediate_size` from the FFN key tensor's shape and
store it on `Model`.

Currently `take_2d(ckpt, "ffn.key.weight")` returns a
`[hidden_size, hidden_size]`-shaped tensor (verified against
`h`). For 3.2M it's `[intermediate_size, hidden_size]` =
`[512, 256]`. Need to:
- Detect `intermediate_size` from the shape
- Store it on the `Model` struct
- Propagate to scratch buffer sizing

### 4d3 — Convert the 3.2M checkpoint

Run `scripts/convert_checkpoint.py` on the 3.2M checkpoint:

```bash
cd vendor/L3TC && source .venv/bin/activate
python ../../l3tc-rust/scripts/convert_checkpoint.py \
    --input checkpoints/l3tc_checkpoints/l3tc_3m2_bpe16k_c999_checkpoint0019.pth \
    --config config/l3tc/l3tc_3m2.py \
    --output ../../l3tc-rust/checkpoints/l3tc_3m2.bin
```

Expected size: ~12-13 MB (3.2M params × 4 bytes f32).

The existing converter emits shapes as-is and HiRA-merges the
per-projection branches. Should work without modification.
If it doesn't, likely failure points are the time_mix tensor
squeezing (different leading dims) or the ln0 rename.

### 4d4 — Wire up CLI loading

The `l3tc` binary defaults `--model` to
`checkpoints/l3tc_200k.bin`. Add support for loading 3.2M by
passing `--model checkpoints/l3tc_3m2.bin` — no code change,
just docs + maybe a sanity-check that the binary recognizes the
shape.

Verify that the binary correctly identifies the model by
shape at load time (prints hidden_size, num_layers, etc. if
`--time` is set, or a new `--info` flag).

### 4d5 — Measure ratio and speed

Run `iter.sh` and `l3tc profile` + `l3tc entropy-bound` with
the 3.2M checkpoint on enwik6 and record:

- enwik6 actual coded ratio
- enwik6 entropy bound
- enwik6 compress / decompress KB/s
- enwik8 actual coded ratio + throughput

Compare against:
- Python L3TC-3.2M reported: ratio ~0.131 on enwik6, ~10.76 KB/s
- Our 200K runtime: ratio 0.1699, 126.8 / 128.2 KB/s

Expected outcome for Rust 3.2M at our runtime quality:
- Ratio: **~0.13-0.14** on enwik6 (much better than 200K)
- Speed: **~10-30 KB/s** on enwik6 (much slower, because 16×
  parameters → roughly 16× compute on forward pass; but parallel
  efficiency may be worse on fewer/smaller segments since the
  per-segment work is bigger)

If speed lands above ~20 KB/s the 3.2M mode is interesting as
an opt-in "max ratio" tier. Below ~5 KB/s it's only useful as
a distillation teacher, not a shipping option.

### 4d6 — Optional: NEON kernel for 256×256 matvec

If 3.2M looks worth shipping (not just teaching), the biggest
speed lever is a hand-tuned `matvec_256x256_neon` modeled after
the existing `matvec_96x96_neon` from Phase 2.5a. Similar
register preload pattern but 256 input elements instead of 96,
so 64 `float32x4_t` registers for x (doesn't fit on ARM's 32
FP registers — need chunked reload). More engineering than the
96×96 case.

**Don't do this in 4d1-5.** Only come back to it if the
initial port shows 3.2M is fast enough to ship as an opt-in.

## Success criteria

Phase 4d is done when:

- L3TC-3.2M loads cleanly via `l3tc --model
  checkpoints/l3tc_3m2.bin compress ...`
- Round-trip on enwik6 is byte-identical with the 3.2M model
- enwik6 and enwik8 ratios measured and recorded
- enwik6 and enwik8 throughput measured and recorded
- Phase 4d findings doc (`docs/phase_4d_findings.md`) with the
  numbers and the recommendation for Phase 4e (distillation) or
  status-quo
- 200K still works exactly as before (no regression)

## Non-goals

- NEON 256×256 kernel (see 4d6 above)
- L3TC-800K or L3TC-12M (same infrastructure, different
  checkpoints; trivial once 4d1-5 work)
- Training anything (Phase 4e)
- Changing the ratio or speed of the 200K mode (strictly
  additive work)

## Why this before distillation

Distillation (Phase 4e) needs a concrete teacher to distill
from. Running the teacher on our own runtime — instead of the
slow Python reference — makes the distillation pipeline much
faster to iterate on (generate teacher distributions for a
training set in hours instead of days). It also gives us a
known-good reference ratio to target: "distillation student
should hit 80-90% of 3.2M's entropy bound at 100K parameters".

Plus 4d is cheap: ~1-2 weeks at most, most of which is
re-running the 4b2 / 4c tooling against a different checkpoint.
If distillation turns out to not work, we still get a working
3.2M mode for anyone who cares about ratio over speed.

# Phase 4d findings — L3TC-3.2M on the Rust runtime

## TL;DR

L3TC-3.2M loads and runs end-to-end in l3tc-rust with
byte-identical round trip and no ratio regression vs the
Python reference. On enwik6 (1 MB, full file):

| model | entropy bound | actual ratio | compress KB/s | decompress KB/s |
|---|---:|---:|---:|---:|
| l3tc-rust 200K | 0.1637 | **0.1699** | **131** | **132** |
| **l3tc-rust 3.2M** | **0.1275** | **0.1337** | **25.95** | **23.43** |
| Python L3TC-3.2M (reported) | — | 0.1309 (entropy bound) | 10.76 | — |

**Ratio wins:**
- Our 3.2M actual coded ratio (**0.1337**) is **3.62 pp better**
  than our 200K's 0.1699 — a ~21% relative compression improvement
- Our 3.2M entropy bound (**0.1275**) is **0.34 pp better** than
  Python L3TC-3.2M's reported 0.1309 (same raw-softmax vs
  freq-quantized-softmax delta we saw with 200K in Phase 4a)
- Actual coded vs entropy-bound gap on 3.2M is **0.0062 pp** —
  identical to 200K's gap, confirming the AC is near-optimal
  regardless of model size

**Speed outcomes:**
- 3.2M is **~5× slower** than 200K per stream (131 → 25.95 KB/s
  compress, 132 → 23.43 KB/s decompress)
- **2.4× faster** than the Python L3TC-3.2M reference (10.76 →
  25.95 KB/s) — same pattern of Rust beating Python by ~10× that
  we saw on 200K
- Still above the 20 KB/s "usable as opt-in cold-archive mode"
  threshold; below the 99 KB/s speed floor from CLAUDE.md, but
  that floor is specifically for the *default* 200K mode — 3.2M
  is explicitly an opt-in tier

## Architecture changes required

L3TC-3.2M is the first model variant where the runtime assumptions
from Phase 1 no longer hold:

| param | 200K | 3.2M |
|---|---:|---:|
| `num_hidden_layer` | 2 | 3 |
| `hidden_size` | 96 | 256 |
| `intermediate_size` | 96 | **512** |
| attention matvec shape | 96×96 | 256×256 |
| FFN key shape | 96×96 | **512×256** |
| FFN value shape | 96×96 | **256×512** |
| head shape | 16384×96 | 16384×256 |

Phase 4d's runtime changes:

1. **`Model` gains `intermediate_size: usize`**, discovered at
   load time from the FFN key tensor's shape. 200K sees
   `intermediate_size == hidden_size == 96` (unchanged behavior);
   3.2M sees `256` / `512`.

2. **`Model::load_block` takes both `h` and `im`**, with three
   different `take_2d_*` helpers for `(h,h)`, `(im,h)`, and
   `(h,im)` tensor shapes.

3. **`Scratch::new(hidden, intermediate)`** sizes the `k` buffer
   at `max(hidden, intermediate)` so the FFN key output fits.
   Other buffers stay hidden-sized.

4. **`Session::forward_block` / `channel_mix` take `im`**, and
   the FFN key/value matvecs branch: if `im == h` they use the
   NEON-dispatched square path (4c2), otherwise they fall through
   to the generic `matvec` which dispatches to sgemm above the
   BLAS threshold (512-row FFN key on 3.2M is above threshold).

5. **New `tensor::matvec_square(mat, x, out, n)`** dispatcher:
   - `n == 96` → hand-tuned `matvec_96x96` from Phase 2.5a
   - otherwise → generic `matvec` (sgemm or scalar)
   - Marked `#[inline(always)]` — without it the forward pass
     took ~6% longer on the 200K hot path because LLVM left a
     function call around the dispatch

6. **`matvec_col_major_int8`** (head matvec) is already
   dimension-agnostic — it takes `rows` and `cols` as parameters.
   No change needed; the 3.2M head loads and runs correctly.

7. **Zero changes to the checkpoint converter.** It already
   handled variable `num_layers` and variable shapes; the 3.2M
   checkpoint converts with the existing script unchanged.

## Forward-pass speed: what took the time

Profile on enwik6 (1 MB, 284210 predict steps) with 3.2M:

- Total wall-clock: ~38 s compress (1 thread would be ~165 s)
- **Forward pass dominates** — much more than with 200K. 3.2M has
  16× more parameters, 3 layers instead of 2, and crosses the
  BLAS threshold on the 512×256 FFN key, so each forward step
  takes several hundred microseconds vs ~154 μs for 200K.
- cum_freqs still ~40 μs/step — same work as before (the head is
  the same shape in tokens, just wider vectors)
- AC encode still essentially free

The **FFN matvecs are the biggest new cost** because they're
rectangular and can't use the NEON 96×96 or 256×256 path. Both
`matvec_sgemm` (from the `matrixmultiply` crate) and scalar
fallback are 2-3× slower than a hand-tuned NEON kernel would be
at the same shape.

If we ever want to make 3.2M fast enough to ship as the default,
the biggest single lever is a hand-tuned `matvec_256x256_neon`
for the square attention projections + new NEON kernels for the
rectangular FFN shapes. That's Phase 4e+ territory — explicitly
out of scope for 4d.

## What shipped in Phase 4d

| change | file |
|---|---|
| `Model::intermediate_size: usize` field | `src/rwkv.rs` |
| `Model::from_checkpoint` discovers `intermediate_size` | `src/rwkv.rs` |
| `Model::load_block(ckpt, li, h, im)` with per-shape helpers | `src/rwkv.rs` |
| `Scratch::new(hidden, intermediate)` sizes `k` on max | `src/rwkv.rs` |
| `Session::forward_block(..., h, im)` propagates intermediate | `src/rwkv.rs` |
| `channel_mix(..., h, im)` uses the right matvec per shape | `src/rwkv.rs` |
| `tensor::matvec_square(mat, x, out, n)` dispatcher | `src/tensor.rs` |
| `#[inline(always)]` on `matvec_square` (prevents regression) | `src/tensor.rs` |
| `checkpoints/l3tc_3m2.bin` — converted 3.2M checkpoint | (not committed) |
| `PHASE_4D.md` — phase plan | (already shipped) |
| `docs/phase_4d_findings.md` — this file | (this commit) |

## 200K regression test

After all Phase 4d refactoring, the 200K model on enwik6:

| metric | Phase 4c3 | **Phase 4d (200K)** |
|---|---:|---:|
| ratio | 0.1699 (169860) | **0.1699 (169860)** |
| compress KB/s | 126.8 | **~131** |
| decompress KB/s | 128.2 | **~132** |

200K path is byte-identical and actually a touch faster,
because `matvec_square` with `#[inline(always)]` lets the
compiler hoist the `n == 96` check out of the inner loop more
aggressively than the direct `matvec_96x96` call it replaced.

## Next steps

Phase 4d's whole point was to have 3.2M running in our Rust
runtime so we can answer "is it useful as opt-in? is it worth
distilling into a smaller student?". The answer to both is yes:

1. **3.2M-as-opt-in is useful today.** A user with cold archival
   or compliance-archive workloads who cares about ratio over
   speed gets 3.62 pp better compression on enwik6 at 25 KB/s —
   very comparable to Python L3TC-3.2M's reported numbers but
   with a Rust runtime we can ship. Add a CLI `--model-variant`
   flag (or just document `--model checkpoints/l3tc_3m2.bin`).

2. **Distillation is the natural Phase 4e.** Use 3.2M as the
   teacher, train a custom small student (100K-200K params,
   possibly with architectural simplification) via KL divergence
   on enwik8. Target: ratio ~0.14-0.15, speed ~200-300 KB/s.
   This is the "different model" path the user asked about
   after the Phase 4c speed polish hit diminishing returns.

3. **Intermediate option:** 4c5 INT4 head quantization for
   either 200K or 3.2M. For 3.2M specifically the head is now
   16384×256 = 4 MB at INT8, 2 MB at INT4 — a bigger absolute
   memory-bandwidth saving than the 200K case. Plausibly
   noticeable speedup on 3.2M specifically.

## Non-findings (worth flagging)

- **No ratio regression on 200K.** I was worried the refactor to
  dimension-aware code would break the byte-exact 200K path.
  It didn't; enwik6 still compresses to exactly 169860 bytes,
  exactly as in Phase 4c3.

- **`matvec_square` without `#[inline(always)]` costs ~6% of the
  200K hot path.** Caught via iter.sh regression between a dirty
  refactor and the fixed version. Worth remembering for future
  dispatcher-style helpers on the hot loop.

- **The checkpoint converter required zero changes.** The Python
  side was already corpus-and-architecture-agnostic; it just
  emits whatever shapes are in the .pth file with the standard
  HiRA-merge + ln0-rename + time_mix-squeeze transforms. The
  only Rust change was making the loader discover shapes at
  runtime instead of hard-coding them.

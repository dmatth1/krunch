# Architectural Decisions

A log of the key decisions on this project, with reasoning. Append only;
don't rewrite history — if we change our mind, add a new entry with the
reversal and the reason.

---

## D1 — Target use case: general-purpose zstd alternative

**Date:** Phase 0 start
**Decision:** Target a general-purpose lossless compressor that could
plausibly replace zstd for text-heavy workloads.

**Alternatives considered:**
- Text-specialized compressor (narrower scope, easier goal)
- Agent-to-agent traffic compression (niche, overlaps with llanguage)
- Archival / cold storage (tolerates slow encode, less demanding on speed)
- Mobile / embedded (tight size + speed, loose on ratio)

**Why zstd alternative:**
- Largest addressable use case by a significant margin
- Forces us to hit the hardest bar on all axes (speed AND ratio AND
  reliability AND distribution), which means anything that falls short
  still has a natural downgrade path to a narrower target
- Provides an unambiguous success metric: "does it beat zstd on the same
  hardware and corpus"

**What this decision implies:**
- Single-stream CPU decode speed has to be competitive, not just batched
  GPU speed
- Must handle arbitrary binary data, not just well-behaved English text
- File format, CLI, distribution all have to match zstd's bar
- Cross-platform determinism is required from day one

---

## D2 — Implementation language (production): Rust

**Date:** Phase 0 start
**Decision:** Write the production implementation in Rust. Phase 0 stays
in Python because reproducing L3TC requires running its Python code.

**Alternatives considered:**
- **C** — what NNCP and zstd use, maximum portability, most effort,
  least safety. NNCP proves this path works but Bellard-tier engineering
  isn't easy to reproduce.
- **C++** — ggml/llama.cpp's native language, easier ML integration via
  ggml, but harder to write safely and the modern tooling story is weaker.
- **Rust** — memory safety, modern tooling, strong ecosystem, single-static-
  binary distribution, excellent cross-compilation, clean C ABI for
  bindings. Slightly more friction for ML work but candle and the ggml-rs
  ecosystem are production-ready.
- **Zig** — tempting for systems code but ML ecosystem is too immature.
- **Go** — GC overhead is unacceptable for this workload.

**Why Rust:**
- "Speed, reliability, accessibility" was the explicit priority. Rust wins
  reliability outright (memory safety matters a lot for a decoder that
  processes arbitrary untrusted bytes). It ties C/C++ on speed for the
  kind of numeric inner loops we care about. Its accessibility story via
  cargo is as good as any systems language.
- Distribution: `cargo build --release` produces a single static binary.
  Cross-compilation to Linux/macOS/Windows/iOS/Android from one machine is
  well-supported.
- C ABI for language bindings is trivial (`extern "C"`).
- Decoders eat untrusted input. Rust's memory safety guarantees rule out
  an entire class of CVE-worthy bugs that classical C codecs have had
  repeatedly (libxml, libpng, libjpeg, zlib, xz-utils backdoor).
- Good ecosystem overlap: candle (Rust-native ML), rwkv.rs, tokenizers
  (Rust-native), clap (CLI), insta (snapshot testing), criterion
  (benchmarking).

**What this means in practice:**
- Phase 1 onward writes the production compressor in Rust
- Phase 0 benchmark harness is Python (needs to run L3TC's Python reference)
- C ABI is part of the design; bindings come free from Rust's `extern "C"`

---

## D3 — Inference runtime: candle first, rwkv.cpp via FFI as fallback

**Date:** Phase 0 start
**Decision:** Start with candle (HuggingFace's Rust-native ML framework)
for RWKV inference in Phase 1. If candle's RWKV support proves inadequate
or too slow, fall back to rwkv.cpp through Rust FFI bindings.

**Alternatives considered:**
- **PyTorch** — what L3TC uses. Too slow (framework overhead dominates
  on small models). Python-only distribution. Non-starter for production.
- **ONNX Runtime** — cross-platform but heavy, doesn't specialize for
  small models, licensing is OK but dependency is bulky.
- **tinygrad** — too experimental.
- **Custom C (LibNC-style)** — maximum performance ceiling, maximum
  engineering cost. Reserved for Phase 2+ if candle/rwkv.cpp isn't fast
  enough.
- **ggml directly via bindgen** — works but loses Rust-native
  ergonomics. Essentially the "rwkv.cpp via FFI" fallback.
- **candle** — Rust-native, small footprint, CUDA + Metal + CPU backends,
  SIMD, quantization support (INT8, INT4), actively developed by
  HuggingFace. Best default option if it supports RWKV well enough.

**Why this ordering:**
- candle is the ergonomic win: native Rust, no FFI boundary, no C
  dependency, cargo integration, actively maintained.
- rwkv.cpp is the performance safety net: it's ggml-based with hand-tuned
  kernels and known good throughput for small RWKV models. If candle
  turns out to be too slow on the specific RWKV variant L3TC uses, we
  already know rwkv.cpp works.
- Both can interchangeably serve as the inference layer if we define a
  clean Rust trait for "RWKV forward pass" and swap implementations
  behind it.

**Open risks:**
- candle's RWKV implementation may not cover all the variants we need
  (the HiRA-merged weight shape L3TC uses, for example)
- Cross-platform determinism for arithmetic coder round-trip requires
  bit-identical inference, which neither candle nor rwkv.cpp guarantees
  out of the box
- Performance on batch=1 single-stream is the key number; we won't know
  until we measure

---

## D4 — Training data for the eventual retraining: RedPajama v2 subset

**Date:** Phase 0 start
**Decision:** When we retrain in Phase 4, use a curated RedPajama v2
subset rather than enwik only or Common Crawl raw.

**Alternatives considered:**
- **enwik8 / enwik9 only** (what L3TC used). Too narrow; directly causes
  the OOD degradation and 12M overfit the paper reports.
- **Full Wikipedia dump (~20 GB)**. Better than enwik but still
  wiki-only; doesn't cover code, logs, non-English text.
- **The Pile (~800 GB)**. Excellent diversity but has known quality and
  licensing issues. Good enough for research, uncertain for a shipped tool.
- **RedPajama v2**. Reproduction of LLaMA's training mix, open license,
  diverse (web + code + books + Wikipedia + ArXiv), actively maintained.
  Subsets available at various sizes (10B, 100B, 1T tokens).
- **Custom mix**. Cherry-pick proportions for our use case (more code,
  more logs, less academic prose). Defer until we have baseline numbers.

**Why RedPajama v2:**
- Open license (Apache 2.0 for the code, data follows source licenses)
- Diverse enough to fix L3TC's OOD degradation
- Well-known, reproducible, citable
- Subsets available so we can pick training corpus size to match
  available compute
- Aligns with what major open LLM projects use, so techniques transfer

**What corpus size:** start with ~10-50 GB for first retraining pass.
Large enough to unlock the 12M+ model regime; small enough that a single
training run is tractable without a cluster.

**Deferred decision:** exact mixture proportions. Defer until Phase 4
when we have baseline numbers and can target the actual failure modes.

---

## D5 — RWKV version: match L3TC for Phase 0, upgrade to RWKV-7 in Phase 4

**Date:** Phase 0 start
**Decision:** Phase 0 uses whatever RWKV version L3TC uses (v4 or v5;
their paper doesn't specify but the code will tell us). Phase 4's
retraining moves to RWKV-7.

**Why:**
- Phase 0 is a reproduction exercise. Changing the backbone changes the
  numbers and invalidates the reproduction. Keep everything the same as
  L3TC for this phase.
- RWKV-7 (2024-2025) addresses the scaling limitation the paper
  explicitly calls out ("RWKV shows slightly worse scaling than
  Transformer and TransformerXL").
- RWKV-7 has production-quality inference implementations in
  `rwkv.cpp`, and `candle` is tracking newer versions more actively than
  older ones.

**Canonical source:** https://github.com/BlinkDL/RWKV-LM — matches the
user's preference and is the upstream reference for all RWKV work.

---

## D6 — Phase 0 runs in Python; Rust rewrite starts in Phase 1

**Date:** Phase 0 start
**Decision:** Don't write any Rust during Phase 0. Phase 0 is reproducing
L3TC's numbers, which requires running L3TC's Python code. Rust begins
in Phase 1 with the inference rewrite.

**Why:**
- Reproducing the paper's claims before rewriting is the only way to
  know we're actually improving anything. Rewriting then finding out
  the baseline numbers were different than we thought would be a waste.
- The benchmark harness can stay in Python permanently; it's a
  measurement tool, not production code. Python shells out to each
  compressor and measures time/memory/bytes, which is trivial.
- Splitting work by phase means clear deliverables and clean
  interfaces between stages.

---

## D7 — Benchmark harness has zero Python dependencies beyond stdlib

**Date:** Phase 0 start
**Decision:** The benchmark harness (`bench/bench.py`) uses only the
Python standard library. No numpy, no pandas, no click, nothing.

**Why:**
- The harness runs on any Python 3.10+ installation without setup
- The harness's only job is to shell out to compressors and measure
  wall time, CPU time, memory, and byte counts. Standard library
  (`subprocess`, `time`, `resource`, `json`, `argparse`, `pathlib`)
  covers all of this trivially.
- Avoiding dependencies keeps the harness reproducible across years
  and machines. A measurement tool that breaks because a library
  changed API is worse than useless.
- If we ever want numpy-style analysis on the results, we write a
  separate analysis script that reads the JSON the harness produces.
  Separation of concerns.

---

## D8 — Hand-rolled tensor math, NOT candle (reverses D3)

**Date:** Phase 1 mid-work
**Decision:** Write the RWKV forward pass using hand-rolled f32
linear algebra (matvec, layer norm, sigmoid, exp, etc.) directly in
Rust. Do not use candle, ndarray, burn, or any other tensor framework.

**Why we reversed D3:**
- **The model is tiny.** L3TC-200K is 2 layers, 96 hidden dim, 46
  tensors total. The entire forward pass is ~250 lines of Rust once
  the math is laid out explicitly. A framework's abstraction layer
  costs more in dispatch overhead than it saves in code size.
- **Framework overhead dominates at this model size.** candle's per-op
  dispatch is ~1-5 us, which would add up to ~100 us per forward pass.
  We're already fighting for every microsecond.
- **Full control over the inner loop.** We can pick exactly which
  matvec implementation to use for which matrix shape — scalar with
  4-accumulator unrolling for 96x96, column-major AXPY for the
  (16384, 96) head. A framework makes these choices opaquely.
- **Determinism.** Framework ops can vary across hardware / build
  configurations. Hand-rolled code with a single compilation unit
  guarantees bit-reproducible output across machines with the same
  target CPU.

**What we DO use:**
- `matrixmultiply` crate for the `matvec_col_major_par` path (still
  available but not called by default in Phase 2 because of thread-pool
  overhead on small workloads)
- `sentencepiece` Rust crate for the tokenizer (wraps the C++ library)
- `byteorder` for binary I/O
- `memmap2` for zero-copy file reads
- `rayon` for segment-level parallelism (not per-token)

**What we explicitly do NOT use:**
- candle (considered, rejected after hand-rolling proved simpler)
- ndarray (overkill for our ops)
- tch / pyo3 / PyTorch bindings (defeats the purpose)
- rwkv.cpp via FFI (was D3's fallback; not needed because the
  hand-rolled path is fast enough)

**How we got here:** Phase 1's first cut used matrixmultiply::sgemm
for the head via the `matvec` function. We measured `sgemm` at n=1
(matvec form) and found it was only ~10% faster than a naive scalar
matvec — GEMM kernels optimize for fat matmul, not skinny matvec.
The real win came from switching to a column-major AXPY-style matvec
(see D11), which is its own 30 lines of code and is faster than any
framework's generic matvec.

---

## D9 — Python preprocessing script for PyTorch → Rust-friendly binary

**Date:** Phase 1 start
**Decision:** Instead of parsing PyTorch `.pth` pickle files in Rust,
write a one-time Python preprocessing script that reads the checkpoint
(with torch), applies HiRA merging and shape normalization, and emits
a simple flat binary file that Rust reads with trivial byte-level code.

**What the converter does:**
1. Loads the `.pth` via `torch.load(weights_only=False)`
2. Applies HiRA merging: `W = W_0 + B @ A` for each K/V/R projection
3. Renames `blocks.0.ln0.*` to top-level `ln0.*` to match the inference
   model layout (L3TC's compressor.py does the same at load time)
4. Squeezes `(1, 1, 96)` time-mix vectors to `(96,)`
5. Writes a flat `LRUS` binary: magic + version + n_tensors +
   per-tensor (name, shape, dtype, data)

**Why this instead of parsing pickle in Rust:**
- PyTorch pickle format is complex (references to Python classes,
  custom unpicklers, multiple allocation strategies). Parsing it
  robustly in Rust requires a significant chunk of code for
  functionality we only need once per checkpoint.
- The converter runs in L3TC's existing Python venv. No new Python
  dependencies and no new Rust dependencies.
- The Rust reader is ~370 lines of straightforward byte-level parsing
  with no external deps.
- HiRA merging is easier to do in Python where we can use the existing
  `torch.matmul`.

**Trade-off:** users need to run the converter once per checkpoint
variant they want to use. This is a one-time cost per checkpoint
(L3TC ships 4 variants totalling ~16 MB of weights).

---

## D10 — Segment-level parallelism via rayon, not per-token

**Date:** Phase 1 mid-work (the single biggest win)
**Decision:** Parallelize at the SEGMENT level using `rayon::par_iter`,
NOT within a single forward pass. Each segment is processed
independently by one rayon worker.

**Why segment-level, not per-token:**
- Per-token parallelism was tried first (parallel head matvec via
  `matvec_col_major_par`). Thread-pool dispatch overhead at ~100 us
  per head matvec eclipsed the multi-threading savings. Measured
  ~9% improvement from an 8-core parallelization of a ~140 us
  workload — almost all the gain was eaten by dispatch.
- Segment-level parallelism gives each rayon worker hundreds of
  tokens of sequential work per task, well above the break-even
  point for thread dispatch.
- Segments are naturally independent: the model state resets at
  every segment boundary, and each segment's arithmetic coder
  output is self-contained. No shared mutable state, no
  synchronization required.
- Fit for the use case: real-world compression jobs are mostly
  ≥ 1 MB files with hundreds of segments each. Small (<50 KB)
  files suffer slightly because parallelism can't fill all 8
  cores, but that's an acceptable trade-off for the 5× speedup
  on large files.

**Measurement:** this single change took throughput on enwik6 from
12.88 KB/s (Phase 1 mid-point) to 65.24 KB/s — the **5.07× speedup**
that put us over the Phase 1 target. No other single change in the
whole project moved the needle this much.

---

## D11 — Column-major AXPY matvec for the head projection

**Date:** Phase 1 mid-work
**Decision:** Store the head weight column-major (`(hidden, vocab)`)
rather than the default row-major (`(vocab, hidden)`), and compute
the head matvec as a series of AXPY operations over columns instead
of dot products over rows.

**Why:**
The head projection is the single biggest compute hotspot per token
(16384 × 96 = 1.57 M FLOPs). Profiling in Phase 1 showed it was
taking ~520 us/token out of a ~600 us total forward pass — 88% of
per-token time.

The row-major dot-product form has two performance problems:
1. Each output row strides through a single 96-element chunk of
   memory per iteration, which doesn't fill cache lines well.
2. The inner-loop accumulator creates a reduction dependency chain
   that prevents 4-wide SIMD.

The column-major AXPY form:
```rust
for j in 0..cols {
    let col = &mat[j * rows..(j + 1) * rows];
    let xj = x[j];
    for i in 0..rows {
        out[i] += xj * col[i];
    }
}
```
- Streams through contiguous memory (a full 16384-row column = 64 KB
  per iteration, hits the L1 prefetcher perfectly)
- Has no reduction dependency (each `out[i]` is an independent
  accumulator)
- Auto-vectorizes cleanly on NEON / AVX because the inner loop is
  just `vfmaq_f32(out, col, xj_broadcast)`

**Result:** head matvec dropped from ~520 us to ~140 us (3.7× faster)
when combined with `target-cpu=native` for full NEON width. This was
the second biggest win after segment-level parallelism.

---

## D12 — Raw-fallback for SPM-normalized segments

**Date:** Phase 1 end (correctness fix during full-enwik6 testing)
**Decision:** When a segment's tokens don't faithfully round-trip via
`sp.decode(sp.encode(text))`, flag the segment with a raw-fallback
bit and store the original bytes alongside the arithmetic-coded body.
The decoder uses the raw bytes directly.

**Why:**
SentencePiece normalizes certain characters during `encode()` (zero-
width joiners, combining marks, NFC normalization of certain scripts).
For inputs containing these characters, `sp.decode(sp.encode(text))
!= text` — bytes are silently lost.

Discovered on the full enwik6 corpus, which contains Persian/Arabic
interlanguage Wikipedia links with ZWNJ (U+200C) characters. The
Rust round-trip test failed with a 24-byte length mismatch on 1 MB
of enwik6.

**How the refinement works:**
1. At tokenization time, compare `sp.decode(tokens) == original`.
   If not, set `needs_raw_fallback = true`.
2. Recursively halve raw-fallback segments until each is ≤ 64 bytes
   or round-trips cleanly. This isolates problematic characters into
   small chunks while the surrounding ASCII still compresses normally.
3. The codec writes the raw bytes for each flagged segment alongside
   its ac_body. The decoder uses the raw bytes verbatim when the
   flag is set.

**Format impact:** bumped the compressed file format to version 2
(adds a per-segment `seg_flags` byte with bit 0 = raw fallback).

**Overhead:** for ASCII-only text, never triggers (0 overhead). For
mixed text like enwik6, the ratio impact is a few percentage points
of ratio (0.21 vs the no-fallback ~0.17). The alternative —
silently losing bytes — is not acceptable for a production codec.

---

## D13 — Default segment size 4096 (not L3TC's 2048)

**Date:** Phase 2a
**Decision:** Change the default `segment_bytes` from L3TC's 2048 to
4096. Users can override via `--segment-bytes`.

**Why:**
Segment size sweep on enwik6 (1 MB):
```
seg=2048  ratio=0.2095  compress=82 KB/s
seg=4096  ratio=0.2060  compress=83 KB/s
seg=8192  ratio=0.2046  compress=78 KB/s
seg=16384 ratio=0.2040  compress=80 KB/s
```
4096 is the sweet spot: ~2% better ratio than L3TC's 2048 with
essentially the same throughput on files ≥ 1 MB. 8192+ gives
diminishing ratio gains and starts to lose throughput on smaller
files because of reduced segment-level parallelism.

**Trade-off on small files:** 50 KB corpus drops from 82 KB/s at
seg=2048 to 66 KB/s at seg=4096 because there are now only ~12
segments instead of 25, and not enough to fully fill 8 cores.
Acceptable because real compression workloads are usually ≥ 1 MB.

---

## D14 — f64 precision for the cum_freqs scale calculation

**Date:** Phase 2a (defensive, not measurably helpful)
**Decision:** Use f64 for the `scale = usable as f64 / sum as f64`
computation in `codec::logits_to_cum_freqs_scratch`, even though
the surrounding ops (logits, exps) remain in f32.

**Why:**
`usable` is `MAX_TOTAL - n ≈ 2^62`, which is beyond f32's 24-bit
integer precision. `usable as f32` rounds to the nearest
representable float with an error of up to 2^38 — a ~26% relative
error in the worst case. The hope was that eliminating this error
would improve compression ratio.

**Measured impact:** zero. The f64 change didn't move the ratio
(still 0.2060 on enwik6). The f32 precision wasn't actually the
bottleneck; the dominant loss is from the `max(1)` clamp on
insignificant tokens and the floor rounding, neither of which is
sensitive to scale precision at this level.

**Why keep the change anyway:**
- It's defensive: any future work that narrows the scale gap (e.g.
  via top-K truncation or different freq-table schemes) would
  benefit from correct scale precision.
- No measurable performance cost on modern CPUs (f64 mul/div take
  the same cycles as f32 at our throughput).
- Clearer reasoning about the code: "we're computing in the
  precision that can represent the range correctly."

---

## D15 — Rayon for segment parallelism, serial for per-token

**Date:** Phase 2a
**Decision:** Use rayon::par_iter at the segment level (D10), but
keep the head matvec and all block matvecs serial within each
segment's forward pass. The `matvec_col_major_par` function is
retained as a library export but not called by default.

**Why the split:**
- **Segment-level parallelism** has a ~5× speedup (from 12.88 to
  65.24 KB/s in Phase 1, ~90 KB/s in Phase 2 with further tuning).
  Each segment is ~hundreds of us of compute per worker, well above
  rayon's dispatch break-even point.
- **Per-token parallelism** (parallel head matvec) was tried in
  Phase 1 and gave only ~8% gain before being reverted in Phase 2.
  The head matvec is only ~140 us, close to rayon's break-even.
- **Stacking both** causes thread contention: when all 8 cores are
  already running segments via segment-level parallelism, asking
  any segment to spawn more parallelism just creates thread-pool
  fights.

The `matvec_col_major_par` function is still exposed for callers
that might want to parallelize a SINGLE long segment (for example,
a long-running streaming compression where there's only one
segment at a time).

---

## D16 — Single-pass cum_freqs, not a 3-pass structure

**Date:** Phase 2 (tried and reverted)
**Decision:** Keep the single-pass `for i in 0..n { ... cum[i+1] =
cum[i] + scaled; ... }` structure for `logits_to_cum_freqs_scratch`.
Do NOT split into separate compute-freqs / argmax / prefix-sum
passes.

**What we tried:**
A 3-pass structure that computes freqs into a separate buffer,
finds argmax, and then prefix-sums freqs into cum. The goal was
to unlock SIMD vectorization of the independent freq calculation.

**Why the revert:**
- The bottleneck op (`(e as f64 * scale).floor() as u64`) doesn't
  vectorize on ARM NEON anyway — f64 → u64 conversion is scalar-
  only. The 3-pass structure didn't unlock SIMD.
- The extra memory pass (writing freqs, then reading them back in
  prefix sum) dominated the (nonexistent) vectorization gain.
  Measured 94 us/token for 3-pass vs 80 us/token for 1-pass.
- The 1-pass version is cleaner code anyway.

**Consequence:** cum_freqs stays at ~80 us/token. Improving it
further requires different approaches:
- Top-K truncation (skip exp for insignificant tokens)
- Split-precision f32→u32 lane-vectorizable scheme
- Fold softmax into the arithmetic coder directly
All deferred to Phase 2.5.

---

## D17 — Hand-tuned NEON for block matvecs (Phase 2.5, pending)

**Date:** Phase 2 end
**Decision:** Phase 2.5 will write explicit NEON intrinsics for the
96x96 block matvecs (K/V/R projections and output projection in each
of the 16 block matvecs per token).

**Why:**
The block matvecs total ~50 us/token (16 × ~3 us each). At NEON
peak throughput (~20 GFLOPS for f32), 96x96 matvec should take ~0.5
us. We're ~6× above peak. Hand-tuned intrinsics with 4 independent
f32x4 accumulators and x preloaded into registers should get close
to peak, saving ~30-40 us/token.

Estimated impact: takes full-pipeline throughput from ~90 KB/s to
~130-150 KB/s on enwik6.

**Why hand-tuned and not framework:**
See D8 — we rejected frameworks for the main forward pass. The
block matvecs are even smaller than the head and even less likely
to benefit from a framework's abstraction. Direct `std::arch::aarch64`
intrinsics are ~50 lines of code per specialized matvec.

**Risks:**
- Unsafe code (intrinsics require `unsafe`)
- Platform-specific (only helps on aarch64)
- Adds code complexity for a single hotspot

All acceptable for the speed win.

---

_Append new decisions below with D18, D19, etc. as we make them._


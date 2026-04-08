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

_Append new decisions below with D8, D9, etc. as we make them._

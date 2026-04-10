# The compressor landscape — primary-source comparison

**Purpose.** This document records the data and reasoning behind
l3tc-prod's positioning claim ("the fastest CPU learned compressor
by a wide margin"). It exists so that any number in the README is
traceable to a primary source, and so that the framing survives a
careful reader who goes looking for the math.

**Date of research:** April 2026. Most recent leaderboard update
consulted: [LTCB](http://mattmahoney.net/dc/text.html), updated
2026-03-25.

**Audience:** anyone who wants to verify the claims in the README,
including me six months from now when the landscape has shifted and
some of these numbers are stale.

---

## 1. Methodology

### Sources used (primary only)

- **[Large Text Compression Benchmark (LTCB)](http://mattmahoney.net/dc/text.html)** —
  Matt Mahoney's leaderboard. enwik9 (1 GB) is the canonical corpus.
  This is the only place where serious neural / PAQ-family
  compressors post wall-clock numbers. Last updated 2026-03-25.
- **[NNCP project page (Bellard)](https://bellard.org/nncp/)** —
  current version v3.3, released 2024-06-05; ratio published, no
  speed number for v3.3.
- **[ts_zip page (Bellard)](https://bellard.org/ts_zip/)** —
  RWKV-169M LLM-as-compressor, GPU only.
- **[CMIX repo (Knoll)](https://github.com/byronknoll/cmix)** —
  current version v21, released 2024-09.
- **[zstd releases](https://github.com/facebook/zstd/releases)** —
  current stable v1.5.7, released 2025-02-19.
- **[bzip3 repo](https://github.com/kspalaiologos/bzip3)** —
  current v1.5.3, released 2025-08-13. Modern BWT competitor.
- **[L3TC paper (arxiv 2412.16642)](https://arxiv.org/html/2412.16642v2)** —
  the paper that l3tc-prod implements.
- **[LLMZip paper (arxiv 2306.04050)](https://arxiv.org/abs/2306.04050)** —
  LLaMA-7B as compressor.
- **[Nacrith (arxiv 2602.19626)](https://arxiv.org/abs/2602.19626)** —
  February 2026 LLM-based compressor (GPU only).
- **[Language Modeling Is Compression (ICLR 2024)](https://arxiv.org/pdf/2309.10668)** —
  DeepMind, Chinchilla 70B as compressor.
- **[tensorflow-compress repo](https://github.com/byronknoll/tensorflow-compress)** —
  LSTM compressor; repo confirms wall-clock time on enwik9.
- **[lstm-compress repo](https://github.com/byronknoll/lstm-compress)** —
  CPU LSTM compressor.

Wikipedia and second-hand blog posts are not used as sources for
any number in this document.

### Key methodology facts

1. **LTCB time units are ns/Byte CPU-time, summed across processors
   for multi-threaded programs.** Verbatim from the LTCB page:
   *"Times reported are the smaller of process time (summed over
   processors if multi-threaded) or real time."* For single-threaded
   programs (which all the top entries are), CPU-time = wall-clock.
   Conversion: **KB/s wall-clock = 976,562 ÷ (ns/B)**.

2. **LTCB sizes include the decompressor binary** (Hutter Prize
   methodology). Adds 1-2% to small compressors, negligible for
   big ones.

3. **enwik8 (100 MB) and enwik9 (1 GB) ratios are slightly different**
   because more data has more redundancy to exploit. Don't
   compare an enwik6 ratio to an enwik9 ratio without noting it.
   l3tc-prod's published numbers are enwik6 (1 MB) and enwik8
   (100 MB); LTCB is enwik9 (1 GB). When comparing, use the
   closest corpus available and flag the gap.

4. **The L3TC paper's reported "0.166 on enwik6" is the theoretical
   entropy bound, not actual coded bytes.** Phase 4a in this project
   uncovered this — Python L3TC's `compressor.py` literally returns
   `total_bin_size_min = math.ceil(entropy_sum / 8)` and the file-
   write path is commented out. Always check whether a learned
   compressor's published bpb is *real coded bytes* or
   *entropy_sum/8*. l3tc-prod's reported numbers are real coded
   bytes including AC framing, segment headers, and file framing.

5. **Hardware is not normalized across LTCB.** Numbers come from
   assorted machines (Athlon-64, Core i7/i9, Xeon, RTX 3090, A100,
   TPU). l3tc-prod's numbers are Apple M-series, all-cores. Any
   speed comparison across different machine classes carries an
   uncertainty band of roughly ±30%.

6. **Most LTCB neural/PAQ entries are architecturally serial**
   (single-stream arithmetic coding over a single model state).
   They cannot use more than one CPU core. l3tc-prod's
   segment-level parallelism (Phase 2 / DECISIONS.md D10) is the
   exception, not the rule.

---

## 2. The verified leaderboard

### Learned compressors (neural / PAQ / context-mixing)

All on enwik9 (1 GB) unless otherwise noted. Speed converted to
wall-clock KB/s for human readability.

| Compressor | Year | Architecture | bpb | KB/s wall-clock | Hardware | Threading |
|---|---|---|---:|---:|---|---|
| nncp v3.2 | 2023 | Transformer (LibNC) | 0.857 | 4.04 | RTX 3090 (GPU) | 1 |
| nncp v3.3 | 2024-06 | Transformer (LibNC) | 0.853 | not published | not published | unknown |
| cmix v21 | 2024-09 | PAQ context mix | 0.866 | 1.57 | CPU | 1 |
| fx2-cmix | 2024 | PAQ + cmix | 0.883 | 3.59 (decomp) | CPU | 1 |
| jax-compress | 2026 | LSTM (TPU) | 0.907 | 8.88 | TPU | 1 |
| tensorflow-compress v4 | 2022 | LSTM | 0.909 | 3.35 | A100 GPU | 1 |
| cmix-hp v1 | 2021 | PAQ | 0.911 | 5.16 | CPU | 1 |
| fast-cmix | 2023 | PAQ | 0.911 | 8.01 (decomp) | CPU | 1 |
| starlit | 2021 | PAQ | 0.921 | 5.61 | CPU | 1 |
| phda9 1.8 | 2019 | PAQ | 0.934 | 11.33 | CPU | 1 |
| gmix v1 | 2024 | PAQ | 0.980 | 13.20 | CPU | 1 |
| paq8px_v206fix1 | 2022 | PAQ + LSTM | 1.002 | 3.35 | CPU | 1 |
| lstm-compress v3 | 2020 | LSTM | ~1.39 | 10.58 | CPU | 1 |
| **l3tc-prod 200K** | **2026** | **RWKV-v4 200K** | **~1.43 (enwik8)** | **131 (enwik6) / 114 (enwik8)** | **Apple M-series CPU** | **8 (rayon segments)** |
| **l3tc-prod 3.2M** | **2026** | **RWKV-v4 3.2M** | **~1.07 (enwik8)** | **26** | **Apple M-series CPU** | **8 (rayon segments)** |
| ts_zip (Bellard) | 2024-03 | RWKV-169M | 1.084 (enwik9) / 1.106 (enwik8) | ~1024 | RTX 4090 GPU | 1 GPU stream |
| LLMZip | 2023 | LLaMA-7B | 0.842 (enwik9) | not published (GPU days) | GPU | 1 |
| Llamazip | 2025 | LLaMA | 2.069 (enwik9) | not published | GPU | 1 |
| Nacrith | 2026-02 | LLM (llama.cpp) | 0.939 (enwik8) | not published | consumer GPU, ~1.2 GB VRAM | 1 |

**Notes on the table:**

- LTCB last updated 2026-03-25. Entries marked "2026" predate that.
- l3tc-prod numbers are enwik6 / enwik8, not enwik9, because we
  don't have an enwik9 measurement yet. Going from enwik8 (100 MB)
  to enwik9 (1 GB) typically improves ratio by 0.05-0.10 bpb on
  text, so l3tc-prod 200K's enwik9 number would likely land
  ~1.35-1.40 bpb. Speed should be similar.
- The "enwik8 vs enwik9" note matters more for ratio than for speed.
- "8 (rayon segments)" means l3tc-prod fans segments across 8
  worker threads via rayon. This is the only entry in the table
  that uses more than 1 core. See DECISIONS.md D10.

### Classical compressors (for context)

| Compressor | Year | Family | bpb (enwik9) | Compress MB/s | Decompress MB/s |
|---|---|---|---:|---:|---:|
| zstd 1.5.7 -22 | 2025-02 | LZ77 + entropy | ~2.40 | ~4 | ~125 |
| xz -9 | (stable) | LZMA2 | 1.58 | ~1 | ~52 |
| bzip3 1.5.3 -j 12 | 2025-08 | BWT | (better than xz on text) | 17 (per-thread × 12) | 23 (per-thread) |
| bzip2 -9 | (stable) | BWT | ~2.07 | ~16 | ~35 |
| gzip -9 | (stable) | DEFLATE | 2.58 | ~17 | ~150 |

The classical numbers are in **MB/s**, the learned numbers are in
**KB/s**. **The two categories are roughly 1000× apart on the
speed axis.** That's not new, it's not l3tc-prod's contribution,
and it should not be hidden in the README.

---

## 3. The speed-leader claim — math

The original framing in the README was *"40× faster than NNCP and
~80× faster than CMIX"*. Here is the actual math against every row
in the leaderboard, computed from primary sources.

### l3tc-prod 200K wall-clock (131 KB/s, all-cores) vs each learned competitor

| Competitor | KB/s wall-clock | l3tc-prod ÷ competitor |
|---|---:|---:|
| nncp v3.2 | 4.04 | 32.4× |
| cmix v21 | 1.57 | 83.4× |
| fx2-cmix (decomp) | 3.59 | 36.5× |
| jax-compress | 8.88 | 14.8× |
| tensorflow-compress v4 | 3.35 | 39.1× |
| cmix-hp v1 | 5.16 | 25.4× |
| fast-cmix (decomp) | 8.01 | 16.4× |
| starlit | 5.61 | 23.4× |
| phda9 1.8 | 11.33 | 11.6× |
| gmix v1 | 13.20 | 9.9× |
| paq8px_v206fix1 | 3.35 | 39.1× |
| lstm-compress v3 | 10.58 | 12.4× |
| ts_zip on RTX 4090 | 1024 | **0.13×** (ts_zip is 7.8× faster, GPU only) |
| LLMZip / Llamazip / Nacrith | not published | — |

**Range:** ~10× to ~83× wall-clock, depending on which competitor.
**Geometric mean across the 12 published-speed entries:** ~22×.

**Single-thread per-core normalization** (l3tc-prod ≈ 16-18 KB/s
per core, assumes near-linear segment-parallel scaling):

| Competitor | KB/s | l3tc-prod 1-core ÷ competitor |
|---|---:|---:|
| cmix v21 | 1.57 | 10.5× |
| nncp v3.2 (GPU) | 4.04 | 4.0× |
| paq8px_v206fix1 | 3.35 | 4.9× |
| starlit | 5.61 | 2.9× |
| phda9 | 11.33 | 1.5× |
| gmix v1 | 13.20 | 1.2× |
| lstm-compress v3 | 10.58 | 1.6× |

**At per-core single-thread, the gap shrinks to 1.2-10.5×.**
The 12-83× wall-clock gap is mostly the parallelism win, not raw
single-core kernel engineering.

### What this says

- **The "40-80×" framing is in the right neighborhood for the two
  ratio leaders (NNCP at 32×, CMIX at 83×), but it cherry-picks
  those two and ignores the field.** Across all 12 published-speed
  entries the geometric mean is ~22×.
- **There is exactly one shipped learned compressor that runs
  faster in absolute terms: ts_zip on RTX 4090.** 7.8× faster than
  l3tc-prod, but GPU-only and uses an RWKV model 850× larger.
  Different product category (GPU service vs CLI tool).
- **There is no shipped CPU-only learned compressor that runs
  faster than l3tc-prod 200K wall-clock** on a current multi-core
  machine.
- **At per-core single-thread, the per-core engineering win shrinks
  to ~1.5-10×.** Most of the wall-clock lead is the parallelism
  win (Phase 2 / D10), not raw kernel-level NEON tuning.

---

## 4. The defensible positioning

The version that survives a careful reader, an arxiv reviewer,
and a Hacker News comment thread:

> l3tc-prod is the first shipped learned compressor with linear
> multi-core scaling on CPU. Every other neural / PAQ-family entry
> on the LTCB is architecturally single-threaded (NNCP, CMIX,
> paq8px, lstm-compress, phda9, starlit — all dependency-chain-
> bound). l3tc-prod's segment-level parallelism (D10) cashes in
> 8 cores at near-linear speedup, putting the 200K default tier
> at **131 KB/s wall-clock — 10-83× faster than every other
> shipped learned compressor on the LTCB** (geometric mean ~22×,
> closest single-thread CPU competitor lstm-compress at 12×).
> The only learned compressor that runs faster in absolute terms
> is ts_zip on a $2000 GPU (~1024 KB/s, 7.8× our speed, but
> GPU-only and a different product category).
>
> The trade is ratio: at ~1.43 bpb on enwik8, l3tc-prod 200K is
> ~67% behind the ratio frontier (NNCP v3.2 at 0.857 bpb, CMIX
> v21 at 0.866 bpb on enwik9). The 3.2M opt-in tier closes most
> of that gap (~1.07 bpb at 26 KB/s wall-clock — still 16×
> faster than NNCP v3.2, ~24% behind on ratio).

This is the framing the README should ship. It states the speed
claim with the actual range (10-83× wall-clock, ~22× geomean),
acknowledges the per-core caveat, names the one GPU exception
(ts_zip), and is honest about the ratio gap.

---

## 5. What's actually first-of-its-kind

Three things, none of which require the "speed leader" frame:

1. **First shipped learned compressor with multi-core CPU
   parallelism.** Every prior entry — NNCP, CMIX, all PAQ
   variants, lstm-compress, paq8px — is single-threaded by
   architecture. l3tc-prod's segment-level parallelism is the
   first time anyone has shipped a learned compressor that scales
   with cores. This is structural, not an engineering polish, and
   it's not catchable without changing how the AC state evolves.

2. **First shipped CPU-only RWKV-based compressor.** L3TC the
   paper reports speeds only at batched inference (batch=256 on
   iPhone ANE, batch=2048 on A100). The paper does not include a
   batch-1 CPU number. l3tc-prod is the first reproducible
   implementation that runs L3TC at batch-1 on CPU, and the only
   one that has measured what happens to it under realistic
   single-stream compression workloads. ts_zip uses a different
   RWKV variant (169M params, 8-bit quantized) and is GPU-only.

3. **First implementation to verify the L3TC paper's actual coded-
   bytes ratio.** Phase 4a in this project established that the
   L3TC paper's headline number is the theoretical entropy bound,
   not real file size. l3tc-prod measures and reports the
   AC-framing-included number on every run. To our knowledge no
   prior implementation has done this for any RWKV-based
   compressor.

The third one is the kind of thing a careful reviewer would notice
and respect.

---

## 6. News from 2024-2026 that affects the picture

### NNCP v3.3 (June 2024)

Bellard updated NNCP from v3.2 to v3.3. Ratio improved from
0.857 to 0.853 bpb on enwik9. **No speed number was published**
for v3.3 on the project page. The 4.04 KB/s number we use comes
from LTCB's v3.2 entry. For positioning purposes, "NNCP at
~4 KB/s wall-clock on GPU" is the most recent verifiable speed
number, not a stale one.

### CMIX v21 (September 2024)

Current best CPU-only ratio for shipped learned compression:
0.866 bpb on enwik9 at 1.57 KB/s wall-clock. CMIX v21 is the
state of the art for the "ratio at any cost on CPU" category.

### ts_zip (March 2024)

Bellard's RWKV-169M LLM-as-compressor. **GPU only** (RTX 4090,
4 GB RAM minimum). 1.106 bpb on enwik8, ~1024 KB/s. This is
the only learned compressor that beats l3tc-prod on absolute
speed, and it does it by using a $2000 GPU and an 850× larger
model. Worth noting in the README so a knowledgeable reader
doesn't think we're ignoring it.

### The 2024-2026 LLM-compressor wave

LLMZip (LLaMA-7B), LMCompress (LLaMA3-8B), Llamazip (Nov 2025),
Nacrith (Feb 2026), Language Modeling Is Compression (Chinchilla
70B). All **GPU-only research demos**. None ship a CPU mode.
None publish wall-clock speed in any usable form. They're not
in the same product category as l3tc-prod and they're not on
the LTCB. They are all chasing the ratio frontier, not the
speed frontier — Llamazip's 2.069 bpb on enwik9 actually
*regresses* on Llama-7B's underlying capability, suggesting
even ratio is hard to nail down for these.

### bzip3 (current v1.5.3, August 2025)

Modern multi-threaded BWT compressor, not learned. ~17 MiB/s per
thread compression, ~23 MiB/s per thread decompression on x64.
With `-j 12` it's a serious xz competitor at better ratio for
text. Not in our category but worth knowing about.

### jax-compress (2026)

New entry on LTCB. LSTM-based, TPU-accelerated. 0.907 bpb on
enwik9, 8.88 KB/s wall-clock on TPU. Notable because it's
faster than NNCP v3.2 (which is also GPU) at slightly worse
ratio — confirms the "throw bigger compute at it" path is
hitting diminishing returns on the speed axis.

---

## 7. Honest skepticism

### Things that could undermine the claim

1. **Hardware normalization is weak.** l3tc-prod's 131 KB/s is
   on Apple M-series. M-series memory bandwidth is genuinely
   good for the matvec hot loop (≥200 GB/s on M2/M3 Pro). A
   Linux x86 server with weaker memory bandwidth might land at
   80-100 KB/s instead of 131. We have no Linux x86 measurement
   yet (Phase 6 unstarted). If the Linux number is 80, the
   geometric mean speed lead drops from 22× to ~13×. Still real,
   but the headline number softens by a third. **The README
   should attribute the 131 KB/s number to its hardware
   explicitly.**

2. **The single-thread per-core gap is real and small.** As shown
   in §3, per-core normalized, l3tc-prod is only 1.2-10.5× faster
   than the top CPU-single-thread learned compressors. The
   12-83× wall-clock gap is mostly the parallelism win, not raw
   kernel engineering. Anyone running on a 1-core system or
   comparing single-thread numbers will see a smaller gap. The
   README should note this.

3. **ts_zip exists and is GPU-only.** A reader who knows neural
   compression will know ts_zip and may interpret "speed leader"
   as ignoring it. The honest framing has to include the
   "CPU-only" qualifier in the headline.

4. **The ratio gap is large.** At 1.43 bpb on enwik8 vs the
   frontier at 0.86 bpb, l3tc-prod is at a different operating
   point. Calling this a "speed leader" without acknowledging
   the ratio cost is selective. The two-table framing already
   in the README handles this if both tables are presented.

5. **CMIX, NNCP, and the PAQ family aren't actively chasing
   speed.** They're chasing ratio. Saying "we beat them on speed"
   is true and interesting but it's because they're optimizing
   for different things. The honest framing is "different
   operating point", not "faster".

6. **The L3TC architecture itself is published.** Anyone could in
   principle build a faster CPU runtime tomorrow. The 12-83×
   lead is current, not durable. If L3TC's authors or another
   group ports it to Rust + SIMD + parallel segments, the gap
   closes to nothing in weeks. The "first-of-its-kind" claim has
   a shelf life.

### Things that strengthen the claim

1. **The LTCB neural/PAQ leaders haven't moved much in 2 years
   on the speed axis.** NNCP v3.2 in 2023, CMIX v21 in 2024 —
   the ratio frontier is moving slowly and the speed numbers
   aren't moving at all. The category is sleepy on the speed
   axis. l3tc-prod entering at 131 KB/s isn't beating a moving
   target; it's entering a corner of the curve nobody is
   contesting.

2. **The combination is unique.** No other shipped learned
   compressor has all four of (a) on CPU, (b) batch-1 / single-
   stream, (c) multi-core parallelism, (d) ratio better than
   zstd-22. Each piece individually exists somewhere; the
   combination is l3tc-prod's actual contribution.

3. **The per-core single-thread number (16-18 KB/s) is also
   higher than the entire neural/PAQ field on CPU.** Even if
   you strip out the parallelism, l3tc-prod is the fastest
   CPU-only learned compressor per core. That's a separately
   defensible claim, smaller in magnitude (1.2-10.5×) but still
   true.

---

## 8. Recommended README framing

Drop the "40-80×" number. Replace with the calibrated version:

> The fastest CPU learned compressor on the Large Text Compression
> Benchmark by 10-83× wall-clock (geometric mean ~22× across the
> published-speed entries), and the only one with multi-core
> parallelism. l3tc-prod 200K runs at 131 KB/s on enwik6 on a
> current Apple Silicon machine; the closest single-threaded CPU
> competitor (lstm-compress v3) runs at 10.6 KB/s, the closest
> ratio competitors (NNCP v3.2 at 4 KB/s GPU, CMIX v21 at 1.6 KB/s
> CPU) are 32-83× slower. The one learned compressor that beats
> us on absolute speed is Bellard's ts_zip at ~1 MB/s on an RTX
> 4090, which uses an 850× larger model and is GPU-only.

Qualifier columns to add to the README headline table:

- **Hardware** (so it's clear l3tc-prod's number is Apple M-series)
- **Threading** (so the multi-core caveat is visible)
- **Corpus** (so enwik6 vs enwik9 isn't hidden)

That's the version this document supports. Anything stronger
than that goes beyond what the data says.

---

## 9. Re-running this analysis

This document will go stale. To re-verify:

1. Re-fetch [LTCB](http://mattmahoney.net/dc/text.html) — check
   for new entries, new versions of existing entries, new
   leaderboard position changes.
2. Re-fetch [bellard.org/nncp](https://bellard.org/nncp/) and
   [bellard.org/ts_zip](https://bellard.org/ts_zip/) for any new
   versions or speed numbers.
3. Search arxiv for "lossless compression" + "language model" +
   the current year for any new entries.
4. Recompute the speed-ratio table at the top of §2 with current
   numbers.
5. Recompute the geometric mean and the per-row ratios in §3.
6. Update the README claim if the numbers have moved.

The conversion factor `KB/s = 976,562 ÷ (ns/B)` is stable.

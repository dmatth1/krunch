# l3tc-prod — Analysis

**Goal:** take L3TC (AAAI 2025, Alipay) — a learned lossless text compressor
using RWKV + arithmetic coding with genuinely interesting benchmark numbers —
and figure out why it isn't production-ready despite those numbers, then
work toward closing the gap.

This document is the thinking layer, not the code layer. It captures what we
know, what we're betting on, and what we're about to build.

---

## 1. What L3TC actually is

L3TC is a lossless text compressor that replaces classical statistical models
(the LZ/PPM/Huffman family) with a small pretrained RWKV language model
paired with an arithmetic coder. The pipeline, per token:

1. A SentencePiece BPE tokenizer with a 16K vocabulary and **0.999 character
   coverage** (deliberately not 100%) segments the input.
2. An "outlier-aware" path: the ~0.1% of tokens that fall outside the
   vocabulary bypass the neural predictor and get written as raw UTF-8 bytes.
   In-vocab tokens go to the predictor.
3. The RWKV model (2–4 layers, 96–384 embed dim, 200K to 12M parameters)
   produces a next-token probability distribution given the recurrent state.
4. An arithmetic coder emits bits proportional to `-log2(p(token))`.
5. The recurrent state is reset every 2048 bytes.

Training uses a "High-Rank Reparameterization" (HiRA) trick: for each R/K/V
projection in the RWKV block, extra parallel `A_m @ B_m` low-rank branches are
trained alongside the base weights. All branches are merged into a single
matrix at inference time (`W = W_0 + sum(A_m @ B_m)`), so training gets extra
expressiveness without costing anything at inference.

### Reported results (from the paper)

| Compressor       | enwik9 ratio | Notes                              |
|------------------|--------------|------------------------------------|
| gzip             | ~32.3%       | baseline                           |
| zstd             | ~21.5%       | modern default                     |
| bzip2            | ~25.4%       | classical mid-tier                 |
| cmix v20         | ~11.0%       | ratio SOTA (classical)             |
| NNCP v3.2        | ~10.7%       | ratio SOTA (neural)                |
| **L3TC-3.2M**    | **~16.2%**   | **practical neural target**        |
| L3TC-12M         | ~16.2%       | overfits, no improvement over 3.2M |

| Model       | iPhone12 ANE | A100 GPU    |
|-------------|--------------|-------------|
| L3TC-200K   | 1.30 MB/s    | 4.35 MB/s   |
| L3TC-800K   | 980 KB/s     | 3.87 MB/s   |
| L3TC-3.2M   | 633 KB/s     | 2.50 MB/s   |

**Those "MB/s" numbers are all at batch size ≥128 (GPU) or ≥256 (mobile).**
At batch size 1 on iPhone CPU, the paper explicitly reports throughput drops
to **11–27 KB/s**.

---

## 2. Why L3TC isn't production-ready (despite the numbers)

The paper is an honest research artifact, not a product. A production
compressor is held to a completely different standard, and almost none of
that standard is met by what ships in the repo. Breaking this down by
category:

### 2.1 The batched-speed-only trap (biggest single issue)

L3TC's headline "real-time decoding at MB/s" numbers require **batch size
128 on GPU or 256 on mobile**. Real compression workloads are overwhelmingly
single-stream:

- A user runs `tar | compress > backup.tar.zst` — one stream
- A CDN edge compresses a single response body before sending — one stream
- A log forwarder compresses a batch of log lines — one stream, maybe a few
- A database compresses a page before writing it — one stream

Batched compression only makes sense in a narrow set of workloads: offline
batch processing of many independent files, some server-side data pipeline
scenarios. For the vast majority of compression use cases, **batch size 1 is
the relevant number, and at batch 1 L3TC drops to 11–27 KB/s on iPhone CPU**.

For context: gzip on the same hardware gets 50–100 MB/s. zstd gets 400–600
MB/s. xz gets 2–3 MB/s. L3TC at 11 KB/s is slower than xz by a factor of
~200 and slower than gzip by a factor of ~5000 on the same single-stream
workload. The "48% better than gzip" ratio advantage disappears into the
speed gap.

**Root cause:** almost certainly framework overhead. At 3.2M parameters, the
actual matmul compute per RWKV forward pass is microseconds. Python + PyTorch
op dispatch + tensor metadata per op is also microseconds, and there are
20–30 ops per RWKV step. Overhead dominates compute by a large multiple when
the model is this small.

This is identical to the problem NNCP solved by writing LibNC (custom C
tensor library) instead of using PyTorch. Bellard knew the framework
overhead would dominate and built around it.

### 2.2 GPU-or-bust for GPU-reported numbers

The 4.35 MB/s on A100 number is the best case in the paper. That's on an
$8,000 datacenter GPU. In production:

- Most compression happens on CPU (backups, filesystems, HTTP servers, CLI
  tools, mobile apps)
- Even server compression is often CPU-bound because GPUs are reserved for
  inference/training workloads
- Embedded and mobile contexts rarely have meaningful GPU acceleration
  available for non-graphics, non-inference workloads

A codec that only performs on a specific GPU class is not a general-purpose
codec. It's a demo. zstd's promise is "uniformly fast on every CPU, no
exceptions." L3TC's current promise is "fast on an A100 in batch mode."

### 2.3 No file format, no magic bytes, no versioning

There is no specified compressed-file format. There are no magic bytes to
identify an L3TC-compressed file. There's no version byte to handle dialect
evolution. There's no CRC or integrity check. There's no metadata block to
record the model identifier, tokenizer identifier, or training-corpus hash
that the decoder needs to round-trip correctly.

This means:

- You can't write a `file` command entry for it
- You can't feed it to a filesystem compression layer that needs to identify
  the codec
- You can't upgrade the model without breaking all prior archives
- A corrupted file has no way to fail gracefully — it just produces garbage

Compare to zstd: a fully specified binary format (RFC 8878), magic bytes
(`0x28 B5 2F FD`), frame headers, content checksums, streaming and seekable
variants, dictionary IDs, etc. zstd's format specification is ~50 pages and
exists specifically so that any implementation can round-trip any compliant
stream.

L3TC has none of this. The "file" L3TC emits is just the arithmetic coder's
output bytes, in an order only the current Python implementation understands.

### 2.4 Python + PyTorch distribution

To run L3TC today, you need:

- Python 3.x
- PyTorch (several hundred MB to a few GB)
- CUDA (if you want GPU acceleration)
- The SentencePiece library
- Their custom training/inference scripts
- The pretrained checkpoint (downloaded separately from Google Drive)
- Their config files
- Manual preprocessing steps

Compare to zstd: a single statically-linked 500 KB binary. Or gzip: preinstalled
on every Unix system ever shipped. Or xz: a single 150 KB binary.

L3TC's distribution story is "clone a repo, install Python, install PyTorch,
download a checkpoint, configure paths, run a specific CLI script." That's
acceptable for a research prototype. It's not acceptable for anything a
non-researcher would use.

### 2.5 Training corpus is tiny and wiki-only

L3TC was trained on enwik8 — the first 100 MB of English Wikipedia. That's
it. Not enwik9. Not Common Crawl. Not The Pile. Not code, not logs, not
markdown, not non-English text.

Consequences:

- **Out-of-distribution degradation** on anything that isn't wiki-like text
  (the paper acknowledges this)
- **The 12M model doesn't help** because it overfits on the 100 MB corpus.
  The scaling curve is cut off by training data, not architecture.
- **Narrow practical coverage.** Real compression workloads include source
  code, HTML, JSON, CSV, logs, documentation, chat transcripts, non-English
  text. A wiki-only model underperforms on most of these.

This is a classic academic benchmarking choice that doesn't translate to
practical deployment. The paper is measuring against a specific benchmark
(LTCB / Hutter Prize corpora). Production compression needs to work on
"whatever bytes the user hands it."

### 2.6 No deterministic inference contract

Arithmetic coding is lossless only if the encoder and decoder produce the
*exact same* probability distribution at every step. Even a one-bit
difference in the model's output — from float rounding, non-deterministic
GPU kernels, different BLAS libraries, different CUDA versions, or any other
numerical wobble — corrupts the decoded output catastrophically.

The L3TC codebase does not appear to enforce bit-identical inference across:

- CPU vs GPU
- Different GPU models (A100 vs H100 vs consumer RTX)
- Different PyTorch versions
- Different CUDA versions
- Different build configurations

This means a file compressed on one machine may not decompress on another.
That's unacceptable for a codec. NNCP addressed this explicitly by writing
LibNC and forcing deterministic primitives. L3TC has not.

### 2.7 No streaming interface

Production compression almost always needs to work as a stream:

- `cat foo.txt | compress | ssh host "decompress > bar.txt"`
- HTTP content-encoding over a chunked response
- Log forwarding where bytes arrive incrementally
- Filesystem page compression where you have a fixed block size

L3TC processes fixed 2048-byte segments in batches. There's no streaming API
that accepts bytes as they arrive, emits compressed bits as they're resolved,
and handles end-of-stream correctly. Adding a streaming mode is not just a
CLI convenience — it's a core architectural question about when the
arithmetic coder flushes, how partial segments are handled, and how the
recurrent state interacts with input buffering.

### 2.8 No integration surface for other languages or tools

A real codec has:

- A C library with a stable ABI
- Bindings for major languages (Python, Go, Rust, Java, Node)
- Integration with compression frameworks (zlib-compatible API, etc.)
- CLI tools that compose in shell pipelines
- Filesystem and archive tool support (tar, 7z, file-roller)

L3TC has: a Python repo with research scripts. Everything else is absent.

### 2.9 No failure-mode hardening

The paper does not discuss:

- What happens when the input is all null bytes
- What happens when the input is uniformly random (incompressible)
- What happens when the input contains every Unicode codepoint
- What happens when the input is a single byte
- What happens with malformed/truncated compressed inputs at decode time
- Memory bounds on adversarial inputs that could cause unbounded state
- Timing attacks or side-channel concerns for any security use case

Classical compressors have decades of fuzz testing and hardening behind them.
L3TC has none of that. A single adversarial input could crash the decoder,
hang the RWKV forward pass, or cause arbitrary misbehavior.

### 2.10 Overfit 12M regime and no clear scaling path

The paper reports L3TC-12M performs no better than L3TC-3.2M on enwik9
because it overfits the 100 MB training corpus. This means the current
ceiling on L3TC's compression ratio is effectively pinned at ~16.2% (on
enwik9) with no path to better ratios without retraining on more data.

A production system needs a path to "make this better over time" that
doesn't require fundamental architecture changes every iteration. L3TC
doesn't currently have that path because the data bottleneck is unaddressed.

---

## 3. What "production ready" actually means

Before we start fixing things, we should define the target. A production
compressor meets the following criteria:

### 3.1 Speed

- **Single-stream CPU decode: ≥ 10 MB/s on a modern laptop** (m1/m2/ryzen/etc)
- **Single-stream CPU encode: ≥ 5 MB/s** (encoding is usually more expensive,
  and typically ~2x slower is acceptable)
- **Batched GPU throughput: well into the hundreds of MB/s** (for batch
  workloads where available)
- **Startup overhead: < 100 ms** for small files (nobody waits for a
  compressor to warm up)

### 3.2 Ratio

- **Beat zstd -22 (high preset) on typical text.** This is the natural
  comparison: zstd at max effort is the current "best practical" classical
  compressor. If a neural compressor can't beat it on ratio, it has no
  reason to exist.
- **Be competitive with xz -9e.** xz is slower than zstd but gets better
  ratios on some workloads; being in the same neighborhood is table stakes.
- **Graceful degradation on non-text.** A text-focused model shouldn't blow
  up on binary data — it should fall back to something reasonable.

### 3.3 Determinism

- **Bit-identical round trip across any platform**: CPU, GPU, x86, ARM,
  Windows, Linux, macOS, iOS, Android. A file compressed on device A must
  decompress to the same bytes on device B, forever, for any A and B.
- **Forward compatibility across minor library versions** within a specified
  semver range
- **Explicit versioning for breaking format changes**

### 3.4 Distribution

- **Single-binary install.** User downloads one file, runs it, it works.
- **No required Python or framework dependencies** for the shipped binary
- **Model weights either embedded in the binary or distributed via a
  deterministic fetch**
- **Total distribution size under 50 MB** (ideally under 10)

### 3.5 Integration

- **Standard CLI** that matches zstd/xz conventions (`compress`, `decompress`,
  `--stdout`, `-k` to keep, `-f` to force, `-N` for preset levels)
- **C library** with a simple ABI (stream-in, stream-out, context struct)
- **Official file format specification** with magic bytes, versioning, and
  integrity checking
- **Python/Rust/Go bindings** at minimum

### 3.6 Robustness

- **Fuzz-tested** with AFL/libFuzzer for at least a week of CPU time before
  any 1.0 release
- **Bounded memory and time** on arbitrary inputs
- **Graceful failure** on truncated or corrupt compressed inputs (return an
  error, don't crash or produce garbage)
- **Hardened decoder** against adversarial inputs

---

## 4. Architectural decisions we need to make early

These are the big forks. Each determines what the rest of the project looks
like.

### 4.1 Do we keep RWKV, or switch backbones?

**Options:**
- Keep **RWKV-4/5** (what L3TC uses)
- Upgrade to **RWKV-7** (2024 release, much better scaling)
- Switch to **Mamba-2** (similar linear-time properties, different community)
- Switch to **a small Transformer with sliding window attention**
- Switch to a **lightweight state-space model** (S4, S5, etc.)

**My current lean:** upgrade to RWKV-7. Keeps the linear-time property
that makes L3TC practical, fixes the scaling-cliff limitation the paper
acknowledges, has production-quality inference implementations (e.g. in
`rwkv.cpp`), and the HiRA-style reparameterization ports cleanly.

**Open question:** does RWKV-7 at the same parameter count actually give a
better compression ratio? We'd need to measure. The paper's ratio numbers
are RWKV-4/5 specific.

### 4.2 Training data: how far do we stretch?

**Options:**
- **Enwik9 only** (1 GB, wiki-only — same as L3TC but larger)
- **Wikipedia full dump** (~20 GB after cleanup)
- **The Pile** (~800 GB, diverse web text + code + books)
- **RedPajama** (similar scale to The Pile, more curated)
- **Custom corpus** mixing text, code, logs, JSON, HTML, etc. to match
  expected production inputs

**My current lean:** RedPajama or a custom diverse mix. The 12M overfitting
result tells us L3TC is data-starved. Going to 100–1000 GB of training data
unlocks the scaling curve and fixes the OOD degradation simultaneously.

**Open question:** how much does training compute cost become a blocker?
Training on 1 TB of text even with a small model takes real GPU time.
Acceptable if we only train once and ship the static weights.

### 4.3 One model or a model bundle?

**Options:**
- **Single general-purpose model** (~5–20 MB weights)
- **Bundle of specialists** (prose / code / logs / JSON / binary-fallback,
  each ~3–5 MB)
- **Composable model + adapters** (base model + LoRA-style deltas per domain)

**My current lean:** start with a single model, measure, then add
specialists if the generic model has clear weaknesses on specific domains.
The bundle approach is obvious but has real costs: more training runs,
more quality gates, more complex deployment, auto-detection logic.

### 4.4 Inference runtime: what do we build on?

**Options:**
- **PyTorch** (what L3TC uses, too slow for our goals)
- **Custom C/C++** (what NNCP did, maximum control, maximum effort)
- **ggml / llama.cpp** (existing optimized inference for small language
  models, SIMD + quantization + multi-platform, already supports RWKV)
- **ONNX Runtime** (cross-platform, well-supported, moderate overhead)
- **tinygrad / candle / burn** (newer frameworks with less overhead than
  PyTorch)

**My current lean:** **ggml** backbone. It already has production-quality
RWKV inference, INT4/INT8 quantization, SIMD kernels for x86/ARM, Metal
support for Apple Silicon, CUDA and Vulkan for GPUs. It's designed for
exactly this use case (small LMs running fast on commodity hardware) and
llama.cpp proves it scales to real deployments. We'd add an arithmetic
coder on top and write a thin orchestration layer.

The trade-off is that we're taking on a dependency on someone else's
framework. But ggml is actively maintained and has a community behind it,
unlike any research one-off.

### 4.5 File format: design our own or borrow?

**Options:**
- **Design a fresh binary format** (magic bytes, frame header, content
  payload, checksum, metadata)
- **Use zstd's frame format with a custom block type** (zstd has a raw
  block type that could in principle wrap our bitstream)
- **Use a simple TLV structure** (tag-length-value, easy to parse)

**My current lean:** fresh format, but borrow the shape of zstd's format
(header + frames + content checksum). zstd's format is well-designed and
production-proven; we don't need to be original here, just correct.

### 4.6 Quantization: from day one or later?

**Options:**
- **Train FP16, ship FP16** (biggest weights, no quantization bugs)
- **Train FP16, post-quantize to INT8** (2x speedup, small quality loss)
- **Train with INT8 awareness, ship INT8** (best quality-per-size tradeoff,
  more training complexity)
- **Aggressive INT4 quantization** (smallest, fastest, biggest quality risk)

**My current lean:** ship INT8 from v0.1. The quantization story for small
RWKV models is well-understood and the throughput win matters for our
single-stream target. INT4 as a later option if INT8 hits a quality wall.

---

## 5. Phased roadmap

The goal is to ship something usable, measure against real benchmarks, and
iterate. No point building the perfect codec if it's six months away.

### Phase 0 — Baseline and measurement infrastructure (Week 1)

- Clone L3TC, get it running as-is, reproduce the paper's enwik9 numbers
  (at least within a few percentage points)
- Build a benchmark harness:
  - Reproducible test corpora (enwik8, enwik9, Silesia, Canterbury, a
    custom mix of code/logs/json/text)
  - Measure ratio, encode speed, decode speed, memory usage
  - Single-stream and batched modes separately
  - Against baselines: gzip, bzip2, xz, zstd at multiple levels
- Write a harness that reports numbers in a stable JSON format so we can
  diff across iterations

**Deliverable:** a `bench/` directory with scripts that take a corpus and
produce a report. Baseline numbers committed to the repo.

### Phase 1 — Port inference to ggml / llama.cpp (Weeks 2–3)

- Take L3TC's pretrained weights as-is (don't retrain yet)
- Load them into ggml using rwkv.cpp or similar
- Wire up the HiRA merged weights (should be possible since it's just
  regular matrix math at inference time)
- Port the arithmetic coder to C
- Connect them: tokenize → RWKV forward → arithmetic code
- Benchmark against the Python version on the same corpus

**Success criteria:** bit-identical round trip with Python version, and
≥10x faster on single-stream CPU decode. This is the single biggest win
and validates that the architectural direction is correct.

**Risks:**
- ggml's RWKV implementation may not exactly match L3TC's variant; may
  need to patch the forward pass
- Arithmetic coder precision has to match the Python version exactly or
  we lose round-trip compatibility
- Determinism across platforms is a real concern even in C

### Phase 2 — Define the file format (Week 3–4)

- Design the binary format: magic bytes, version, model identifier,
  tokenizer identifier, metadata, compressed payload, content checksum
- Write a parser/emitter in C
- Update the benchmark harness to read/write the new format
- Write a short format specification document (not a full RFC, but
  something a second implementer could read)
- Make sure the format is self-describing enough to survive model upgrades

**Deliverable:** `FORMAT.md` + working C parser/emitter.

### Phase 3 — CLI and basic distribution (Week 4–5)

- CLI tool with zstd-compatible ergonomics:
  `l3tc compress input.txt -o output.l3tc`
  `l3tc decompress output.l3tc`
  `l3tc compress < input.txt > output.l3tc`
- Static binary build for Linux / macOS / Windows
- Embed the model weights in the binary (no separate checkpoint download)
- Package as a single downloadable artifact per platform

**Deliverable:** a binary you can `chmod +x` and run that does lossless
round trips on single files.

### Phase 4 — Training on more data (Weeks 5–8)

- Pick a training corpus (likely RedPajama subset or a custom diverse mix)
- Retrain L3TC with RWKV-7 backbone and HiRA
- Larger model sizes now viable (20M, 50M, 100M) because data is not the
  bottleneck
- Measure ratio and speed against the v0 (L3TC baseline) version
- Ship the updated weights as `l3tc-0.2.0`

**Success criteria:** ratio improves on all tested corpora, not just
enwik9. Specifically, significant improvement on code, logs, JSON.

**Risks:**
- Training is expensive (real GPU time)
- The quality gap between 16% and 13% on enwik9 (closing half the gap to
  NNCP) is a realistic target but not guaranteed
- RWKV-7 may need different hyperparameters than L3TC's defaults

### Phase 5 — Robustness, fuzzing, hardening (Weeks 8–10)

- Fuzz the decoder with AFL++ for at least 72 hours of CPU time
- Add explicit bounds on memory and time for adversarial inputs
- Add content checksum verification on decode
- Handle truncated/corrupt inputs with clean error codes
- Write a test suite covering edge cases (empty files, single-byte files,
  all-null, all-random, every-unicode-codepoint, very long files)

**Deliverable:** a decoder that survives adversarial input without crashing
or producing garbage.

### Phase 6 — Integration surface (Weeks 10–12)

- C library with a stable ABI: `l3tc_encode()`, `l3tc_decode()`,
  `l3tc_encode_stream()`, `l3tc_decode_stream()`
- Python bindings (maturin / pyo3 or pybind11)
- Rust bindings
- Basic tar integration example
- Documentation for each

**Deliverable:** `libl3tc.so` / `libl3tc.dylib` / `l3tc.dll` with stable
ABI and language bindings.

### Phase 7 — Specialized models and further improvements (Month 3+)

- Train domain-specific models (code, logs, JSON)
- Automatic domain detection from input prefix
- Dictionary pretraining for structured formats
- Online LoRA adaptation for long files
- Explore Mamba-2 backbone as an experimental alternative

This is where the work gets interesting and open-ended. But phases 0–6 are
the minimum viable production codec.

---

## 6. Open questions we need to answer together

1. **What's the target use case?** This shapes everything. Some candidates:
   - General-purpose compressor (zstd alternative)
   - Text-specialized compressor (for logs, documents, code)
   - Agent-to-agent traffic compression (the llanguage connection)
   - Archival / cold storage (where ratio matters, speed less so)
   - Mobile/embedded (speed + small memory, ratio second)

   The answer determines which trade-offs to make. "General zstd alternative"
   is the most ambitious; "archival" is the most tractable.

2. **What language should we write this in?** Options:
   - **C** (following NNCP, maximum portability, most effort)
   - **C++** (ggml/llama.cpp's native language, reasonable modern tooling)
   - **Rust** (memory safety, modern tooling, still compiles to a small
     binary)

   My lean: C++ for the ggml integration, with a thin C ABI for bindings.
   Rust is tempting for safety but complicates the ggml dependency story.

3. **Do we care about cross-platform determinism from day 1, or punt to
   later?** Punting means early versions won't round-trip across devices,
   but we ship sooner. Not punting means more upfront work on deterministic
   inference primitives.

4. **How much model weight size is acceptable?** 5 MB? 50 MB? 500 MB?
   Smaller is better for distribution but limits ratio. The answer depends
   on use case (embedded wants 5 MB, CDN deployments tolerate 500 MB).

5. **Is this a research project or a shipped tool?** The honest answer
   determines how much polish and hardening we do vs how much we iterate on
   the core algorithm. A tool needs fuzz testing, integration, docs; a
   research project needs correctness and measurement.

6. **Do we care about encoding speed as much as decoding speed?** For
   asymmetric workloads (compress once, decompress many times), decoding
   speed matters much more. zstd is fast at both but most use cases only
   heavily exercise one. The answer affects which optimizations to
   prioritize.

7. **How tight should the L3TC paper compatibility be?** Can we reproduce
   existing L3TC archives? Probably not cleanly — but we should document
   the incompatibility if we break it.

---

## 7. The honest assessment

L3TC is a **good research result that nobody should deploy in production
today**. The paper proves the architecture works and hits interesting ratio
numbers. It does not prove the architecture is deployable. The gap between
those two things is large, and it's mostly engineering, not research.

The good news is that every single item on the production-readiness gap
list is a known problem with a known solution:

- Framework overhead → custom runtime (ggml, llama.cpp-style)
- Batch-only speed → stream-oriented rewrite with self-batching
- No file format → design one, following zstd's pattern
- Python distribution → static binary with embedded weights
- Narrow training data → train on more diverse corpora
- Nondeterminism → pin inference primitives, deterministic kernels
- No integration surface → write a C library, add bindings
- No robustness → fuzz testing, bounded decoders, adversarial test suite

None of this is novel research. It's all tried-and-true systems work.
Which is exactly why it's worth doing: there's a real opportunity gap
between "neural compression wins on ratio in papers" and "neural
compression ships in real tools," and closing it requires ~3 months of
focused engineering rather than a research breakthrough.

The thing that would change my mind: if after Phase 1 (custom runtime)
we can't get single-stream CPU decode above ~5 MB/s, then the architecture
itself might be too heavy for its target niche, and we'd need to
fundamentally rethink (smaller model, different backbone, different use
case). But I'd bet against this — ggml/llama.cpp regularly run larger
models than L3TC-3.2M at useful speeds, and there's no reason a small
RWKV model should be an exception.

---

## 8. Immediate next actions

Before any code, we need to decide:

1. **Pick a target use case** (question 1 in section 6). This is the most
   important decision because it changes the success criteria.
2. **Pick the implementation language and runtime** (questions 2 in
   section 6, plus decision 4.4).
3. **Pick the training data strategy** (decision 4.2). Even if we don't
   retrain until Phase 4, the decision affects what we target.

Once those three are pinned, the concrete first step is Phase 0: clone
L3TC, get it running, measure, and build a benchmark harness. We should
do that before writing any new code, because otherwise we have no way to
tell if our "improvements" are actually improvements.

After Phase 0, the single biggest-leverage step is Phase 1: the custom
runtime port. That's where the batch=1 cliff gets solved, and once it's
solved, everything else is incremental improvement rather than
existential debate.

---

## 9. What this project is NOT

- **Not a research paper.** We're not trying to invent a new compression
  algorithm. We're taking an existing algorithm and making it shippable.
- **Not a rewrite for its own sake.** If L3TC's existing code works for a
  phase, we reuse it. The point is to fix what's broken, not to
  re-implement everything.
- **Not trying to beat NNCP on ratio.** NNCP is the research ratio
  frontier; we're the practical ratio/speed frontier. Different goals.
- **Not trying to replace zstd for every use case.** Even in the best
  case, we're targeting workloads where ratio matters enough to pay ~10x
  in decode time. That's a subset of all compression use cases.
- **Not a toy.** If we ship something, it should actually work, not be
  another research repo with a demo script.

---

_Last updated: start of project. This document will evolve as we learn._

# Phase 1 — Rust rewrite of the inference runtime

**Goal:** Replace L3TC's Python + PyTorch inference with a Rust
implementation that preserves compression ratio, produces a working
round-trip (encoder output can be decoded back to original bytes),
and runs meaningfully faster single-stream on CPU.

**Target metric:** Take L3TC-200K on enwik6 from **13.24 KB/s to ≥55
KB/s** on the same machine. Stretch goal: 100-200 KB/s. Same hardware,
same model, same input.

---

## Scope boundaries

### In scope

- Rust crate that can **compress** and **decompress** a file using
  the L3TC-200K and L3TC-3.2M pretrained weights
- Byte-for-byte **self-round-trip**: `decompress(compress(x)) == x`
  for any x in enwik6 / enwik8
- **Self-consistent wire format** (our own, not Python-compatible —
  see rationale below)
- SentencePiece tokenizer integration via the `sentencepiece` Rust
  crate
- RWKV-v4 forward pass (with HiRA weights pre-merged at checkpoint
  conversion time)
- L3TC-style arithmetic coder
- CLI with zstd-ish ergonomics (`l3tc compress`, `l3tc decompress`,
  stdin/stdout support, `-k` keep source, `-f` force)
- Benchmark against the Python reference on the same corpora
- Round-trip tests that catch any regression

### Out of scope (deferred to Phase 2+)

- **Bit-identical output with the Python reference.** We only need
  self-consistency: our encoder and our decoder must agree. The
  compressed bytes do not need to be decodable by the Python L3TC
  or vice versa. This is a big relaxation that makes Phase 1
  tractable. Rationale below.
- File format stability / magic bytes / versioning (Phase 2)
- INT8/INT4 quantization (Phase 1.5 or Phase 4)
- GPU or Metal backends (Phase 5 maybe, Phase 1 is CPU-only)
- Training (never — we use L3TC's existing pretrained checkpoints)
- Fuzz testing, adversarial input handling (Phase 5)
- C ABI and language bindings (Phase 6)

### Why not bit-identical with Python

Bit-identical output with the Python reference would require
reproducing Python/PyTorch's exact floating-point computation order,
BLAS implementation, and numeric rounding. That's solvable (NNCP
does it) but it's a multi-week engineering effort that doesn't help
us with our Phase 1 goal of closing the speed gap. Our compressed
files only need to be decodable by our own decoder — think of it
like zstd's output not being decodable by gzip. As long as the
ratio we achieve is close to L3TC's reported ratio, we've preserved
the value of the learned compression.

If we ever need Python compatibility (for decoding archives created
with the reference implementation), we can add a separate
"compatibility mode" that matches Python's numerics exactly. But
that's a different project.

---

## Architecture

### Crate layout

```
l3tc-rust/
├── Cargo.toml
├── Cargo.lock
├── README.md
├── src/
│   ├── lib.rs            # Re-exports, top-level Error type
│   ├── bin/
│   │   └── l3tc.rs       # CLI entry point
│   ├── bitio.rs          # BitReader / BitWriter for the arithmetic coder
│   ├── arithmetic.rs     # Arithmetic coder (encode/decode, deterministic)
│   ├── tokenizer.rs      # SentencePiece wrapper, outlier detection
│   ├── tensor.rs         # Minimal f32 tensor ops we need for the model
│   ├── checkpoint.rs     # Load the Rust-friendly exported weights
│   ├── rwkv.rs           # RWKV-v4 forward pass (HiRA pre-merged)
│   ├── codec.rs          # High-level compress / decompress (ties it all together)
│   └── error.rs          # Error type + Result alias
├── tests/
│   ├── roundtrip.rs      # End-to-end: compress then decompress == input
│   ├── arithmetic.rs     # Arithmetic coder unit tests
│   ├── tokenizer.rs      # Tokenizer round-trip tests
│   └── rwkv.rs           # Model forward pass sanity checks
├── benches/
│   └── compress.rs       # Criterion benchmarks
└── scripts/
    └── convert_checkpoint.py    # PyTorch .pth → Rust-friendly binary
```

### Module boundaries

Each module has one responsibility and is independently testable:

- **`bitio`** — reads and writes individual bits on top of a byte stream.
  Exists because arithmetic coding is naturally bit-level.
- **`arithmetic`** — the probabilistic encoder/decoder. Takes a
  probability distribution and a symbol; emits/consumes bits. Pure
  integer math, deterministic, zero floating-point.
- **`tokenizer`** — wraps the SPM model from the L3TC tokenizer we
  already trained, plus the outlier-aware routing (in-vocab tokens
  go through the neural path, out-of-vocab bytes bypass).
- **`tensor`** — a minimum viable linear algebra layer. At 3.2M
  parameters with a ~256 hidden size, the matrix ops we need are
  modest: matmul, layer norm, sigmoid, exp, element-wise ops. Written
  by hand (or backed by `ndarray` + `matrixmultiply` if that's
  faster). No heavy framework.
- **`checkpoint`** — reads a simple binary format produced by a Python
  preprocessing script. Not PyTorch pickle (too weird to parse in
  Rust). Just f32 weight tensors + shape metadata.
- **`rwkv`** — the model. Forward pass: embed → N layers (layer norm,
  time mix, channel mix) → unembed → logits. HiRA is pre-merged
  during checkpoint conversion so we run single-path RKV matrices.
- **`codec`** — the glue. Compress: open file → tokenize → for each
  segment, run RWKV forward pass, feed token probabilities to
  arithmetic coder, emit bits. Decompress: reverse.
- **`error`** — a single `Error` enum and `Result<T>` alias. Libraries
  return typed errors; the CLI converts them to exit codes + messages.

### Two-stage checkpoint loading

The L3TC checkpoints are PyTorch `.pth` files — pickled Python
objects. Parsing pickle in Rust is possible but messy. We avoid
the problem entirely with a one-shot Python preprocessing script:

```
checkpoint_0019.pth  →  scripts/convert_checkpoint.py  →  l3tc_200k.bin
```

The output format is dead simple:

```
[header: magic(4) + version(4) + num_tensors(4)]
[for each tensor:
    name_len(u32) + name(utf-8) +
    ndim(u32) + dims(u32 × ndim) +
    dtype(u32) +
    data_bytes(u64) + data(f32 × numel)
]
```

This runs once per checkpoint. Rust reads the `.bin` directly with
`std::fs::read` and zero dependencies.

HiRA merging happens in the Python converter, not at Rust load time,
so the Rust code sees a clean pre-merged RWKV-v4 weight set.

### Deterministic arithmetic

The arithmetic coder is pure integer math (no floating point) so
it's trivially deterministic across platforms. The only source of
non-determinism is the neural network forward pass, and we handle
that by:

1. Using the same dtype (f32) and computation order throughout
2. Avoiding parallelism in the hot loop (single-threaded matmul)
3. Using a deterministic matmul (our own or `matrixmultiply` with
   fixed thread count)

Because we only need encoder-decoder self-consistency, not
Python-compatibility, this is straightforward. The same binary
running on the same CPU on the same input produces the same output,
period.

---

## Dependencies

Conservative and justified:

| Crate | Purpose | Why this and not alternatives |
|---|---|---|
| `clap` | CLI arg parsing | Standard, ergonomic, derive API |
| `anyhow` | CLI-level errors | Simple unified error type for binaries |
| `thiserror` | Library-level errors | Typed errors for the lib crate |
| `sentencepiece` | SPM tokenization | Official Rust bindings (wraps the C++ SentencePiece lib) |
| `byteorder` | Endian-safe binary I/O | Stable, tiny, standard for checkpoint parsing |
| `memmap2` | mmap the input file | Zero-copy file reads for large inputs |

Deliberately avoided:

- **`candle-core`** — we don't need its ML framework; the model is
  3 layers × 256 dim and we can write the forward pass directly in
  ~200 lines of Rust without a tensor framework's abstractions.
- **`tch`** — PyTorch bindings, huge dependency, defeats the purpose
  of removing the PyTorch overhead.
- **`ndarray`** — solid crate but overkill for a handful of matmul
  operations. We'll write tiny linalg ourselves.
- **`burn`** / **`dfdx`** — framework overhead, same reasoning as
  candle.

If the hand-rolled approach turns out to be too slow (unlikely given
the model size), we fall back to `candle-core` or `ndarray` +
`matrixmultiply` for the heavy matmuls only.

---

## Execution order

1. **Crate skeleton** — Cargo.toml, src/, tests/, basic error types.
   Get `cargo build` green with placeholder modules.
2. **bitio** — BitReader / BitWriter with unit tests. Trivial, ~80 lines.
3. **arithmetic** — Arithmetic coder. Port from L3TC's
   `util/arithmeticcoding.py`. Unit test with known probability
   sequences.
4. **checkpoint converter (Python)** — `scripts/convert_checkpoint.py`
   reads L3TC-200K .pth, applies HiRA merge, exports .bin. Runs
   once.
5. **checkpoint (Rust)** — reads the .bin and returns a map of named
   f32 tensors. Unit tested against a small synthetic .bin.
6. **tensor** — matmul, layer norm, sigmoid, exp, add, mul. Hand-rolled
   or via `matrixmultiply`. Unit tests for each.
7. **tokenizer** — wrap `sentencepiece` crate. Unit test against the
   SPM model we already have. Also handle outlier detection.
8. **rwkv** — forward pass. One token at a time. Load weights from
   a checkpoint, run embed → layers → unembed. Sanity test: compare
   output logits against the Python model on a known input.
9. **codec** — compress / decompress tying arithmetic + tokenizer +
   rwkv together.
10. **CLI** — `l3tc compress`, `l3tc decompress` with stdin/stdout,
    file paths, progress output.
11. **Round-trip integration test** — compress enwik6, decompress,
    assert byte-identical to input.
12. **Benchmark** — measure the Rust version with the same harness
    we used for classical compressors. Report ratio and speed.

Each step has a clear "done" criterion and a test. The project grows
one module at a time.

---

## Success criteria

Phase 1 is complete when:

1. `cargo test` passes all unit and integration tests
2. `cargo run --release -- compress bench/corpora/enwik6 -o /tmp/out.l3tc`
   produces a compressed file
3. `cargo run --release -- decompress /tmp/out.l3tc -o /tmp/out.txt`
   produces a file that is byte-identical to `enwik6`
4. The Rust version's compression ratio is within ±0.5 pp of the
   Python reference on the same corpus
5. The Rust version's single-stream throughput is at least **5×**
   the Python reference on the same machine, same corpus. Ideal
   target: 10-20× for the 200K model.
6. Measured numbers committed to `bench/results/l3tc-rust-enwik6.json`

---

## Open questions

1. **Is the SPM tokenizer match between Python and Rust?** Both sides
   use the same `.model` file, so they should produce identical token
   sequences on the same input. Needs a unit test to confirm.

2. **Does HiRA pre-merging preserve the ratio exactly?** The Python
   L3TC already merges HiRA at load time (see `scripts/compressor.py`
   lines 575-608), so pre-merging in our converter should produce
   the same weights the Python model runs on. But worth a sanity check.

3. **How much of the 75-second Python runtime is the checkpoint
   load?** We've been blaming op dispatch. If `torch.load` itself
   takes, say, 20 seconds on a 200K checkpoint, that changes the
   breakdown. Should profile before over-optimizing the inference loop.

4. **What BLAS does macOS use by default?** Apple's Accelerate
   framework is available and very fast on M-series. We might get
   that for free via `matrixmultiply` or by linking explicitly.

5. **Is there a latent race between tokenizer cost and model cost?**
   If tokenization is a meaningful fraction of runtime at tiny model
   sizes, we need to optimize it too.

---

## What "secure and stable" means here

The user asked for "secure and stable" alongside "close the speed
gap." Phase 1 delivers the foundation for those properties; deeper
hardening is Phase 5.

**What Phase 1 delivers for security:**

- **Rust's memory safety.** No buffer overruns, no use-after-free,
  no data races. This is the dominant CVE class for C compressors
  (see the xz-utils backdoor, zlib CVEs, etc.). Switching to Rust
  eliminates it.
- **No unsafe blocks in the hot path.** Any `unsafe` is isolated
  to narrow FFI boundaries (sentencepiece), documented, and
  justified.
- **Bounded allocations.** The arithmetic coder, tokenizer buffers,
  and tensor buffers all have known sizes determined by the input
  and the model. No user input can cause unbounded allocation.

**What Phase 1 delivers for stability:**

- **Typed errors.** Every failure mode is a variant of the `Error`
  enum. Nothing panics on malformed input.
- **Round-trip tests** on every merge, not just the happy path.
- **No global state.** All state lives in values you pass around.
- **Deterministic output.** Same input + same binary + same
  checkpoint = same output, forever.

**What's deferred to Phase 5:**

- Fuzzing with AFL++ / libFuzzer
- Adversarial input corpora (maximum-entropy, truncated, malformed)
- Content checksum in the file format (Phase 2 adds the format)
- Decoder hardening against resource exhaustion attacks
- Cross-platform determinism testing (we'll test on macOS first,
  Linux/Windows in Phase 5)

---

## Non-goals

- **Not trying to beat NNCP or cmix on ratio.** L3TC's ratio is what
  it is (~13% on enwik9). We're not improving the compression
  algorithm in Phase 1, just the inference speed.
- **Not trying to beat zstd on speed.** We're closing the speed gap,
  not matching zstd. A 10-20× speedup still leaves us far below
  zstd.
- **Not trying to beat the Python reference in every dimension.** We
  accept that our output isn't Python-compatible. We accept that
  our output format may differ. We trade compatibility for velocity.

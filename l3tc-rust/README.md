# l3tc-rust

Rust port of L3TC learned lossless text compression.

See the parent project ([`l3tc-prod`](..)) for context. This crate
is the production inference runtime. Phase 0 numbers (the baseline
we're beating) are in
[`../docs/phase_0_findings.md`](../docs/phase_0_findings.md). TL;DR:
the Python reference runs at 10-13 KB/s single-stream on Apple
Silicon because framework overhead dominates runtime. This crate
removes the framework overhead.

## Status

**Phases 0, 1, 2 done. Phase 2.5 in progress** (INT8 head +
NEON block matvecs + vectorized cum_freqs).

Current throughput on enwik6 (1 MB, round-trip verified):
- **89 KB/s compress** (6.9× Python L3TC-200K)
- **92 KB/s decompress**
- **ratio 0.2060** (best of any compressor in the benchmark suite,
  27% better than bzip2-9)

Scaling:
- 50 KB: 71.6 KB/s / ratio 0.1815
- 1 MB enwik6: 89.8 KB/s / ratio 0.2060
- 10 MB enwik8 subset: 88.4 KB/s / ratio 0.2216

See [`../PHASE_2_5.md`](../PHASE_2_5.md) for the current phase plan
and targets.

## Build

```bash
cargo build --release
cargo test --release                      # 35 unit tests
./target/release/l3tc --help
```

The `.cargo/config.toml` enables `target-cpu=native` for release
builds so the compiler can use the full SIMD width of the local
machine. This makes the release binary non-portable (built for the
exact CPU model) but is the fastest option for development and
benchmarking. Production artifacts for multiple platforms will
build per-target without this flag in Phase 3.

## Usage

```bash
# Compress
l3tc compress input.txt -o input.l3tc

# Decompress
l3tc decompress input.l3tc -o input.txt

# With timing + verify
l3tc compress input.txt -o input.l3tc --verify --time

# Override the default 4096-byte segment size
l3tc compress input.txt -o input.l3tc --segment-bytes 8192
```

The CLI defaults `--model` to `checkpoints/l3tc_200k.bin` (the
converted binary) and `--tokenizer` to the SentencePiece model
living under `../vendor/L3TC/dictionary/...`. Override both for
production use or to point at a different L3TC variant (the
converter supports 200K, 800K, 3.2M, and 12M).

## Fast iteration loop

```bash
./iter.sh                     # 50 KB corpus, default
./iter.sh /tmp/my_test.txt    # explicit corpus
INPUT=../bench/corpora/enwik6 ./iter.sh    # full 1 MB enwik6
```

`iter.sh` rebuilds, compresses, decompresses, and diffs against the
original in one command — about 10 seconds on a 50 KB corpus. This
is the workhorse for Phase 2.5 optimization work.

## Architecture

- `src/error.rs` — thiserror-based typed errors
- `src/bitio.rs` — bit-level I/O for the arithmetic coder
- `src/arithmetic.rs` — Nayuki-style arithmetic coder (integer-only,
  deterministic, same as the Python reference)
- `src/tensor.rs` — hand-rolled f32 linear algebra: matvec (scalar,
  col-major AXPY, optional parallel), layer_norm, sigmoid, relu,
  square, exp, time_mix. Phase 2.5 adds NEON intrinsics for the
  96x96 block matvecs.
- `src/checkpoint.rs` — reader for the Rust-friendly binary format
  produced by `scripts/convert_checkpoint.py`. Phase 2.5 adds INT8
  tensor support for the head weight.
- `src/rwkv.rs` — the Model and Session types. RWKV-v4 + HiRA
  forward pass (one token at a time). HiRA weights are pre-merged
  at checkpoint conversion time so the Rust runtime only sees
  standard projections.
- `src/tokenizer.rs` — SentencePiece wrapper + raw-fallback
  refinement for segments where SPM's `decode(encode(x)) != x`.
- `src/codec.rs` — high-level compress/decompress, segment-level
  parallelism via rayon, the compressed file format.
- `src/bin/l3tc.rs` — CLI binary

## Dependencies

Intentionally minimal. See `Cargo.toml`. No ML framework — we write
the RWKV forward pass by hand because the model is 2-4 layers and a
framework's abstractions cost more than they save (see DECISIONS.md
D8). No Python or PyTorch at runtime. No CUDA.

Runtime deps:
- `thiserror` — typed library errors
- `byteorder` — endian-safe binary I/O
- `memmap2` — mmap file reads (reserved for Phase 3 streaming)
- `matrixmultiply` — SIMD matmul (retained for the parallel head
  path; not called on the default path)
- `rayon` — segment-level parallelism
- `sentencepiece` — tokenizer (wraps the C++ library)

Binary-only deps (behind the `cli` feature):
- `clap` — CLI parser
- `anyhow` — binary-level error reporting

Dev deps:
- `rand` — deterministic RNG for tests

## Testing

```bash
cargo test --release                                   # 35 unit tests
cargo test --release --test end_to_end -- --ignored \
    --test-threads=1 --nocapture                       # 4 integration tests
```

The integration tests need:
- `checkpoints/l3tc_200k.bin` (produced by
  `scripts/convert_checkpoint.py`)
- `../vendor/L3TC/dictionary/.../spm_enwik8_bpe_16384_0.999.model`
  (produced by `../scripts/setup.sh` + the L3TC tokenizer trainer)

See `../PHASE_0.md` and `../docs/phase_0_findings.md` for the full
setup sequence.

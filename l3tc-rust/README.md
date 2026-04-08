# l3tc-rust

Rust port of L3TC learned lossless text compression.

See the parent project ([`l3tc-prod`](..)) for context: Phase 1 of a
broader effort to productionize learned compression. This crate is the
inference runtime rewrite. Phase 0 numbers are in
[`../docs/phase_0_findings.md`](../docs/phase_0_findings.md) — TL;DR:
the Python reference implementation runs at 10-13 KB/s single stream
on Apple Silicon because framework overhead dominates runtime. This
crate removes the framework overhead.

## Phase 1 target

- Preserve L3TC's compression ratio (~13% on Wikipedia text)
- Close the single-stream speed gap by at least 5× vs Python
- Full round-trip: `decompress(compress(x)) == x` for any x
- Pure Rust, single static binary, zero Python dependency at runtime

See [`../PHASE_1.md`](../PHASE_1.md) for the full plan.

## Build

```bash
cargo build --release
cargo test
./target/release/l3tc --help
```

## Usage

```bash
# Compress a file
l3tc compress input.txt -o input.l3tc

# Decompress
l3tc decompress input.l3tc -o input.txt

# Stream mode
cat input.txt | l3tc compress > input.l3tc
cat input.l3tc | l3tc decompress > input.txt
```

## Dependencies

Intentionally minimal. See `Cargo.toml`. No ML framework (we write
the RWKV forward pass by hand because the model is 3 layers and we
don't need candle's abstractions). No Python. No PyTorch. No CUDA.
Just Rust, a BLAS for matmul, and the SentencePiece C++ library via
its Rust bindings for tokenization.

## Status

Phase 1 in progress. See [`../PHASE_1.md`](../PHASE_1.md) for the
current checklist.

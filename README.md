# l3tc-prod

A neural lossless compressor for text and structured data.
10-83× faster than every other neural/PAQ compressor on the
[Large Text Compression Benchmark](http://mattmahoney.net/dc/text.html),
with 30-45% better compression ratio than zstd/xz/bzip2 on
text.

Built for backup, archival, log shipping, and cold storage —
anywhere ratio matters more than speed. Production Rust
implementation of RWKV-v4 + HiRA driving an arithmetic coder.
6.6K LOC, minimal deps, hand-rolled NEON kernels, no ML
framework at runtime.

**Current status:** the shipped 200K model is trained on
Wikipedia (enwik8) and compresses prose well. A generalist
model trained on 52 GB of diverse data (prose + GitHub code +
configs + logs + CSV) is in development to cover all text and
structured data types — see `docs/phases/PHASE_11.md`.

## Numbers

### vs learned compressors (CPU, wall-clock)

| Compressor | bpb | KB/s | Hardware |
|---|---:|---:|---|
| **l3tc-prod 200K** | **~1.43** | **131** | Apple M-series, 8 cores |
| **l3tc-prod 3.2M** | **~1.07** | **26** | Apple M-series, 8 cores |
| nncp v3.2 | 0.857 | 4.04 | RTX 3090 GPU |
| cmix v21 | 0.866 | 1.57 | CPU |
| lstm-compress v3 | ~1.39 | 10.58 | CPU |

l3tc-prod is 10-83× faster wall-clock than every LTCB entry.
The trade: ~67% behind the ratio frontier. See
[`docs/COMPARISON.md`](docs/COMPARISON.md) for the full
primary-source analysis.

### vs classical compressors (enwik6, 1 MB)

| Compressor | Ratio | Compress MB/s |
|---|---:|---:|
| **l3tc-prod 200K** | **0.1699** | **0.131** |
| **l3tc-prod 3.2M** | **0.1337** | **0.026** |
| bzip2-9 | 0.2813 | 16.67 |
| xz-9e | 0.2907 | 3.77 |
| zstd-22 | 0.3001 | 4.34 |

Best ratio in the suite — 41% better than bzip2, 43% better than
zstd. The cost is wall time: xz is ~30× faster. This is a
ratio-first tool for cold archive, compliance, and scientific
text corpora.

## Quick start

```bash
cd l3tc-rust
cargo build --release
cargo test --release

# Round-trip compress + verify
./iter.sh

# Explicit CLI
./target/release/l3tc compress input.txt -o out.l3tc --verify --time
./target/release/l3tc decompress out.l3tc -o back.txt --time

# Use the high-ratio 3.2M model
./target/release/l3tc compress input.txt --model checkpoints/l3tc_3m2.bin -o out.l3tc
```

## How it works

A small recurrent language model (RWKV-v4 + HiRA, 200K non-embed
params) predicts the next BPE token given the context. The
prediction's probability distribution feeds an arithmetic coder
that encodes the actual next token in fewer bits when the model
is confident. Better predictions = fewer bits = smaller file.

The Rust runtime hand-rolls the forward pass with NEON intrinsics,
INT8-quantizes the head weight, and parallelizes across segments
via rayon. Runs at batch-1 on CPU — no GPU, no Python, no
framework.

## Building from source

```bash
# Prerequisites: Rust toolchain, Python 3.10+ (for checkpoint conversion)
git clone https://github.com/dmatth1/ltec.git l3tc-prod
cd l3tc-prod

# Set up the L3TC reference (for checkpoint conversion)
./scripts/setup.sh

# Convert the checkpoint to Rust format
cd vendor/L3TC && source .venv/bin/activate
python ../../l3tc-rust/scripts/convert_checkpoint.py \
    --input checkpoints/l3tc_checkpoints/l3tc_200k_bpe16k_c999_checkpoint0019.pth \
    --config config/l3tc/l3tc_200k.py \
    --output ../../l3tc-rust/checkpoints/l3tc_200k.bin

# Build and test
cd ../../l3tc-rust
cargo build --release
cargo test --release
./iter.sh
```

## Documentation

- [`docs/COMPARISON.md`](docs/COMPARISON.md) — primary-source
  compressor landscape analysis with LTCB numbers
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — architectural
  decision log including reversals
- [`docs/phases/`](docs/phases/) — per-phase plans
- [`docs/phase-findings/`](docs/phase-findings/) — per-phase
  results
- [`docs/DETAILED_README.md`](docs/DETAILED_README.md) — full
  project history and detailed status

## License

See the L3TC paper license. This is a research-to-product
translation of L3TC (AAAI 2025).

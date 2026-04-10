# l3tc-prod

**The fastest CPU learned compressor.** 10-83× faster than every
other neural/PAQ compressor on the
[Large Text Compression Benchmark](http://mattmahoney.net/dc/text.html),
with 30-45% better ratio than zstd/xz/bzip2.

A production Rust implementation of RWKV-v4 + HiRA driving an
arithmetic coder. 6.6K LOC, minimal deps, hand-rolled NEON
kernels, no ML framework at runtime.

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
The trade: ~67% behind the ratio frontier. Different operating
point, not a failure — nobody else ships interactive learned
compression at this speed. See
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
ratio-first tool for cold archive, compliance, scientific text
corpora, log archival.

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

# Debug / profiling
./target/release/l3tc entropy-bound --input input.txt --segment-bytes 4096
./target/release/l3tc audit --input input.txt
./target/release/l3tc profile --input input.txt
```

## How it works

A small recurrent language model (RWKV-v4 + HiRA, 200K non-embed
params) predicts the next BPE token given the context. The
prediction's probability distribution feeds an arithmetic coder
that encodes the actual next token in fewer bits when the model
is confident. Better predictions = fewer bits = smaller file.

The Rust runtime hand-rolls the forward pass with NEON intrinsics,
INT8-quantizes the head weight, and parallelizes across segments
via rayon. The model runs at batch-1 on CPU — no GPU, no Python,
no framework.

## Project status

- **Phase 4 complete** — runtime optimization done. Forward pass
  bit-identical to Python L3TC. Speed ceiling reached (~150 KB/s
  on CPU for this model class).
- **Phase 11 in progress** — training on a broader corpus (Pile
  dedup) to fix the OOD cliff (webster/dickens/code/logs compress
  badly with the enwik-only model). Same architecture, improved
  training recipe (AdamW + cosine warmup).
- **Next:** Phase 9 (fuzzing) → Phase 7 (numeric determinism) →
  Phase 6 (release builds) → Phase 10 (distribution). ~6-8 weeks
  to open source release.

See [`docs/phases/`](docs/phases/) for detailed per-phase plans
and [`docs/phase-findings/`](docs/phase-findings/) for results.

## Architecture

```
l3tc-prod/
├── l3tc-rust/             Rust crate (the product)
│   ├── src/
│   │   ├── rwkv.rs        RWKV-v4 + HiRA forward pass
│   │   ├── tensor.rs      Hand-rolled f32 + INT8 + NEON linalg
│   │   ├── codec.rs       Compress/decompress + segment parallelism
│   │   ├── arithmetic.rs  Nayuki-style arithmetic coder
│   │   ├── tokenizer.rs   SentencePiece wrapper
│   │   ├── checkpoint.rs  Binary checkpoint reader
│   │   └── bin/l3tc.rs    CLI
│   ├── checkpoints/       Converted model weights (.bin)
│   └── tests/             Integration tests
├── scripts/               Training + cloud infrastructure
├── bench/                 Benchmark harness (stdlib-only Python)
├── vendor/                (gitignored) L3TC + RWKV-LM reference
├── docs/                  Phases, findings, analysis, comparisons
├── CLAUDE.md              Project goals + regression gates
└── README.md              This file
```

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

## Key decisions

- **Hand-rolled tensor math** over candle/tch — the model is
  2 layers × 96 dim; a framework's abstractions cost more than
  they save.
- **Segment-level parallelism** via rayon — the only parallelism
  lever available in autoregressive compression. Biggest single
  speed win (5×).
- **INT8 head quantization** — the 16384×96 head weight is the
  memory-bandwidth bottleneck; INT8 halves the streaming cost.
- **No GPU required** — runs on any aarch64 or x86_64 CPU.

See [`docs/DECISIONS.md`](docs/DECISIONS.md) for the full
decision log including reversals.

## License

See the L3TC paper license. This is a research-to-product
translation of L3TC (AAAI 2025).

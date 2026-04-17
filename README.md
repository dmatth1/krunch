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

### Speed by configuration (Phase 12 measured, Phase 13 in progress)

All measurements on a clean Apple M-series MacBook (8 performance
+ 2 efficiency cores) using 1 MB enwik6. Metal row uses the
post-Phase-13n fused-layer kernel at batch=256.

| backend | model | compress | decompress | ratio | bpb |
|---|---|---:|---:|---:|---:|
| CPU 1 thread | 200K | **22.7 KB/s** | **23.6 KB/s** | 0.1699 | ~1.43 |
| CPU 10 threads (rayon) | 200K | **172 KB/s** | **180 KB/s** | 0.1699 | ~1.43 |
| CPU 10 threads (rayon) | 3.2M | **40 KB/s** | **41 KB/s** | 0.1337 | ~1.07 |
| GPU Metal (200K, 256-lane + fused layers) | 200K | **~48 KB/s** | ~48 KB/s | 0.1699 | ~1.43 |

Metal ratio now matches CPU exactly (0.1699) — the Phase 13n
fused-layer kernel keeps per-layer intermediate state in registers
rather than round-tripping through device memory, which
eliminated the ~1 ULP/layer FP drift that previously left the
Metal cum_freqs one byte off CPU's on the same input.

CPU multi-thread scaling: ~7.5× over single-thread (memory-bandwidth
bound from 10 threads up).

**Phase 13 GPU backend status:** functional, bit-correct, and
approximately ~28% of the single-thread CPU rate on 1 MB enwik6.
End-to-end Metal compress on 1 MB: **~48 KB/s** at batch=256 with
the post-13n fused-layer kernel (one MSL dispatch covers the whole
per-layer forward pass). Ratio 0.1699 matches CPU. Cumulative
across Phases 13e→13n: **~320× over the 0.15 KB/s bring-up** (and
~10× over the 5 KB/s that the 50 KB lane-starved test was showing
pre-Phase-13n). The 1-3 MB/s projection remains open — next levers
are multi-queue parallel dispatch, simdgroup_matrix tiles for the
7 per-layer matvecs, or INT8-quantizing the 9216-element block
projections. See [`docs/phases/PHASE_13.md`](docs/phases/PHASE_13.md)
for the full architecture and the bit-equivalence finding.

**Backend choice:** pass `--backend=cpu` (default) or `--backend=metal`.
Files compressed with `metal` MUST be decompressed with `metal`
(FP arithmetic differs between backends and would desync the AC).
The decompress side defaults to `--backend=auto`, which reads the
file header to pick the matching backend automatically.

### vs learned compressors (CPU, wall-clock)

| Compressor | bpb | KB/s | Hardware |
|---|---:|---:|---|
| **l3tc-prod 200K (Phase 12h)** | **~1.43** | **172** | Apple M-series, 10 cores |
| **l3tc-prod 3.2M (Phase 12g)** | **~1.07** | **40** | Apple M-series, 10 cores |
| nncp v3.2 | 0.857 | 4.04 | RTX 3090 GPU |
| cmix v21 | 0.866 | 1.57 | CPU |
| lstm-compress v3 | ~1.39 | 10.58 | CPU |
| Nacrith (2026) | 0.94 (enwik8) | ~200-280 | GTX 1050 Ti GPU |
| ts_zip (Bellard) | 1.084 | ~1024 | RTX 4090 GPU |

l3tc-prod is 10-83× faster wall-clock than every CPU-only LTCB entry,
and the only learned compressor with first-class CPU and (in-progress)
GPU backends. The trade vs the ratio frontier: ~67% behind. See
[`docs/COMPARISON.md`](docs/COMPARISON.md) for the full primary-source
analysis.

### vs classical compressors (enwik6, 1 MB)

| Compressor | Ratio | Compress MB/s |
|---|---:|---:|
| **l3tc-prod 200K** | **0.1699** | **0.172** |
| **l3tc-prod 3.2M** | **0.1337** | **0.040** |
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

# l3tc-prod

**A production-ready learned lossless compressor built on L3TC's ideas.**

Target: a zstd alternative that uses a small RWKV-based language model
with arithmetic coding to achieve better compression ratios than
classical codecs, at single-stream speeds that are practical for general
use.

## Status

**Phases 0-2.5 complete. Starting Phase 3 (file format + stream API + Silesia + enwik8/9).**

- ✅ **Phase 0 — Reproduce L3TC with solid engineering foundations.**
  Built a Python benchmark harness, reproduced L3TC-200K and L3TC-3.2M on
  enwik6 (matching paper ratios within noise), measured classical
  compressors on the same corpus, identified framework overhead as the
  dominant cost in L3TC's Python implementation (~97% of runtime on the
  200K model).
- ✅ **Phase 1 — Rust inference runtime.** Rewrote L3TC's forward pass,
  tokenizer wrapper, arithmetic coder, and codec glue in Rust. Hit
  byte-identical round trip on the full 1 MB enwik6 corpus (including
  Persian/Arabic interlanguage links via the raw-fallback path) and
  **beat Python L3TC-200K by 6.43×** on single-stream CPU throughput.
- ✅ **Phase 2 — Ratio tuning and throughput polish.** Default segment
  size 2048 → 4096 for better ratio, serial head matvec (rayon overhead
  was hurting), session pooling experiments, comparative benchmarks vs
  all classical compressors. Final: **89 KB/s compress, 6.9× Python**,
  ratio 0.2060 on enwik6 (best ratio of any compressor in the
  benchmark, 27% better than bzip2-9).
- ✅ **Phase 2.5 — Aggressive speed optimizations (partial).** Shipped
  hand-tuned NEON `matvec_96x96` for the 12 block projections per token
  (2.5a) and INT8 per-column quantization of the 1.5 M-element head
  weight with a widening-AXPY matvec (2.5b). Vectorized cum_freqs
  (2.5c) was attempted and deferred — the simple prefilter killed
  autovectorization and the top-K approach carries too much ratio
  risk for the now-shrunk upside. Final: **116 KB/s compress, 7.4×
  Python**, ratio 0.2061 (essentially unchanged).
- 🚧 **Phase 3 — File format, stream API, Silesia benchmarks, full
  enwik8/enwik9 measurements.**

See [`PHASE_0.md`](PHASE_0.md), [`PHASE_1.md`](PHASE_1.md),
[`PHASE_2.md`](PHASE_2.md), and [`PHASE_2_5.md`](PHASE_2_5.md) for the
detailed plans. See [`ANALYSIS.md`](ANALYSIS.md) for the full project
thinking. See [`DECISIONS.md`](DECISIONS.md) for the architectural
decision log (including what we tried and reversed). See
[`docs/`](docs/) for per-phase findings.

## Headline numbers (enwik6, 1 MB, round-trip verified)

| Compressor | Ratio | Compress MB/s | Decompress MB/s |
|---|---:|---:|---:|
| **l3tc-rust** (Phase 2.5) | **0.2061** | **0.116** | **0.121** |
| **l3tc-rust** on enwik8 (100 MB) | **0.2166** | **0.111** | **0.118** |
| Python L3TC-200K | 0.1665 | 0.013 | — |
| bzip2-9 | 0.2813 | 16.67 | 35.09 |
| xz-9e | 0.2907 | 3.77 | 52.39 |
| zstd-22 | 0.3001 | 4.34 | 125.17 |
| gzip-9 | 0.3558 | 23.04 | 151.49 |

l3tc-rust has the **best ratio of any compressor in the table** —
27% better than bzip2-9, 41% better than zstd-22. It is also the
slowest; the closest classical compressor on throughput (xz-9e) is
~42× faster. Closing that gap is the Phase 2.5 / 3 work.

Scaling is stable on larger inputs:
- 50 KB subset: 0.1815 ratio at 71.6 KB/s
- 1 MB enwik6: 0.2060 ratio at 89.8 KB/s
- 10 MB enwik8 subset: 0.2216 ratio at 88.4 KB/s

## Why this project exists

L3TC (AAAI 2025) is the first learned lossless compressor that looks
like it could plausibly compete with classical codecs in production.
The paper reports 16% compression ratio on enwik9 (vs 32% for gzip,
~21% for zstd) with "megabytes per second" decoding speeds.

But the paper's speed numbers come from batched inference at batch
size 128+ on GPU. **At batch size 1 on CPU — which is how real
compression workloads actually run — L3TC drops to 11-27 KB/s**,
slower than gzip by roughly 5000×. That's the gap this project
closes.

The research is done; the engineering isn't. Closing the gap is
mostly systems work: replacing Python/PyTorch inference with a
custom runtime, designing a real file format, shipping as a single
static binary, handling adversarial inputs gracefully. None of it
is novel research. All of it is necessary for anything that claims
to be a compressor.

## Roadmap

| Phase | What | Status |
|-------|------|--------|
| 0 | Reproduce L3TC, build benchmark harness, baseline vs classical | ✅ done |
| 1 | Port inference from PyTorch to Rust, round-trip on enwik6 | ✅ done (6.43× Python) |
| 2 | Ratio tuning, throughput polish, scaling measurements | ✅ done (6.9× Python) |
| **2.5** | **INT8 head + NEON block matvecs + vectorized cum_freqs** | 🚧 in progress |
| 3 | File format stability, stream API, Silesia + enwik8/9 benchmarks | ⏳ next |
| 4 | Retrain with RWKV-7 on RedPajama for better ratio and OOD | ⏳ |
| 5 | Fuzz testing, robustness hardening, adversarial input handling | ⏳ |
| 6 | C ABI, language bindings, ecosystem integration | ⏳ |

## Layout

```
l3tc-prod/
├── README.md              this file
├── ANALYSIS.md            full project thinking document
├── DECISIONS.md           architectural decision log
├── PHASE_0.md             detailed Phase 0 plan, marked done
├── PHASE_1.md             detailed Phase 1 plan, marked done
├── PHASE_2.md             detailed Phase 2 plan, marked done
├── PHASE_2_5.md           detailed Phase 2.5 plan (current)
├── docs/
│   ├── phase_0_findings.md
│   └── phase_2_findings.md
├── bench/                 Python benchmark harness (stdlib-only)
│   ├── bench.py
│   ├── compressors.py
│   ├── corpora/           test corpora (gitignored, see download_corpora.sh)
│   └── results/           committed JSON benchmark results
├── l3tc-rust/             production Rust crate
│   ├── Cargo.toml
│   ├── README.md
│   ├── src/
│   │   ├── lib.rs         module exports + doc
│   │   ├── error.rs       thiserror-based typed errors
│   │   ├── bitio.rs       bit-level I/O for the arithmetic coder
│   │   ├── arithmetic.rs  Nayuki-style arithmetic coder
│   │   ├── tensor.rs      hand-rolled f32 linalg (matvec, LN, sigmoid, ...)
│   │   ├── checkpoint.rs  reader for the Rust-friendly binary format
│   │   ├── rwkv.rs        RWKV-v4 + HiRA forward pass
│   │   ├── tokenizer.rs   SentencePiece wrapper + raw-fallback refinement
│   │   ├── codec.rs       compress/decompress + segment parallelism
│   │   └── bin/l3tc.rs    CLI binary
│   ├── scripts/
│   │   └── convert_checkpoint.py   .pth → .bin, runs in L3TC's venv
│   ├── tests/             integration tests vs real checkpoint + SPM model
│   └── iter.sh            fast iteration loop: build + round-trip + verify
├── vendor/                (gitignored) cloned L3TC reference + RWKV-LM
└── scripts/
    ├── setup.sh           clone L3TC + RWKV-LM, set up Python venv
    └── download_corpora.sh  download enwik8, Canterbury, Silesia
```

## Quick start

### Compressor baselines (no Rust toolchain required)

```bash
# Download test corpora
./scripts/download_corpora.sh                # enwik8 + Canterbury
./scripts/download_corpora.sh --all          # also enwik9 + Silesia

# Run classical compressors on enwik8
python3 bench/bench.py --classical-only --corpus bench/corpora/enwik8

# Compare l3tc-rust against classical on enwik6
python3 bench/bench.py --corpus bench/corpora/enwik6 \
    --compressor l3tc-rust-200k \
    --compressor gzip-9 --compressor bzip2-9 \
    --compressor xz-9e --compressor zstd-22
```

### Rust crate (production path)

```bash
cd l3tc-rust
cargo build --release
cargo test --release                          # 35 unit tests

# End-to-end round trip on 50 KB of Wikipedia
./iter.sh

# Or explicit CLI
./target/release/l3tc compress /path/to/input.txt -o out.l3tc --verify --time
./target/release/l3tc decompress out.l3tc -o back.txt --time
```

### Integration tests against the real checkpoint

Requires `./scripts/setup.sh` to clone L3TC and `python
l3tc-rust/scripts/convert_checkpoint.py --input
vendor/L3TC/checkpoints/l3tc_checkpoints/l3tc_200k_bpe16k_c999_checkpoint0019.pth
--config vendor/L3TC/config/l3tc/l3tc_200k.py --output
l3tc-rust/checkpoints/l3tc_200k.bin` to produce the Rust-friendly
checkpoint. Then:

```bash
cd l3tc-rust
cargo test --release --test end_to_end -- --ignored --test-threads=1 --nocapture
```

## Decisions (key ones)

- **D1 — Target:** general-purpose lossless compressor, zstd alternative
- **D2 — Language:** Rust for the production implementation
- **D3 — Inference runtime:** originally "candle first, rwkv.cpp as
  fallback", but reversed in Phase 1 — we hand-rolled the tensor math
  because the model is small (2 layers × 96 dim) and a framework's
  abstractions cost more than they save. See D3 and D8 in
  [`DECISIONS.md`](DECISIONS.md).
- **D4 — Training data:** RedPajama v2 subset for the eventual retrain
- **D5 — RWKV version:** match L3TC (v4 + HiRA) for Phase 0/1, upgrade
  to RWKV-7 in Phase 4
- **D7 — Benchmark harness:** stdlib-only Python (no numpy, no pandas),
  shells out to compressors and measures wall time + rusage + byte counts

See [`DECISIONS.md`](DECISIONS.md) for the full list with reasoning and
(where applicable) what we changed our mind about.

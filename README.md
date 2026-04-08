# l3tc-prod

**A production-ready learned lossless compressor built on L3TC's ideas.**

Target: a zstd alternative that uses a small RWKV-based language model
with arithmetic coding to achieve better compression ratios than classical
codecs, at single-stream speeds that are practical for general use.

## Status

**Phase 0 — reproducing L3TC with solid engineering foundations.**

We're establishing a rigorous baseline before touching the core algorithm:
reproducing L3TC's reported enwik9 numbers, measuring real single-stream
speeds (not the batched numbers the paper headlines), and comparing
against the full classical-compressor lineup under identical conditions.

See [`PHASE_0.md`](PHASE_0.md) for the detailed plan and current progress.
See [`ANALYSIS.md`](ANALYSIS.md) for the full project thinking.
See [`DECISIONS.md`](DECISIONS.md) for the architectural decision log.

## Why this project exists

L3TC (AAAI 2025) is the first learned lossless compressor that looks
like it could plausibly compete with classical codecs in production.
The paper reports 16% compression ratio on enwik9 (vs 32% for gzip,
~21% for zstd) with "megabytes per second" decoding speeds.

But the paper's speed numbers come from batched inference at batch size
128+ on GPU. At batch size 1 on CPU — which is how real compression
workloads actually run — L3TC drops to 11-27 KB/s, slower than gzip by
roughly 5000×. That's the gap we're closing.

The research is done; the engineering isn't. Closing the gap is mostly
systems work: replacing Python/PyTorch inference with a custom runtime,
designing a real file format, shipping as a single static binary,
handling adversarial inputs gracefully. None of it is novel research.
All of it is necessary for anything that claims to be a compressor.

## Roadmap

| Phase | What | Status |
|-------|------|--------|
| **0** | Reproduce L3TC, build benchmark harness, baseline against classical codecs | in progress |
| 1 | Port inference from PyTorch to Rust (candle / rwkv.cpp), measure single-stream speedup | next |
| 2 | Design binary file format with magic bytes, versioning, checksums | |
| 3 | CLI + single-binary distribution | |
| 4 | Retrain with RWKV-7 on RedPajama for better ratio and OOD coverage | |
| 5 | Fuzz testing, robustness hardening, adversarial input handling | |
| 6 | C ABI, language bindings, ecosystem integration | |

## Layout

```
l3tc-prod/
├── README.md          this file
├── ANALYSIS.md        full project thinking document
├── DECISIONS.md       architectural decision log
├── PHASE_0.md         detailed Phase 0 plan and status
├── bench/
│   ├── bench.py       main benchmark harness
│   ├── compressors.py compressor wrappers
│   ├── corpora/       test corpora (gitignored, use download_corpora.sh)
│   └── results/       JSON benchmark results
├── scripts/
│   ├── setup.sh           set up L3TC and its dependencies
│   └── download_corpora.sh  download enwik8, enwik9, Silesia, Canterbury
└── .gitignore
```

## Quick start

```bash
# Download test corpora (enwik8 ~100 MB, enwik9 ~1 GB)
./scripts/download_corpora.sh

# Run the benchmark harness on classical compressors only (fast)
python3 bench/bench.py --classical-only --corpus enwik8

# Run L3TC baselines (requires scripts/setup.sh first, which installs
# Python dependencies and clones the L3TC reference implementation)
./scripts/setup.sh
python3 bench/bench.py --corpus enwik8

# Run everything on all corpora and save results
python3 bench/bench.py --all --output bench/results/$(date +%Y%m%d).json
```

## Decisions pinned so far

- **Target use case:** general-purpose lossless compressor, zstd alternative
- **Implementation language (for production):** Rust
- **Inference runtime (for production):** candle (Rust-native ML) with
  rwkv.cpp via FFI as fallback
- **Phase 0 language:** Python (L3TC is in Python, reproducing it
  requires running Python)
- **Training data for Phase 4 retrain:** RedPajama v2 subset
- **RWKV version:** match L3TC's choice for Phase 0 reproduction, upgrade
  to RWKV-7 in Phase 4
- **Canonical RWKV source:** [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM)

See [`DECISIONS.md`](DECISIONS.md) for the reasoning on each.

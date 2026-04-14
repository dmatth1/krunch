# Phase 0 — Reproduce L3TC with solid engineering foundations  ✅ COMPLETE

Established a rigorous, reproducible baseline for the project by benchmarking L3TC and all classical compressors on identical hardware. Key finding: L3TC-3.2M runs only 1.23x slower than L3TC-200K despite 11x more compute, meaning framework overhead is ~97% of runtime on the 200K model. This motivated the Phase 1 Rust rewrite.

See `docs/phase_0_findings.md` for the full analysis.

## Classical baselines on enwik8

| Compressor | Ratio | Compress MB/s | Decompress MB/s |
|---|---:|---:|---:|
| xz-9e | 24.83% | 1.1 | 83 |
| zstd-22 | 25.27% | 0.93 | 780 |
| xz-6 | 26.67% | 5.8 | 262 |
| zstd-19 | 26.94% | 2.2 | 753 |
| bzip2-9 | 29.01% | 18 | 39 |
| zstd-3 | 35.45% | 450 | 843 |
| gzip-9 | 36.48% | 24 | 546 |
| gzip-6 | 36.55% | 29 | 521 |

## L3TC on enwik6

| Model | Ratio | Speed (CPU batch=1) |
|---|---:|---:|
| L3TC-200K | 16.65% | 13.24 KB/s |
| L3TC-3.2M | 13.09% | 10.76 KB/s |

## Key deliverables

- `bench/bench.py` — benchmark harness with JSON output
- `bench/compressors.py` — wrappers for gzip/bzip2/xz/zstd/l3tc
- `bench/results/enwik8-classical.json`, `bench/results/l3tc-enwik6-baseline.json`
- `scripts/setup.sh`, `scripts/download_corpora.sh`
- `docs/phase_0_findings.md`

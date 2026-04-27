# bench/ — historical Krunch CLI benchmarks

Everything in this directory predates the 2026-04-21 pivot from the
krunch CLI compressor project to Krunch (managed storage
service). These results are kept because they back the "ratio wins
from specialization" argument in the service pitch — *not* because
the harness or corpora are still in active use.

- **No service code reads any of these results.**
- **No active benchmarks run against these corpora.** The service
  measures ratios on *customer data*, via
  `scripts/measure_held_out_ratio.py` emitted as part of each
  training job's `metadata.json` (see `TRAINING_FLOW.md`).
- Test-corpus downloaders under `scripts/download_corpora.sh` still
  exist but are only useful for reproducing these historical numbers.

## What's in here

All results-only. The harness and loader code (`bench.py`,
`compressors.py`, `build_*.py`, `detection_eval.py`) were removed
during the 2026-04-21 cleanup — they referenced corpora and Krunch CLI
paths that no longer exist in this repo. Full executable history is
still in the pre-pivot commits (before `698ad88`).

| file | contents |
|---|---|
| `corpora/README.md` | Canonical lossless-compression test corpora metadata (enwik6/8/9, Silesia, Canterbury). Corpora themselves are gitignored. |
| `results/enwik6-classical.*` | zstd / bzip2 / xz / gzip on 1 MB English Wikipedia |
| `results/enwik6-phase2.*`, `results/enwik6-phase2-final.*` | Krunch-200K Rust runtime on enwik6 during Phase 2 |
| `results/enwik8-classical.*` | Classical compressors on 100 MB Wikipedia |
| `results/enwik8-krunch.*` | Krunch-200K Rust runtime on enwik8 |
| `vs_traditional.md`, `vs_traditional.json` | Side-by-side tables vs. zstd / bzip2 / xz |
| `zstd_reference.json` | zstd reference points |
| `v010_bt_and_zstd22.txt` | Bytes-per-token + zstd-22 numbers from the v0.1.0 release gate |

## Why these numbers still matter

Historical Krunch-200K on enwik8 (English Wikipedia) hit ~0.20 ratio vs
zstd-22's ~0.26 — ~23% tighter on heterogeneous English text. The
service thesis is the same compression tech on *more homogeneous*
data (one customer's logs) should do meaningfully better — 2-4× tighter
per `STORAGE_SERVICE.md`. Spike 1 is the first real test of that
thesis on customer-shape data; see `SPIKE_1_LOG.md`.

## If you want to reproduce

The Krunch Rust runtime under `krunch-rust/` still builds and runs; its
README has the cargo commands. The corpus-loader + compressor wrapper
scripts were removed in the cleanup — check out the pre-pivot git
history if you need them.

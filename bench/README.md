# bench/ — historical L3TC CLI benchmarks

Everything in this directory predates the 2026-04-21 pivot from the
l3tc-prod CLI compressor project to Krunch (managed storage
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

| file | contents | status |
|---|---|---|
| `corpora/README.md` | Canonical lossless-compression test corpora (enwik6/8/9, Silesia, Canterbury) | historical |
| `results/enwik6-classical.*` | zstd/bzip2/xz/gzip on 1 MB English Wikipedia | historical evidence |
| `results/enwik6-phase2.*`, `enwik6-phase2-final.*` | L3TC-200K Rust runtime on enwik6 during Phase 2 | historical evidence |
| `results/enwik8-classical.*` | Classical compressors on 100 MB Wikipedia | historical evidence |
| `results/enwik8-l3tc.*` | L3TC-200K Rust runtime on enwik8 | historical evidence |
| `vs_traditional.md` | Side-by-side table vs. zstd/bzip2/xz | historical evidence |
| `detection_eval.md`, `detection_eval_findings.md` | Phase 14 domain detection for specialist SPM routing | obsolete — the service uses per-dataset tokenizers, not domain routing |

## Why these numbers still matter

Historical L3TC-200K on enwik8 (English Wikipedia) hit ~0.20 ratio vs
zstd-22's ~0.26. That's ~23% tighter on heterogeneous English text.
The service thesis is the same compression tech on *more homogeneous*
data (one customer's logs) should do meaningfully better — 2-4× tighter
per `STORAGE_SERVICE.md`. Spike 1 is the first real test of that
thesis on customer-shape data; see `SPIKE_1_LOG.md`.

## If you want to reproduce

The L3TC Rust runtime under `l3tc-rust/` still builds and runs; its
README has the cargo commands. Corpora come from
`scripts/download_corpora.sh` (edit paths in `bench/` as needed).

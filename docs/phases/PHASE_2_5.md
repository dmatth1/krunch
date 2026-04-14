# Phase 2.5 — Aggressive speed optimizations  ✅ CLOSED (partial)

116 KB/s compress, 121 KB/s decompress on enwik6 (7.4x Python). +30% compress / +31% decompress over Phase 2 with ratio essentially unchanged at 0.2061. Two of three planned items shipped; the third deferred.

## Results

| | Compress | Decompress | Ratio |
|---|---:|---:|---:|
| Phase 2 baseline | 89 KB/s | 92 KB/s | 0.2060 |
| + 2.5a NEON blocks | 97 KB/s | 109 KB/s | 0.2060 |
| + 2.5b INT8 head | **116 KB/s** | **121 KB/s** | 0.2061 |

## What shipped

- **2.5a** (commit `e0ade41`) — Hand-tuned NEON `matvec_96x96` for the 12 block projections per token. All 96 x elements preloaded into 24 `float32x4_t` registers, 4-way accumulator split. +9 KB/s.
- **2.5b** (commit `e8042d8`) — INT8 head quantization with per-column symmetric scales. Quantized in-memory at load time. Head memory traffic drops 4x (6.3 MB to 1.6 MB per token). +19 KB/s. Ratio impact +0.0001.

## Deferred

- **2.5c** — Vectorized `cum_freqs`. A prefilter attempt broke autovectorization and net-regressed. The planned top-K approach carries real ratio risk and its upside shrank after 2.5b moved the bottleneck. Deferred.

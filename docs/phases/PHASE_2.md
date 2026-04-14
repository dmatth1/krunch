# Phase 2 — Improve ratio and push speed further  ✅ COMPLETE

Modest gains over Phase 1: 89 KB/s compress (+5%), ratio 0.2060 (-1.6 pp) on enwik6. The deeper changes required for the 0.17 ratio target and 150+ KB/s speed target were deferred to Phase 2.5. See `docs/phase_2_findings.md`.

## Key results

| Metric | Phase 1 | Phase 2 |
|---|---:|---:|
| Compress speed (enwik6) | 85 KB/s | 89 KB/s |
| Ratio (enwik6) | 0.2094 | 0.2060 |
| enwik8 subset (10 MB) | — | 88 KB/s, 0.2216 |

## Root cause of the ratio gap identified

The f32-precision `scale` calculation in `logits_to_cum_freqs_scratch` loses precision when `target_total ~ 2^62` exceeds f32's 24-bit mantissa. Fix options: f64 for scale math, or smaller target_total with u32 freqs.

## Per-token profile (measured)

| Stage | us/token | % |
|---|---:|---:|
| Forward pass (total) | 193 | 75% |
| cum_freqs | 68 | 24% |
| Tokenize | 0.9 | 0.3% |
| AC encode | 1 | 0.4% |
| **Total** | **~262** | |

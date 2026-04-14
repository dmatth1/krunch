# Phase 3 — File format hardening, stream API, real benchmarks  ✅ MOSTLY SHIPPED

v3 file format with CRC32 + magic bytes, streaming encode + raw-store streaming decode, binary input support, full enwik8 baseline (110.65 KB/s / 0.2166). Multi-platform release builds (3d) deferred to Phase 6; true tokenized streaming decode and UTF-8 robustness fixes deferred to Phase 4a.

## Key results

| Corpus | Compress | Decompress | Ratio |
|---|---:|---:|---:|
| enwik8 (100 MB) | 110.65 KB/s | 117.50 KB/s | 0.2166 |

## What shipped

- **3a** (commit `641dd19`) — File format v3: magic bytes, version field, CRC32 integrity trailer. Reader accepts v2 for backward compat.
- **3b** (commit `e465d03`) — Streaming encode via `encode_reader` with bounded ~4 MB carry buffer, parallel batches via rayon, `N_SEGMENTS_IMPLICIT` framing (no `Seek` required).
- **3b-decode** (commit `21929bd`) — Streaming decode for raw-store path with rolling 8-byte tail buffer, incremental CRC validation.
- **Binary input** (commit `3f5bc0d`) — `FLAG_RAW_STORE` for non-UTF-8 files: encoder probes first batch, binary files get piped verbatim through v3 framing.
- **3c-lite** (commit `6dd2296`) — Full enwik8 baseline committed to `bench/results/enwik8-l3tc.md`.
- Silesia text + binary measured (numbers in `docs/phase_3_findings.md`).

## Known gaps deferred

- True tokenized streaming decode (compressed body still slurped for tokenized files)
- Bug A: stray non-UTF-8 byte poisons whole file to raw-store (dickens)
- Bug B: mid-stream UTF-8 failure crashes encode_reader (reymont, xml)
- 3d multi-platform release builds (moved to Phase 6)

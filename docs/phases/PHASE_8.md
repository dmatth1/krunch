# Phase 8 — Multi-model dispatch + specialist model registry

**Status:** back-burner. Addresses the out-of-distribution failure mode structurally.

**Goal:** instead of shipping one universal model that fails on non-Wikipedia text, ship several small specialist models and dispatch to the right one per file. This is the core mechanism for achieving universal text and structured text compression.

## The problem

A 200K-param model trained on enwik8 gets ratio 0.17 on Wikipedia prose and 1.26 on webster. The fix is multiple models and a dispatcher -- how cmix, NNCP, and paq8 all generalize.

## Deliverables

1. **Specialist training pipeline.** Same architecture as L3TC-200K (or v7 from Phase 5), trained per data family:
   - English prose (what we have now)
   - Source code (e.g. The Stack)
   - Structured data (JSON/YAML/CSV)
   - Logs (OpenStack, HDFS, BGL)
   - Markup (HTML/XML/Markdown)
   - Non-English prose (multilingual mix)
   - Each specialist ~200 KB INT8, bundle of 6-8 specialists ~1-2 MB total.

2. **Content-type detection.** Per-file classifier: file extension + byte-frequency histogram, or a tiny classifier on the first N bytes. Ambiguous files fall back to "generic English".

3. **File format extension (v5).** `model_id` field in header (or per-segment for segment-level dispatch). Reader loads the right model.

4. **Model registry.** Local manifest `(name, hash, size, path)`. CLI `--model auto` enables dispatch; `--model code` etc. forces specific specialist.

5. **Classical fallback cascade.** The last tier -- add `zstd-rs` as a dependency, `FLAG_CLASSICAL_FALLBACK` header flag, fall through to zstd-19 when no specialist fits. Guarantee: `final_size = min(specialist_size, zstd_size)`.

## Relationship to Phases 5 and 11

Three orthogonal model levers:
- **Phase 5:** better architecture (v4 -> v7), same corpus, same params
- **Phase 11:** broader corpus (enwik8 -> The Pile / RedPajama), one model
- **Phase 8 (this):** multiple specialist models + dispatch

If Phase 11's broader model covers enough distributions, Phase 8 reduces to just the classical fallback cascade. Phase 8 is higher complexity but higher upside because specialist models on homogeneous corpora beat general-purpose models dramatically.

## Success criteria

- l3tc-rust beats zstd-19 on at least 10 of 12 Silesia files without falling back to classical
- enwik6 ratio stays at or below current 0.1699
- CLI `--model auto` is the default; explicit specialist selection works
- Total shipped weight under ~5 MB

# Phase 8 — Multi-model dispatch + specialist model registry

**Status:** back-burner. Addresses the OOD failure mode
structurally: instead of shipping one universal model and hoping
the distribution fits, ship several small specialist models and
dispatch to the right one per file or per segment.

**The problem.** A 200K-param model trained on enwik8 is
excellent on Wikipedia prose (ratio 0.17) and terrible on
anything else (webster 1.26, reymont/xml failing, binary files
going raw-store). Phase 4b2 fixed the crash behavior via the
extraction-to-unk path, and Phase 4/Phase 5 close the in-
distribution gap, but nothing fixes "this model has the wrong
priors for this data". The fix is multiple models and a
dispatcher.

This is how most real-world learned compressors generalize: cmix
has dozens of context models, NNCP uses an online-learned mix of
specialists, paq8 has hundreds of submodels. We don't need the
full ensemble — a handful of specialists per major text family
would cover 90% of real workloads.

## Concrete deliverables

1. **Specialist training pipeline.** Same architecture as
   L3TC-200K, trained from scratch on each major data family:
   - English prose (Wikipedia-trained, what we have now)
   - Source code (a code corpus like The Stack)
   - Structured data (JSON/YAML/CSV mix)
   - Logs (production log mix — OpenStack, HDFS, BGL)
   - Markup (HTML/XML/Markdown)
   - Non-English prose (starting with a multilingual mix)
   - Optional: domain-specific (genomic reads, chemistry SMILES)

   Each specialist is ~800 KB of f32 weights converted to ~200 KB
   INT8 (Phase 7 prerequisite). A bundle of 6-8 specialists is
   ~1-2 MB of total ship weight.

2. **Content-type detection.** Per-file or per-segment
   classifier. First pass: file extension + shallow heuristics
   (byte-frequency histogram → family). Second pass: a tiny
   classifier that reads the first N bytes and outputs a
   specialist ID. Ambiguous files get the "generic English" fall-
   back.

3. **File format extension (v5).** Add a `model_id` field to the
   header (or per-segment if we want segment-level dispatch).
   Model id is a short hash or an index into a shipped registry.
   Reader loads the right model; writer records which one was
   used.

4. **Model registry format.** Local manifest file listing
   `(name, hash, size, path)` for each shipped specialist. CLI
   `--model auto` enables dispatch; `--model text` / `--model
   code` / etc. forces a specific specialist for benchmarking.

5. **Fallback cascade.** If no specialist fits well (measured by
   a quick entropy estimate on the first few hundred tokens),
   fall through to the classical fallback that Phase 4c ships.
   Never silently use a bad specialist — always pick `min(best
   specialist, classical)`.

## Why Phase 8 not Phase 5

Phase 5 is "train one broader model on a bigger corpus" — same
architecture, same inference path, one file instead of several.
Phase 8 is "train N specialists and dispatch". They're
complementary: Phase 5 improves the fallback specialist; Phase 8
adds the dispatch infrastructure and the other specialists.

Phase 8 is higher engineering complexity but higher upside
because specialist models on homogeneous corpora hit ratios that
a broader model can't touch (this is the reason the service
vision works — custom-trained models per customer get
dramatically better ratios than general-purpose ones).

## Success criteria

- l3tc-rust beats zstd-19 on at least 10 of 12 Silesia files
  directly (without falling back to classical)
- enwik6 ratio stays at or below current 0.1699
- A new bench result file `bench/results/silesia-phase8.md`
  compares specialist dispatch to the Phase 4 single-model
  baseline
- CLI `--model auto` is the default; explicit specialist
  selection works for reproducibility
- Total shipped weight bytes stay under ~5 MB

## Non-goals

- Hundreds of submodels (cmix-style) — we're aiming for the
  6-8 specialists that cover real workloads, not the
  compression-benchmark maximum
- Per-token model mixing (ensemble prediction) — much more
  compute, bigger engineering lift, save for Phase 11+
- Online / adaptive model selection based on the stream so far
  — same reason
- Multilingual sub-specialists beyond a broad multilingual
  bucket

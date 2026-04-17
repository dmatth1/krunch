# Phase 14 — Mixture of Specialists

**Goal:** ship a CLI compressor that achieves L3TC-200K-class ratios
across all common text and structured-text domains, with
L3TC-200K-class speed (~130 KB/s on M-series, ~170 KB/s with rayon).

**Status as of 2026-04-17:** proposed. Phase 11's generalist 2L
experiment confirmed that a single 200K-param model cannot
generalize across domains without losing both ratio AND speed.
This phase replaces "one generalist" with "many specialists +
content-aware routing."

---

## Why this phase exists

Phase 11 tried two approaches to "ratio + breadth at speed":

| approach | enwik6 ratio | python ratio | speed | verdict |
|---|---:|---:|---:|---|
| L3TC-200K (Wiki specialist) | **0.170** | 0.473 | **130 KB/s** | great on prose, bad elsewhere |
| Exp D 6L × 32K generalist | 0.335 | 0.354 | 49 KB/s | balanced ratios but 2.7× slower |
| 2L × 32K balanced (Phase 11 final attempt) | 0.40 (mid-train) | 0.87 (mid-train) | 56 KB/s | worst of both worlds |

**The capacity ceiling:** 200K transformer params split across many
domains gets ~40K params per domain — too thin to model anything
well. 32K vocab to "help structured text" doubled head matvec cost
without giving structured text enough capacity to compensate.

**The fix:** stop spreading 200K params thin. Each specialist gets
the full 200K (and a domain-tuned 16K vocab). Content-aware router
picks the right one. Both ratio AND speed match L3TC-200K's
profile on each specialist's domain.

---

## Architecture

### Specialists (Tier 1: must-have for v1)

Each specialist is **2L × 96H × 16K vocab** (same as L3TC-200K).
Trained on its own domain corpus with a domain-trained tokenizer.

| specialist | covers | training corpus | tokenizer | target ratio | target speed |
|---|---|---|---|---:|---:|
| **prose** | English text, books, articles, docs, emails (body) | enwik8 + Books3 + Wikipedia + ArXiv (15 GB) | enwik8 SPM 16K | **~0.17** on enwik6 | **130 KB/s** |
| **code** | Python, JS/TS, Java, C/C++, Go, Rust, Ruby, etc. | nick007x curated, all major langs (15 GB) | code-specific 16K | **~0.20** on python_source | **130 KB/s** |
| **structured** | YAML, JSON, TOML, XML, INI, .env | lumees configs + GitHub config files (10 GB) | structured-specific 16K | **~0.10-0.15** on JSON | **130 KB/s** |
| **logs** | syslog, nginx/apache, app logs, JSON logs, stack traces | Zenodo Loghub full + nginx samples (10 GB) | log-specific 16K | **~0.05-0.10** on syslog | **130 KB/s** |
| **tabular** | CSV, TSV (column-oriented data) | data.gov + NYT + GitHub CSVs (10 GB) | CSV-specific 16K | **~0.10** on CSV | **130 KB/s** |
| **markup** | HTML, Markdown, LaTeX | Common Crawl HTML + GitHub markdown (10 GB) | markup-specific 16K | **~0.20** on HTML | **130 KB/s** |
| **fallback** | mixed/unknown content | balanced 51 GB corpus from Phase 11 | balanced 16K | best-effort | **130 KB/s** |

**Total: 7 specialists, ~13 MB each = ~90 MB binary footprint.**

### Specialists (Tier 2: nice-to-have, v1.1+)

| specialist | rationale | priority |
|---|---|---|
| **sql** | Queries vs dumps have different patterns; both compressible | medium |
| **diff** | git/patch output, highly compressible (lots of context lines repeat) | medium |
| **multi-lang prose** | Non-English text (Chinese/Japanese/Spanish/etc) | low — depends on user demand |

### Out of scope

- **Email/mbox** — niche format, body uses prose specialist anyway
- **Binary files** — existing `raw_store` fallback handles these
- **Already-compressed** (ZIP, JPEG, etc.) — `raw_store` or zstd
  passthrough (Phase 8 hybrid dispatch covers this)

### Detection (router)

Cheap heuristics on the first 4 KB of input. Target <1 ms detection
time so it's negligible vs compression cost.

```
fn detect_domain(sample: &[u8]) -> (Domain, confidence: f32) {
    let text = String::from_utf8_lossy(sample);

    // High-precision strong signals first
    if starts_with_yaml_indicators(&text) || count_yaml_keys(&text) > 5 {
        return (Domain::Structured, 0.95);
    }
    if text.starts_with("{") && balanced_braces_score(&text) > 0.9 {
        return (Domain::Structured, 0.90);  // JSON
    }
    if text.starts_with("<?xml") || count_html_tags(&text) > 10 {
        return (Domain::Markup, 0.90);
    }
    if has_log_pattern(&text) {  // [INFO], 2024-01-01 12:34:56, IPs
        return (Domain::Logs, 0.85);
    }
    if column_consistency_score(&text) > 0.85 {  // consistent commas
        return (Domain::Tabular, 0.85);
    }
    if has_code_keywords(&text) > threshold {  // def/class/function/import/{}
        return (Domain::Code, 0.80);
    }
    if natural_language_score(&text) > 0.75 {
        return (Domain::Prose, 0.85);
    }

    // Fallback for low-confidence detection (mixed content, niche formats)
    (Domain::Fallback, 0.5)
}
```

Router heuristics tested against a labelled corpus of 1000+ files
across all domains. Target ≥95% accuracy on common types,
≥80% on edge cases.

### File format

Add 1 byte for `model_id` to the `.l3tc` header:

```
[4 bytes magic "L3TC"][1 byte version][1 byte model_id][2 bytes flags][...payload...]
```

`model_id` enum:
- 0: prose
- 1: code
- 2: structured
- 3: logs
- 4: tabular
- 5: markup
- 6: fallback
- 7-127: reserved for v2 specialists
- 128+: reserved for OOB user-supplied models

Decompression auto-loads the right model based on this byte. Users
never specify a model on decompress — fully transparent.

### CLI UX

**Default (auto-detect):**
```bash
l3tc compress big.json     # auto: structured specialist
l3tc compress code.tar     # auto: code specialist (or fallback if mixed)
l3tc compress book.txt     # auto: prose specialist
l3tc compress mystery.bin  # auto: fallback (or raw_store for non-text)
```

**Verbose mode (shows what was picked):**
```bash
l3tc compress -v big.json
> detected: structured (confidence 0.95)
> model: structured-2l-96h-16k
> 5.0 MB → 1.2 MB (ratio 0.24, 142 KB/s)
```

**Power user override:**
```bash
l3tc compress --model=prose file.txt
l3tc compress --no-detect file.txt  # alias for --model=fallback
l3tc compress --model=auto file.txt  # default behavior
```

---

## Training pipeline

### Per-domain corpus curation

Reuse the 51 GB diverse corpus from Phase 11, split by domain:

| specialist | source | size after dedup |
|---|---|---|
| prose | Pile slice (web, books, papers, articles) | ~40 GB |
| code | nick007x diverse code | ~5 GB |
| structured | lumees configs + structured | ~5 GB |
| logs | Zenodo Loghub | ~1 GB |
| tabular | data.gov CSV + GitHub CSVs | ~1 GB |
| markup | TBD — pull from Common Crawl + GitHub markdown | ~10 GB needed |
| fallback | full balanced 51 GB | — |

**New work:** scrape ~10 GB markup corpus (HTML from Common Crawl,
markdown from nick007x markdown split).

### Per-domain tokenizers

Train a 16K SPM unigram on each domain's corpus. ~5 min per
tokenizer (200 MB sample, single-threaded BPE limits).

Each tokenizer optimizes for its domain's token distribution:
- prose: word/subword tokens
- code: keyword + identifier tokens (`def `, `function `, `return`)
- structured: key tokens (`apiVersion:`, `kind:`, `{`, `}`)
- logs: timestamp/level tokens (`INFO `, `ERROR `, `2024-`, IPs)
- tabular: separator + common type tokens
- markup: tag tokens (`<div>`, `</p>`, `# `, `**`)

### Training each specialist

Same pipeline as Phase 11 but per-specialist:
1. Tokenize the domain corpus with the domain tokenizer (build .npy)
2. Upload .npy to S3
3. Launch training: 2L × 96H × 16K, 20 epochs, batch 32, on g5.xlarge
4. ~$10 per specialist on spot, ~$30 on-demand

**Total cost:** 7 specialists × $30 = ~$210 on-demand,
~$70 on spot. Add ~$30 for tokenizer training and corpus prep.

### Eval methodology

For each specialist, measure ratio + speed on:
1. Its own domain (the validation case — ratio should match the
   "specialist target" in the table above)
2. Every other domain (the cross-domain case — should be WORSE
   than other specialists, confirming the specialty actually
   matters)
3. Mixed-content files (the fallback case — fallback should win
   here)

Eval suite extension: add per-domain test files for each specialist
(15-20 files per domain, 100 KB to 5 MB each).

---

## Implementation roadmap

| step | effort | cost |
|---|---|---|
| 1. Curate per-domain corpora (split + dedup) | 1-2 days | — |
| 2. Source/scrape markup corpus (10 GB HTML/Markdown) | 1 day | — |
| 3. Train 7 domain-specific 16K tokenizers (~5 min each) | 0.5 day | — |
| 4. Build .npy caches for each domain (parallel re-tokenization) | 0.5 day | — |
| 5. Upload corpora + tokenizers + .npy to S3 | 0.5 day (network-bound) | — |
| 6. Train 7 specialists in parallel on spot | 1-2 days wall, $70-100 | $70-100 |
| 7. Per-specialist eval + cross-domain validation | 1-2 days | — |
| 8. Detection logic in Rust (~300 LOC + tests) | 2-3 days | — |
| 9. Build labelled detection test corpus (1000+ files) | 1 day | — |
| 10. File format extension (model_id byte) + decompression auto-load | 1 day | — |
| 11. CLI integration: `--model` flag, verbose output | 1 day | — |
| 12. Eval suite extension (15-20 files per domain) | 1 day | — |
| 13. Update README, COMPARISON, PHASE_11 closing notes | 0.5 day | — |
| **Total** | **~2 weeks dev + $70-100 GPU** | **$70-100** |

---

## Success criteria

Ship Phase 14 only if all of these hold:

1. **Per-specialist ratios match Phase 11 specialist targets** (within
   10%). E.g., prose specialist hits ≤0.19 on enwik6 (target 0.17).
2. **Per-specialist speed matches L3TC-200K** (≥120 KB/s on M-series
   multi-thread). The 16K vocab is the speed enabler — must be
   preserved.
3. **Detection accuracy ≥95%** on the labelled 1000-file test corpus
   for common domains; ≥80% on edge cases.
4. **Cross-domain regression check:** specialists must be NOTICEABLY
   worse on out-of-domain content than the right specialist would
   be (otherwise the specialization didn't actually specialize).
5. **Decompression works without user intervention** — the
   `model_id` byte routes correctly, all specialists round-trip
   bit-identically.

---

## Open questions

1. **Do we ship the 32K balanced unigram tokenizer or its model
   anywhere?** Probably no — Phase 11's "one model fits all"
   approach is being abandoned. The .npy and tokenizer can be
   deleted from S3 after Phase 14 lands. Worth keeping the model
   .pth as a baseline for "what NOT to ship" comparison.

2. **Mixed-content files (e.g., a tarball with prose + code).**
   Options:
   - Detection picks dominant content type, applies single specialist
     to whole file. Suboptimal but simple.
   - Per-segment detection (each 4 KB segment gets its own model).
     More complex header (model_id per segment). Better ratio.
   - Always use fallback for low-confidence detections. Simplest.

   v1: ship dominant-type routing; revisit per-segment if user
   feedback shows mixed content is common.

3. **HTML and Markdown together as one specialist?** They share
   structural markup patterns but differ a lot in content. Could
   split into two specialists later if the combined one
   underperforms. Start combined.

4. **Multi-language prose** (Chinese, Japanese, Spanish, etc.) is
   a Tier 2 problem. Single English-only prose specialist for v1.
   Add multi-lang specialist if there's demand.

5. **Should the detection model itself be learned (small
   classifier) instead of heuristics?** Heuristics simpler and
   debuggable; learned classifier maybe 5-10% more accurate.
   Defer learned approach to v1.1.

6. **Distillation as a stacking option:** could each specialist
   be distilled from a much bigger teacher to tighten ratio
   further at fixed inference cost? See Phase 12 "Ratio-preserving
   distillation" notes. Stacks cleanly with Phase 14 if pursued
   later.

---

## Relationship to other phases

- **Phase 8 (hybrid dispatch):** complementary. Phase 14 routes
  text to the right specialist; Phase 8 routes already-compressed
  / encrypted / random data to zstd or raw_store.
- **Phase 12 (CPU optimization):** orthogonal. Phase 12 made the
  200K/16K-vocab inference path fast (172 KB/s). Phase 14 ships
  multiple of those at the same speed.
- **Phase 13 (GPU backend):** orthogonal. GPU backend gives 1-3
  MB/s compress on prose; Phase 14 shipping multiple specialists
  means each is GPU-eligible too. Detection runs CPU-side in <1 ms.
- **Phase 11 (broader corpus):** sets up Phase 14. The 51 GB
  diverse corpus, balanced unigram tokenizer, and parallel re-
  tokenization scripts all transfer directly. Phase 11's 2L
  generalist run is closed (failed) — its data informed this phase
  but no model artifacts ship.

---

## Why "specialists at scale" is the right v1 product

It matches how zstd shipped: one tool, predictable behavior, clear
levers. We just have multiple models hiding behind the same CLI
instead of multiple parameter levels of one algorithm. From the
user's view: `l3tc compress file.json` Just Works™ at full
specialist speed and ratio.

It also matches how cmix and nncp shipped (single algorithm) but
beats them on speed by 5-50×. We're shipping a smarter compressor,
not a slower one.

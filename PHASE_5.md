# Phase 5 — Train on a broader corpus, optionally a bigger model

**Goal:** improve l3tc-rust's *generalization* — its ratio on
inputs that are not enwik8-style English Wikipedia prose. Phase 4
caps the OOD downside via a classical fallback; Phase 5 raises
the LM ceiling so the LM wins on more inputs in the first place.

**Starting point (assumed end of Phase 4):**
- enwik6 ratio ≤ 0.1715 (≈ Python L3TC-200K)
- enwik8 ratio ≤ 0.18
- 116 KB/s compress on enwik6, 110 KB/s on enwik8
- Hybrid classical fallback default-on, OOD inputs never larger
  than zstd-19's output

---

## Why broader training matters

The L3TC paper trains on enwik8. Our reproduction inherits that.
The result is that the model is excellent on Wikipedia prose
(0.16-0.21 ratios) and poor on:
- Structured English with custom formats (webster: 1.26 on the
  pre-fallback tokenized path)
- Source code (Silesia samba, ooffice, mozilla — these all need
  to fall back to classical today)
- Markup and structured data (XML, JSON, YAML)
- Non-English text
- Logs and machine-generated streams

A single model trained on a broader, well-curated corpus can
match the enwik8 ratio on Wikipedia *and* reach competitive
ratios on these other inputs without changing the runtime cost.

---

## 5a — Broader training corpus

**Candidates:**

1. **The Pile (deduplicated subset)** — 200 GB+, mixed-domain,
   well-known reference. Includes Wikipedia, books, GitHub code,
   StackExchange, ArXiv, web crawl, legal text, more. License is
   research-friendly. Used by every major open LM training run
   2020-2023.
2. **CommonCrawl-Text** (cleaned + deduplicated) — broader web
   coverage than Pile but messier. Useful for robustness, not as
   careful as Pile for quality.
3. **A bespoke domain mix** — e.g. 40% Wikipedia + 20% GitHub
   code + 20% books + 10% logs + 10% markup. Optimized for
   l3tc-rust's likely workloads. Cleanest, most work to assemble.

**Recommendation:** start with The Pile for the first run because
the data and tokenizer infrastructure already exist. Move to a
bespoke mix if Pile proves too noisy or too biased toward web
text.

**Tasks:**

1. Set up a data pipeline that yields shuffled text shards in
   the format L3TC's training script expects.
2. Train a fresh L3TC-200K-architecture model from scratch on
   the new corpus. Same tokenizer (SentencePiece BPE 16384) so
   our existing inference path keeps working without changes.
3. Convert the new checkpoint via `scripts/convert_checkpoint.py`
   and drop it next to the existing one.
4. Run the full bench suite (enwik6/8/9, Canterbury, Silesia text
   files, Silesia binaries via raw-store + classical fallback)
   against both the old and new checkpoints. Commit a side-by-side
   `bench/results/phase5-corpus-comparison.md`.
5. Measure how many Silesia files now win against zstd directly
   (without needing the fallback). The headline number is "fraction
   of files where l3tc-rust beats zstd-19".

**Compute estimate:** training a 200K-parameter RWKV on a few
billion tokens of The Pile is small enough to run on a single
A100 in ~24-72 hours, or distributed across cheaper GPUs over a
few days. Substantially less than the original L3TC paper
training cost.

**Success criteria:**

- enwik6 ratio is within 1 pp of Phase 4 (we're allowed to lose
  a little on the in-distribution case for general competence)
- enwik8 ratio is within 1 pp of Phase 4
- Webster tokenized ratio drops from 1.26 to ≤ 0.30
- l3tc-rust beats zstd-19 directly on at least 8 of 12 Silesia
  files (currently it beats zstd on enwik6/8 only)

---

## 5b — Optional: bigger model variants

L3TC ships 200K, 800K, 3.2M, and 12M parameter models. The Phase
0 finding showed 3.2M is only 1.23× slower than 200K *in
Python*, but in our hand-rolled Rust runtime the slowdown is
roughly linear in parameter count: 16× more compute → ~16× lower
throughput, so L3TC-3.2M would land at ~7 KB/s and L3TC-12M at
~2 KB/s. Way below the speed budget for default use.

**If we ever want to ship a "max ratio" mode**, the path is:

1. Wire the Rust runtime to load the L3TC-3.2M checkpoint (the
   converter already supports it).
2. Add a CLI `--ratio-mode` flag that opts in to the bigger model.
3. Document the speed cost up front.
4. Use it as a research tool for measuring what ratio the
   architecture *can* achieve, separately from what we ship by
   default.

**Not on the critical path.** Skip unless 5a leaves a real
quality gap that scaling parameters would close.

---

## Non-goals

- Training a brand new architecture (RWKV-v6, Mamba, etc.) —
  separate, much larger work.
- Online learning or per-file adaptation — interesting but
  Phase 7+ at the earliest.
- Multilingual tokenizer — the existing SentencePiece BPE 16384
  is English-biased; a multilingual rebuild is its own subproject.
- Distillation from a bigger model into a 200K — only if Phase
  5a doesn't move the numbers enough.

## Success criteria (Phase 5 exit)

Phase 5 is done when:
- A new L3TC-200K checkpoint exists, trained on a broader corpus
- l3tc-rust beats zstd-19 on ≥ 8 of 12 Silesia files
- enwik6/8 ratios stay within 1 pp of Phase 4
- Throughput unchanged (same architecture, same parameters)
- `docs/phase_5_findings.md` summarizes the corpus + training
  setup and the per-corpus ratio deltas

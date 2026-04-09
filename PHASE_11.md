# Phase 11 — Broader training corpus

**Goal:** retrain at the same parameter count (200K) on a
broader corpus than enwik8, measure the tradeoff (lose a little
on enwik6, win substantially on everything else), and decide
whether the broader model becomes the default or ships
alongside the enwik-trained specialist.

**Status:** back-burner. Downstream of Phases 5 (architecture
upgrade to v7) and 8 (specialist dispatch), but logically
independent — Phase 11 can run with v4 or v7, and its output
can be used as a standalone generalist or as one specialist
among Phase 8's dispatch targets.

---

## Why a broader corpus matters

Everything we've measured so far has been on enwik6/enwik8.
That's convenient because it's the training distribution and
we know the ratios are good. But it's also the *only*
distribution where the ratios are good. Phase 3's Silesia run
showed the cliff on webster / nci / reymont / xml; Phase 4b2's
unk-extraction patched the crash symptoms without fixing the
underlying prediction quality problem.

The model isn't bad at "English text" — it's bad at "English
text that's not Wikipedia". A dictionary format, a legal
corpus, production logs, markdown docs, code comments, emails
— each of those is a different enough distribution that a
Wikipedia-trained LM struggles. Broader training corpora are
the standard fix.

## Corpus candidates

Listed in rough order from "safe incremental" to "big leap":

1. **enwik9** (1 GB, same source as enwik8 but larger). Trivial
   generalization improvement. Same domain, so enwik6 ratio
   stays about the same; enwik9 ratio improves modestly. Cheap
   to try first as a sanity check.

2. **The Pile deduplicated subset** (~200-800 GB depending on
   which subset). Mixed-domain: Wikipedia, books, ArXiv, code,
   StackExchange, legal, web crawl, emails. Well-curated,
   research-friendly licensing, the go-to for open LM training
   runs 2020-2023. Expected tradeoff on enwik6: lose ~1-2 pp
   ratio in exchange for much better coverage on everything
   else. Probably the right starter broader corpus.

3. **RedPajama / SlimPajama** (~1.2 TB / ~600 GB). Newer and
   better-curated than The Pile. Larger, more diverse. Similar
   tradeoff profile but more compute-hungry.

4. **A bespoke domain mix.** E.g., 40% Wikipedia + 20% GitHub
   code + 20% books + 10% logs + 10% JSON/markup. Optimized for
   our likely workloads rather than for general language
   modeling. Most work to assemble but best for tailoring to a
   target use case (service vision customers, specific
   archival markets).

5. **Customer-specific corpora.** Single-customer data (from
   the storage service vision), trained per customer. The most
   extreme specialization — best ratio on that customer's data,
   useless for anyone else. Discussed in
   `STORAGE_SERVICE_VISION.md`.

---

## Tasks

1. **Pick a candidate corpus.** Recommend The Pile dedup
   (option 2) as the first serious broader-training run. It's
   the mainstream choice, the data is curated, and the license
   is friendly.

2. **Set up the training pipeline.** Starting point: upstream
   RWKV-LM training script (same one Phase 5 uses for v7).
   Swap the data loader for The Pile. Keep everything else
   identical: 200K parameters, same tokenizer, same optimizer,
   same epoch budget. The variable under test is the corpus,
   not the architecture or hyperparameters.

3. **Tokenizer question.** Our current SPM BPE 16384 is trained
   on enwik8. Broader corpora introduce vocab the tokenizer
   doesn't handle well (code keywords, non-English, rare
   scientific terms). Options:
   - Keep the enwik8 tokenizer and accept more `<unk>` tokens
     on the broader corpus (works but bleeds ratio).
   - Train a new SPM BPE 16384 on the broader corpus and use
     that (best ratio on the broader set, loses the ability to
     directly compare enwik6 numbers against prior phases).
   - Train a larger vocab (e.g., 32K BPE) — more parameters
     in the embedding and head, so not strictly 200K anymore.
   Start with option 1 for the first run to keep the
   comparison clean, then revisit.

4. **Train.** Compute budget: 200K params on The Pile dedup
   for ~10 epochs on a reasonable subset (say 50 GB sample) is
   a few days on a single A10G / L4. Larger subsets scale
   linearly.

5. **Evaluate on the full bench suite.** Run the Rust
   `entropy-bound` and actual-ratio measurements on:
   - enwik6 (the old baseline — expect modest regression)
   - enwik8 (ditto)
   - Canterbury corpus (mixed, different distribution)
   - Silesia text files (dickens / webster / nci / reymont /
     xml) — this is the real test of whether broader training
     actually generalizes
   - A sample of real log data if we can find some
   - A sample of real code if we can find some

6. **Decide:** does the broader model become the new default,
   stay as a "broader" opt-in alongside the enwik-trained
   default, or feed into Phase 8 as one specialist among
   several?

---

## Expected outcomes

Predictions before running (so we can check our intuition
afterward):

| corpus | enwik8-trained | broader-trained | delta |
|---|---:|---:|---|
| enwik6 | 0.1699 | ~0.180 | -0.01 pp (small regression) |
| enwik8 | ~0.2166 | ~0.220 | -0.005 pp |
| Canterbury | ~0.30 | ~0.25 | +0.05 pp (win) |
| webster (current 1.26) | 1.2613 | ~0.35 | +0.91 pp (huge win) |
| dickens (was raw-store) | 1.0000 | ~0.30 | +0.70 pp (huge win) |
| logs | never measured | ~0.25 | n/a |

If these predictions hold, the broader model is worse on
Wikipedia but MUCH better on everything else. That's the
tradeoff we'd want to understand before deciding the default.

If the broader model doesn't meaningfully close the gap on
webster/dickens/logs, then the problem isn't the corpus — it's
the parameter count. 200K is small. In that case Phase 11
produces a negative result pointing at Phase 5b (bigger model)
or Phase 8 (specialist dispatch, because no single model can
fit the distribution zoo at this parameter budget).

---

## Success criteria

- A new L3TC-200K checkpoint exists, trained on a broader
  corpus, converted to our Rust binary format, and loadable
  via the existing checkpoint reader
- Bench result in `bench/results/phase11-broader-corpus.md`
  with direct comparison against the enwik8-trained model on
  every corpus in the suite
- `docs/phase_11_findings.md` documents the corpus choice, the
  tokenizer decision, the training hyperparameters, the
  per-corpus ratio deltas, and the "default or opt-in" decision
- If the broader model ships as an alternative, the CLI's
  `--model` flag accepts the new model name

## Non-goals

- Training on proprietary customer data (that's the service
  vision, not this phase)
- Retraining the tokenizer from scratch on the broader corpus
  (defer unless Phase 11's first attempt shows the enwik8
  tokenizer is the bottleneck)
- Models larger than 200K (those are Phase 5b territory —
  bigger model at the same corpus, not broader corpus at the
  same size)
- Multilingual specialization beyond "include multilingual in
  the mix" (that's Phase 8 specialist dispatch)
- Distillation from a bigger model into the 200K (interesting
  but separate — could be Phase 12 if needed)

---

## Why this is its own phase and not folded into Phase 5 or 8

- **Not Phase 5** because Phase 5 is an apples-to-apples
  architecture comparison on the same training corpus. Mixing
  "v4 → v7" and "enwik8 → The Pile" in the same phase confounds
  the two variables; you can't tell which change caused which
  result.
- **Not Phase 8** because Phase 8 is the dispatch
  infrastructure and specialist-training pipeline. Phase 11 is
  a single broader-training experiment. If the broader model
  wins on enough corpora to be the new default, Phase 8 never
  needs to dispatch to a specialist — one broader model covers
  the distribution. If the broader model loses or barely ties,
  Phase 11's output becomes one input to Phase 8's specialist
  roster.

Phase 11 is the experiment that decides whether Phase 8 is
worth doing at all. It's independent of Phase 8 and should be
run first if generalization is the goal.

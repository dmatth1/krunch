# Phase 11 — Broader training corpus

**Goal:** retrain at the same parameter count (200K) on a
broader corpus than enwik8, with the speed and architecture
held constant, and measure whether a single broader-trained
model can fix the OOD cliff (webster, dickens, code, logs)
without giving up the in-distribution ratio that is the entire
reason anyone would reach for a learned compressor in the first
place.

## Hard constraints (do not relax these without an explicit re-plan)

These five constraints define what Phase 11 actually is. If a
proposal in this phase changes any of them, it's not Phase 11
anymore — it's a different phase that should be planned and
named separately.

1. **Same architecture.** RWKV-v4 + HiRA, 2 layers × 96 hidden,
   intermediate 96, rwkv_rank 4, vocab 16384. Don't touch the
   model class. If we want to test v7, that's Phase 5; mixing
   "v4 → v7" with "enwik8 → broader corpus" confounds two
   variables in one experiment.

2. **Same parameter count: 200K, fixed.** Don't grow it. If a
   broader corpus doesn't fit at 200K, **that is a real finding
   that points at Phase 5b (bigger model) as a separate
   experiment** — it is not a license to bump the param count
   inside Phase 11. The variable under test is the corpus, not
   the capacity. Bumping params would also break constraint #4.

3. **Broader corpus, pinned upfront.** The corpus composition
   (which datasets, in what proportions, how deduplicated) is
   pinned before training starts and recorded in
   `docs/phase_11_findings.md`. Reproducibility matters more
   than picking the optimal mix on the first try. See
   [Corpus candidates](#corpus-candidates) for the starting
   point (The Pile dedup subset).

4. **Same speed target.** The new 200K checkpoint must run at
   the same wall-clock speed as the current default tier
   (~131 KB/s on enwik6 on M-series CPU, ±5%). Same parameter
   count + same architecture means same FLOPs per token, so
   same speed. **If the new model is meaningfully slower
   something is wrong with the conversion or the runtime, not
   with the training** — investigate before shipping.

5. **In-distribution ratio is a hard floor — this is the
   blocker constraint.** Ratio is the entire benefit l3tc-prod
   offers over classical compressors. If we trade enwik6 ratio
   for OOD coverage and the result is "mediocre everywhere,"
   we have lost the only thing that justifies the speed cost
   versus zstd / xz / bzip2.

   **The hard floor: enwik6 actual coded ratio must remain
   ≤ 0.20** (current is 0.1699; this is roughly a 17% relative
   regression budget). At 0.20 we are still ~33% better than
   zstd-22 (0.30) and ~29% better than bzip2-9 (0.28), which
   is a comfortable margin and preserves the value prop. At
   0.22 the value prop weakens to "marginal ratio winner";
   at 0.25 we are roughly tied with bzip2 and the project no
   longer beats classical on the metric we care about.

   **If a Phase 11 candidate model exceeds the 0.20 floor on
   enwik6, the phase fails and the model does not ship**, even
   if the OOD numbers are excellent. The fallback in that
   case is to either (a) re-tune the corpus mix to give
   enwik more weight, or (b) accept that 200K cannot cover the
   broader distribution at acceptable in-distribution ratio
   and escalate to Phase 5b (bigger model) or Phase 8
   (specialist dispatch).

**Status:** decision gate for Phase 8. Logically independent of
Phase 5 (v4 or v7 both work as the underlying architecture)
but **ordered before Phase 8** because Phase 11's outcome
determines whether Phase 8 needs to exist at all.

**Why this is a gate, not just a phase:** Phase 8 (specialist
model dispatch + classical fallback) is ~2-3 months of work
that only pays off if **no single broader-corpus model can
cover the distributions we care about** at acceptable ratio.
If Phase 11 trains a 200K (or 3.2M) model on Pile / RedPajama
and that single model handles enwik / webster / code / logs
all within 10-20% of the enwik-specialized number, Phase 8 is
unnecessary — we ship the generalist as the default and the
OOD problem is solved. If Phase 11's broader model loses badly
on every non-Pile distribution, Phase 8 becomes mandatory and
we know exactly which specialists to train.

**Run Phase 11 before committing to Phase 8.** It saves
2-3 months of speculative dispatch-engineering work in the
"single broader model wins" case.

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

Predictions before running, so we can check our intuition
afterward and so the decision criteria above aren't a moving
target:

| corpus | current 200K | predicted Phase 11 200K | hard floor | comment |
|---|---:|---:|---:|---|
| enwik6 | 0.1699 | ~0.180–0.190 | **0.20** | small regression expected; must stay under floor |
| enwik8 | 0.1793 | ~0.190–0.200 | **0.21** | tracks enwik6 |
| Silesia/webster | 1.2613 | ~0.30–0.40 | — | the big win we're hoping for |
| Silesia/dickens | ~1.0 (raw-store) | ~0.25–0.35 | — | the big win we're hoping for |
| code (~10 MB) | not measured | ~0.30–0.45 | — | new ground |
| logs (~10 MB) | not measured | ~0.25–0.40 | — | new ground |

If these predictions hold, the broader model is slightly worse
on Wikipedia (within the 0.20 floor) and dramatically better on
everything else. That's the trade Phase 11 is buying.

**If the broader model doesn't meaningfully close the gap on
webster / dickens / logs / code, the problem isn't the corpus
— it's the parameter count.** 200K is small. In that case
Phase 11 produces a negative result pointing at Phase 5b
(bigger model at the broader corpus, separate experiment) or
Phase 8 (specialist dispatch, because no single 200K can fit
the distribution zoo). Either escalation is fine; what's not
fine is using Phase 11 as cover to silently bump the param
count.

**If the broader model closes the OOD gap but breaks the
in-distribution floor**, the same escalation logic applies.
Phase 11 has answered the question "can a single 200K model
cover everything?" and the answer is no. That is a useful
result. Don't ship it as Phase 11.

---

## Success criteria

Phase 11 ships only if **all** of the following are true. The
hard-constraint floor in §"Hard constraints" is the gate; the
matrix below is the ship condition.

### Engineering gates

- A new L3TC-200K checkpoint exists, trained on a broader
  corpus, converted to our Rust binary format, and loadable
  via the existing checkpoint reader
- Wall-clock compress speed on enwik6 is within ±5% of the
  current default tier (~131 KB/s on M-series). If it isn't,
  investigate the runtime or conversion before debugging the
  training
- All 40+ Rust unit tests pass; no regression in the existing
  200K default tier (which stays in the repo as the
  in-distribution specialist)

### Ratio matrix — the actual ship gate

Run the Rust runtime's `entropy-bound` and full-compress paths
on every corpus in the matrix and record both numbers:

| corpus | current 200K | Phase 11 200K | hard floor | win threshold |
|---|---:|---:|---:|---|
| enwik6 (1 MB) | 0.1699 | ? | **≤ 0.20** | unchanged |
| enwik8 (100 MB) | 0.1793 | ? | **≤ 0.21** | unchanged |
| Canterbury (mixed) | not measured cleanly | ? | — | meaningfully better |
| Silesia/dickens | ~1.0 (raw-store) | ? | — | ≤ 0.40 |
| Silesia/webster | 1.2613 | ? | — | ≤ 0.40 |
| Silesia/nci | TBD | ? | — | meaningfully better |
| code sample (~10 MB real source) | not measured | ? | — | ≤ 0.45 |
| log sample (~10 MB real logs) | not measured | ? | — | ≤ 0.40 |

**Phase 11 ships if:**

1. **The two hard floors hold:** enwik6 ≤ 0.20 AND enwik8 ≤ 0.21.
   These are non-negotiable. If either is exceeded, Phase 11
   fails regardless of how good the OOD numbers are.
2. **The OOD numbers improve substantially.** "Substantially"
   means: webster, dickens, code, and logs all hit ≤ 0.45 (vs
   current ≥ 1.0 on the ones we've measured) AND the
   *geometric mean of the eight rows* is at least 30% better
   than the current model's geomean across the same corpora.
3. The bench result is committed to
   `bench/results/phase11-broader-corpus.md` with the full
   table and the comparison against the current default.
4. `docs/phase_11_findings.md` documents the corpus mix, the
   tokenizer decision, the training hyperparameters, the
   per-corpus ratio deltas, the speed measurement, and the
   "default vs opt-in" decision.

### Possible outcomes and what each means

- **Both floors hold AND OOD wins land:** ship as the new
  default. The current 200K model becomes a "enwik specialist"
  opt-in tier (kept around for users who only compress
  Wikipedia-style text and want the absolute best ratio there).
  Phase 8 becomes unnecessary.
- **Floors hold but OOD wins are weak:** ship as an opt-in
  "broader" tier alongside the current default. The OOD problem
  is partially fixed; Phase 8 (specialist dispatch) is still
  on the table for the cases the broader model didn't cover.
- **Floors break (enwik6 > 0.20 or enwik8 > 0.21):** Phase 11
  fails. The 200K capacity floor is too tight to cover both
  Wikipedia and the broader distribution at our ratio
  requirements. Decision: escalate to Phase 5b (bigger model
  at the broader corpus) or Phase 8 (specialist dispatch with
  the current 200K as the enwik specialist + a different
  broader model as the generalist). Don't ship anything from
  Phase 11 in this case.
- **OOD wins land but floors break by a small margin** (e.g.,
  enwik6 lands at 0.205): re-tune the corpus mix to give
  enwik more weight and re-train once. If two re-tunes don't
  recover the floor, escalate as above.

## Non-goals

- **Increasing the parameter count above 200K.** This is the
  most tempting drift in the phase and the one that would
  silently invalidate the experiment. If the 200K model can't
  cover the broader corpus, that's a finding — escalate it as
  a separate Phase 5b experiment (bigger model at the broader
  corpus) instead of bumping params inside Phase 11. See
  constraint #2 in §"Hard constraints".
- **Changing the architecture** (adding layers, changing
  hidden, switching to v7, etc.). All of those confound the
  experiment with "did the corpus help or did the architecture
  help?" Phase 5 owns architecture changes; Phase 11 owns
  corpus changes; we run them separately so we can attribute
  the wins.
- **Sacrificing the in-distribution ratio floor** (enwik6
  > 0.20 or enwik8 > 0.21) in pursuit of OOD wins. This is
  the blocker constraint and it's non-negotiable inside this
  phase. See constraint #5 in §"Hard constraints".
- Training on proprietary customer data (that's the service
  vision, not this phase)
- Retraining the tokenizer from scratch on the broader corpus
  (defer unless Phase 11's first attempt shows the enwik8
  tokenizer is the bottleneck)
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

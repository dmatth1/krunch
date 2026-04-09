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

## Current progress

**Status as of 2026-04-09:** Phase 11 cloud training infrastructure
landed. Pass 1 (enwik9 sanity check) is ready to launch; Pass 2
(Pile dedup broader corpus) requires the corpus build step before
it can run. No training has been kicked off yet — the next action
is human review of this section, then `./scripts/launch-spot-fleet.sh
pass1`.

### Why cloud, not the MacBook

Phase 4e3 measured ~27 minutes for 5 MB × 2 epochs of L3TC training
on Apple Silicon MPS via the pure-PyTorch WKV monkey-patch in
`scripts/distill_l3tc.py`. Linear extrapolation:

| corpus | 2 epochs on MacBook MPS |
|---|---:|
| 5 MB (Phase 4e3) | 27 min ✅ done |
| 100 MB (enwik8) | ~9 hours |
| 1 GB (enwik9) | ~90 hours (~4 days) |
| 50 GB (Pile dedup, the actual Phase 11 target) | **~190 days** |

The bottleneck is structural: our WKV kernel is a `for t in range(T)`
Python loop because L3TC's CUDA WKV is GPU-only and we don't have a
custom MPS Metal kernel. That loop is the entire forward pass for
the recurrent dimension and there's no way to vectorize it across
the time axis without changing the WKV math. **Local training is
fine for distillation experiments on tens of MB; it is not fine for
training from scratch on tens of GB.**

On a single CUDA GPU using L3TC's native CUDA WKV kernel, the same
training runs ~50-100× faster (one fused CUDA op instead of 2048
Python iterations, fp16 mixed precision works, data loading
parallelizes). At 200K params and an A10G/L40S, the run is dominated
by data loading and the sequential time dimension, not raw compute.
A single GPU is plenty.

### Cost and wall-clock budget

| GPU | spot $/hr | Pass 1 (enwik9, 1 GB × 15 epochs) | Pass 2 (Pile, 50 GB × 10 epochs) |
|---|---:|---|---|
| g5.xlarge (A10G 24GB) | ~$0.40-0.80 | ~1-2 hours, **~$1-2** | ~6-10 hours, **~$3-8** |
| g6e.xlarge (L40S 48GB) | ~$0.80-1.30 | ~30-60 min, **~$1** | ~3-5 hours, **~$3-7** |

We default to **g6e.xlarge** for consistency with the bnn spot
fleet pattern (same instance type, same launcher pattern, same
IAM/SG/key). 200K params at L40S is overkill but the cost
difference is negligible at this scale and the consistency saves
infrastructure work. Total Phase 11 cloud spend including debug
iterations should land **under $20**.

### Infrastructure landed

Two scripts forked from the bnn pattern, adapted for L3TC training:

- **`scripts/launch-spot-fleet.sh`** — adapted from
  `bnn/scripts/launch-spot-fleet.sh`. Maintains 1 g6e.xlarge spot
  instance, self-healing on reclaim. Reuses the existing bnn AWS
  infrastructure verbatim:
  - Region: `us-east-1`
  - Key pair: `swarm-ec2`
  - Security group: `sg-0af8b62d12cf4272c` (`bnn-training`)
  - Instance profile: `bnn-s3-access` (already has access to the
    bucket via `s3://dmatth1-bnn-checkpoints/*`)
  - S3 path: **`s3://dmatth1-bnn-checkpoints/l3tc/<RUN_ID>/`** —
    we share the bnn bucket with an `l3tc/` prefix instead of
    creating a parallel bucket. The IAM role's wildcard policy
    transparently covers it. No new infrastructure required.
  - Tags: `l3tc-phase11-<RUN_ID>`, `l3tc-run-id`, `l3tc-pass`
  - Usage: `L3TC_GITHUB_PAT=ghp_... ./scripts/launch-spot-fleet.sh pass1 [RUN_ID]`

- **`scripts/spot-fleet-userdata.sh`** — instance bootstrap.
  Stateless: every fresh instance OR replacement after spot
  reclaim runs the same script. It clones l3tc-prod, runs
  `scripts/setup.sh` to clone vendor/L3TC + RWKV-LM, sets up the
  L3TC venv with CUDA-PyTorch, downloads the corpus from S3,
  tokenizes it with the existing SPM tokenizer (vocab 16384, the
  enwik8-trained one), checks for an existing `.pth` checkpoint
  for the run ID and resumes if present, then runs
  `vendor/L3TC/main.py` with the standard `config/l3tc/l3tc_200k.py`
  config and a `train_file=...` override pointing at our
  preprocessed corpus. Background loop uploads checkpoints + logs
  to S3 every 5 minutes so we can monitor and resume.

### Secret handling

The bnn userdata script has the GitHub PAT hardcoded into the
file (`GITHUB_PAT="github_pat_..."` at line 14 of
`bnn/scripts/spot-fleet-userdata.sh`). That works but means the
PAT is committed to git. **For l3tc-prod we do not commit the
PAT.** Instead, the launcher reads it from the
`L3TC_GITHUB_PAT` environment variable at launch time and
substitutes it into the userdata via `sed` before base64-encoding
into the EC2 user data. The userdata script ships with the
literal placeholder string `PLACEHOLDER_GITHUB_PAT`, never a real
secret.

To launch:

```bash
# One-time: create a fine-grained GitHub PAT scoped to dmatth1/l3tc-prod
# with read-only contents access. Save it somewhere local:
echo 'ghp_yourtokenhere' > ~/.l3tc-pat
chmod 600 ~/.l3tc-pat

# Per-launch:
export L3TC_GITHUB_PAT=$(cat ~/.l3tc-pat)
./scripts/launch-spot-fleet.sh pass1
```

### Pass 1 — enwik9 sanity check (READY TO LAUNCH)

**Goal:** verify the cloud training pipeline end-to-end on the
simplest possible corpus before touching the broader-corpus
experiment. enwik9 is 1 GB of the same source as enwik8 (Wikipedia
XML, March 2006 dump); training on it is "more in-distribution
data" with no other variables changed. Expected outcome: a
checkpoint that produces a slightly better enwik6 ratio than the
current default (more training data → marginally better fit) and
identical speed.

**Pre-launch checklist:**

- [x] Launcher + userdata scripts written
- [x] Reusing bnn AWS infrastructure verified (same IAM, SG,
      key pair, bucket; just `l3tc/` prefix)
- [x] Secret handling: GitHub PAT not committed to repo;
      injected at launch via env var
- [ ] enwik9 uploaded to
      `s3://dmatth1-bnn-checkpoints/l3tc/corpora/enwik9.xz`
      (one-time, do this once before first launch — see
      "Corpus upload" below)
- [ ] `L3TC_GITHUB_PAT` env var set in the launching shell
- [ ] Spot fleet quota verified (g6e.xlarge in us-east-1)

**Corpus upload (one-time, before first launch):**

```bash
# Local machine. enwik9 is 1 GB raw, ~322 MB xz-compressed.
curl -O http://mattmahoney.net/dc/enwik9.zip
unzip enwik9.zip
xz -9 enwik9
aws s3 cp enwik9.xz s3://dmatth1-bnn-checkpoints/l3tc/corpora/enwik9.xz
```

**Launch:**

```bash
export L3TC_GITHUB_PAT=$(cat ~/.l3tc-pat)
./scripts/launch-spot-fleet.sh pass1
# Captures FLEET_ID, RUN_ID, S3 paths, monitoring commands.
```

**Monitor:**

```bash
RUN_ID=phase11_pass1_<timestamp>
# Bootstrap log (instance setup, before training starts):
aws s3 cp s3://dmatth1-bnn-checkpoints/l3tc/${RUN_ID}/bootstrap.log - | tail -50
# Training log (after training starts):
aws s3 cp s3://dmatth1-bnn-checkpoints/l3tc/${RUN_ID}/train.log - | tail -50
# Latest checkpoint pulled back to local for inspection:
aws s3 sync s3://dmatth1-bnn-checkpoints/l3tc/${RUN_ID}/ ./checkpoints-pass1/ --exclude "*" --include "checkpoint*.pth"
```

**Pass 1 success criteria** (gate before Pass 2):

1. Bootstrap completes without errors (corpus downloaded,
   tokenized, training started). Reproducible on a fresh instance.
2. Training runs for the full 15 epochs without OOM or NaN loss.
3. Final `.pth` exists in S3 and converts cleanly via
   `l3tc-rust/scripts/convert_checkpoint.py` to a `.bin` that
   the existing Rust runtime loads.
4. Round-trip enwik6 with the new `.bin`: byte-identical, ratio
   within ±0.005 of the existing model (we expect a small
   improvement, but anything in that band confirms the pipeline
   produces a usable model). Speed within ±5% of 131 KB/s.

If all four hold, the pipeline is verified and we move to Pass 2.
If any fail, debug Pass 1 before spending Pass 2's compute.

### Pass 2 — Pile dedup broader corpus (NOT YET READY)

**Blocker:** corpus build step. Pass 2 needs a single concatenated
text file of the Pile dedup subset (~50 GB) uploaded to
`s3://dmatth1-bnn-checkpoints/l3tc/corpora/pile_dedup_50gb.tar`.
Building it requires:

1. A `scripts/build_pile_corpus.py` (not yet written) that pulls
   the Pile deduplicated subset from HuggingFace datasets,
   selects ~50 GB of well-distributed shards, concatenates them
   with newline separators into a single text file, and uploads
   it to S3.
2. ~30-60 GB of local or instance disk for the build.
3. ~30-60 min of HuggingFace download time on a fast connection.

This step is intentionally deferred until Pass 1 succeeds — no
point assembling the Pile if the cloud training pipeline doesn't
work for enwik9 first.

Once Pass 1 succeeds, the Pass 2 work list is:

- [ ] Write `scripts/build_pile_corpus.py`
- [ ] Run it locally or on a one-off EC2 instance to assemble
      `pile_dedup_50gb.tar` and upload to S3
- [ ] Pin the corpus composition in `docs/phase_11_findings.md`
      (which Pile shards, dedup version, ordering, total bytes)
- [ ] Launch Pass 2: `./scripts/launch-spot-fleet.sh pass2`
- [ ] Monitor, pull checkpoint, run the ratio matrix benchmark
      (the eight corpora in §"Success criteria"), apply the hard
      floors, write `docs/phase_11_findings.md`

### What's NOT done yet

- Pass 1 has not been launched. Awaiting human review of
  this section + the launcher/userdata scripts.
- Pass 2 corpus build script not written.
- enwik9 not yet uploaded to S3 (one-time prereq for Pass 1).
- The ratio matrix benchmark harness (which corpora, how
  measured, where committed) is described in §"Success
  criteria" but not yet wired into a script. We can run the
  Rust runtime's `entropy-bound` and full-compress paths
  manually for the first iteration; if Phase 11 produces
  multiple candidate checkpoints we'll script it.
- No CI integration. Phase 11 is a one-shot research run, not
  a recurring training job.

---

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

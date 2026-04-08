# Phase 0 — Reproduce L3TC with solid engineering foundations

**Goal:** Establish a rigorous, reproducible baseline for the entire project.
We need to know, with hard numbers on identical hardware, how L3TC actually
performs across ratio, single-stream speed, batched speed, and memory —
and how it compares to every classical compressor we'd want to beat.

Every later phase's claim of improvement will be measured against these
numbers. If the benchmark harness isn't right, nothing else matters.

---

## Success criteria

Phase 0 is complete when:

1. We can reproduce L3TC's paper numbers on enwik9 within ±0.5 percentage
   points of reported ratio
2. We have our own measured numbers for L3TC at **batch size 1 CPU single
   stream** — the number the paper glosses over — on a specific known
   machine
3. We have baseline measurements for gzip, bzip2, xz, and zstd at multiple
   preset levels, all on the same corpora, all on the same machine, all
   produced by the same measurement code
4. We have a benchmark harness that produces **reproducible, diffable
   JSON results** for any (compressor, corpus) pair
5. All of this is version-controlled, documented, and someone else could
   run `./scripts/setup.sh && python3 bench/bench.py --all` and get the
   same numbers

---

## What we're NOT doing in Phase 0

- **Not writing any Rust.** That's Phase 1.
- **Not modifying L3TC.** We run it as-is to reproduce its behavior.
- **Not retraining.** We use L3TC's shipped checkpoint.
- **Not designing a file format.** That's Phase 2.
- **Not optimizing L3TC's Python code.** If it's slow, that's the measurement.

The whole point of Phase 0 is to honestly characterize what L3TC is
today. Any optimization we do in Phase 0 pollutes the baseline.

---

## Deliverables

By the end of Phase 0 the repo should contain:

| File | Purpose |
|---|---|
| `bench/bench.py` | Main benchmark harness, standard library only |
| `bench/compressors.py` | Wrappers for each compressor (gzip/bzip2/xz/zstd/l3tc) |
| `bench/results/<date>.json` | Measured numbers for all (compressor, corpus) pairs |
| `bench/results/summary.md` | Human-readable summary table, auto-generated |
| `scripts/setup.sh` | One-command setup: clones L3TC, creates venv, installs deps |
| `scripts/download_corpora.sh` | Downloads enwik8, enwik9, Silesia, Canterbury |
| `vendor/L3TC/` | Cloned L3TC repo (via submodule or clone script) |
| `vendor/RWKV-LM/` | Cloned BlinkDL/RWKV-LM reference |
| `docs/phase_0_findings.md` | Written-up analysis of what the numbers actually say |

---

## Checklist

### Environment and setup

- [x] Directory structure created under `l3tc-prod/`
- [x] Root docs: `README.md`, `ANALYSIS.md`, `DECISIONS.md`, `PHASE_0.md`
- [x] `.gitignore` configured to exclude corpora, venv, results, vendor
- [x] `git init` and first commit
- [ ] `scripts/setup.sh` that clones L3TC + RWKV-LM and sets up Python venv
- [ ] `scripts/download_corpora.sh` that pulls enwik8/enwik9/Silesia/Canterbury

### Benchmark harness

- [x] `bench/compressors.py` with abstract interface + classical wrappers
- [x] `bench/bench.py` with CLI, measurement, JSON output
- [ ] L3TC wrapper in `bench/compressors.py` (depends on setup.sh)
- [ ] End-to-end harness test on a small corpus (enwik6 or local file)
- [ ] Validate round-trip: decompress output exactly equals input for
      every compressor

### Baseline measurements

- [ ] Run classical compressors (gzip -9, bzip2 -9, xz -9e, zstd -22)
      on enwik6 first for quick validation
- [ ] Run classical compressors on enwik8
- [ ] Run classical compressors on enwik9
- [ ] Run classical compressors on Silesia corpus
- [ ] Run classical compressors on a mixed-format corpus (code + logs +
      json + text) to establish real-world baselines

### L3TC measurements

- [ ] Clone L3TC repo, install PyTorch + dependencies
- [ ] Download pretrained checkpoint from their Google Drive link
- [ ] Verify L3TC runs on enwik6 / enwik8 (reproduce paper numbers
      within tolerance)
- [ ] Measure L3TC at **batch size 128** (paper's headline condition)
- [ ] Measure L3TC at **batch size 1 on CPU** (our actual target condition)
- [ ] Measure L3TC at **batch size 1 on GPU** (if GPU available)
- [ ] Record memory usage alongside speed numbers

### Analysis

- [ ] Write `docs/phase_0_findings.md` analyzing the results
- [ ] Compare L3TC's reproduced numbers to the paper's claims
- [ ] Document the single-stream speed gap with hard numbers
- [ ] Identify the actual bottleneck (framework overhead? model compute?
      arithmetic coder? tokenization?) via profiling
- [ ] Decide whether Phase 1 should target ggml/candle or rwkv.cpp based
      on measurements

---

## Practical execution order

1. **Infrastructure first** — docs, harness, baseline measurements on
   classical compressors. This gives us something to compare against even
   before L3TC is running.
2. **Then L3TC setup** — this is the part with the most friction (PyTorch
   install, checkpoint download from Google Drive, reproducing their
   config). Doing it second means we don't block on it.
3. **Then L3TC measurements** — once setup works, the actual runs are
   straightforward.
4. **Then analysis** — write up what we learned, decide Phase 1 direction.

This order matters because the infrastructure is the deliverable that
outlasts any single compressor. Even if L3TC turns out to be broken in
some way, the benchmark harness is useful for anything we build later.

---

## Expected output (what we think we'll see)

Rough predictions before we run anything, so we can check our intuition:

### Compression ratio on enwik9

| Compressor | Expected ratio | Source |
|---|---|---|
| gzip -9 | ~32% | well-known |
| bzip2 -9 | ~25% | well-known |
| zstd -19 | ~23% | well-known |
| zstd -22 | ~22% | well-known |
| xz -9e | ~21% | well-known |
| L3TC-3.2M | ~16% | paper claim |
| NNCP (not running) | ~11% | paper reference |
| cmix (not running) | ~11% | paper reference |

### Decode speed, single stream, Apple M-series CPU

| Compressor | Expected MB/s | Notes |
|---|---|---|
| gzip -d | 200-500 MB/s | stdlib fast path |
| bzip2 -d | 30-60 MB/s | slower decode by design |
| zstd -d | 800-1500 MB/s | zstd is extremely fast |
| xz -d | 70-120 MB/s | decent decode speed |
| L3TC (Python, batch 1) | **0.01-0.05 MB/s** | paper's 11-27 KB/s figure |
| L3TC (Python, batch 128) | **0.5-3 MB/s** | paper's headline |

The **50-100× gap** between L3TC batch-1 and L3TC batch-128, plus the
**10,000× gap** between L3TC batch-1 and zstd, is the thing Phase 1 needs
to close.

If the measurements show something radically different — say L3TC batch-1
is actually only 10× slower than zstd, not 10,000× — then the problem is
different from what we think and the roadmap needs to change.

### Phase 1 decision gates

After Phase 0 finishes, we should be able to answer:

1. **Is framework overhead actually the bottleneck?** (Profile L3TC:
   what percentage of runtime is in Python vs PyTorch vs native CUDA
   kernels vs the arithmetic coder?)
2. **How much does batch size matter?** (Measure at batch 1, 4, 16,
   64, 128.)
3. **Is ratio reproducible?** (If we can't reproduce the paper's ratio
   within 0.5 pp, something is wrong with our setup or their numbers.)
4. **What's the realistic single-stream target?** (Given the model size
   and the classical compressor speeds on the same hardware, what speed
   should a well-engineered L3TC actually achieve?)

Those four answers determine the Phase 1 plan.

---

## Risks and open questions

- **L3TC's checkpoint may not be exactly reproducible.** Google Drive
  downloads are fragile; if the checkpoint disappears or changes, we
  have a problem.
- **PyTorch install on Python 3.14.** Python 3.14 is very new; some
  PyTorch versions may not support it. May need to pin to Python 3.11 or 3.12
  via pyenv or uv.
- **L3TC's training data was enwik8 — we may see different numbers on
  Silesia/Canterbury than expected.** That's fine; it's data for the
  eventual retraining decision.
- **Batch size 1 measurements may vary significantly between runs.**
  Need to do multiple runs and report median + spread, not a single
  number.
- **Round-trip validation is expensive on large corpora.** Decompressing
  all of enwik9 back and diffing vs original takes a while. Do it
  selectively (small corpora always, large corpora once per version).

---

## When Phase 0 is done, Phase 1 starts

Phase 1 is the Rust rewrite of the inference runtime. We don't start it
until Phase 0's numbers are committed and analyzed, because without
baseline numbers we can't tell if Phase 1 actually helps.

Phase 1's first success criterion is: **reproduce the L3TC ratio exactly
(bit-identical round trip) at ≥10× the Phase 0 single-stream speed on
the same machine.** Bit-identical, because arithmetic coding is
unforgiving; 10×, because if we can't even hit 10× by eliminating Python
overhead, we're barking up the wrong tree.

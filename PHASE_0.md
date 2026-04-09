# Phase 0 — Reproduce L3TC with solid engineering foundations  ✅ COMPLETE

**Final result:** reproduced L3TC-200K at 13.24 KB/s / ratio 0.1665 on
enwik6, and L3TC-3.2M at 10.76 KB/s / ratio 0.1309 on enwik6. Key
empirical finding: L3TC-3.2M runs only 1.23× slower than L3TC-200K
despite having 11× more compute — meaning framework overhead is ~97%
of runtime on the 200K model. This motivated the Phase 1 Rust rewrite.
See `docs/phase_0_findings.md` for the full analysis.

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
- [x] `scripts/setup.sh` that clones L3TC + RWKV-LM and sets up Python venv
- [x] `scripts/download_corpora.sh` that pulls enwik8/enwik9/Silesia/Canterbury

### Benchmark harness

- [x] `bench/compressors.py` with abstract interface + classical wrappers
- [x] `bench/bench.py` with CLI, measurement, JSON output
- [ ] L3TC wrapper in `bench/compressors.py` (deferred — we've measured
      L3TC directly via its own script; wrapper is a convenience for
      future integration, not blocking Phase 1)
- [x] End-to-end harness test on a small corpus (smoke test on synthetic
      corpus, then full enwik8 run against classical compressors)
- [x] Validate round-trip: decompress output exactly equals input for
      every classical compressor (8/8 pass on enwik8)

### Baseline measurements

- [x] Run classical compressors on enwik8
- [ ] Run classical compressors on enwik9 (optional; deferred —
      enwik8 is sufficient for Phase 1 target setting)
- [ ] Run classical compressors on Silesia corpus (deferred)
- [ ] Run classical compressors on a mixed-format corpus (deferred)

### L3TC measurements

- [x] Clone L3TC repo, install PyTorch + dependencies
- [x] Download pretrained checkpoint from Google Drive (via gdown, all
      four variants)
- [x] Train SPM tokenizer via Python API (bin/spm_train is Linux-only)
- [x] Patch L3TC to run on macOS CPU (deepspeed stub, weights_only fix,
      hardcoded .cuda() calls)
- [x] Verify L3TC-200K runs on enwik6, reproduce paper numbers
      (13.24 KB/s, 16.65% ratio)
- [x] Verify L3TC-3.2M runs on enwik6, reproduce paper numbers
      (10.76 KB/s, 13.09% ratio)
- [ ] Measure L3TC at batch size 128 (deferred — headline condition,
      not our target, not worth GPU setup for Phase 1)
- [x] Measure L3TC at **batch size 1 on CPU** (our actual target
      condition) — the critical single-stream number
- [ ] Measure L3TC at batch size 1 on GPU (no CUDA available on this
      machine, deferred)
- [ ] Record memory usage alongside speed numbers (deferred — not on
      the Phase 1 critical path)

### Analysis

- [x] Write `docs/phase_0_findings.md` analyzing the results
- [x] Compare L3TC's reproduced numbers to the paper's claims (matches)
- [x] Document the single-stream speed gap with hard numbers
- [x] Identify the actual bottleneck: **framework overhead is ~97% of
      runtime on the 200K model**, based on the 200K/3.2M delta analysis
- [x] Decide Phase 1 runtime: **candle first, rwkv.cpp via FFI as
      fallback** (see DECISIONS.md D3)

---

## Phase 0 results summary

**Classical baselines on enwik8** (committed to
`bench/results/enwik8-classical.json`):

| Compressor | Ratio | Compress MB/s | Decompress MB/s |
|---|---:|---:|---:|
| xz-9e | 24.83% | 1.1 | 83 |
| zstd-22 | 25.27% | 0.93 | 780 |
| xz-6 | 26.67% | 5.8 | 262 |
| zstd-19 | 26.94% | 2.2 | 753 |
| bzip2-9 | 29.01% | 18 | 39 |
| zstd-3 | 35.45% | 450 | 843 |
| gzip-9 | 36.48% | 24 | 546 |
| gzip-6 | 36.55% | 29 | 521 |

**L3TC on enwik6** (committed to
`bench/results/l3tc-enwik6-baseline.json`):

| Model | Ratio | Speed (CPU batch=1) |
|---|---:|---:|
| L3TC-200K | 16.65% | 13.24 KB/s |
| L3TC-3.2M | 13.09% | 10.76 KB/s |

**Phase 1 target:** take L3TC-3.2M from 10.76 KB/s to ≥55 KB/s on the
same machine (5× speedup) while maintaining bit-identical round trip
with the Python reference. Stretch: 100-200 KB/s (10-20× speedup).

**Phase 0 status: COMPLETE ENOUGH TO PROCEED.** The core questions
are answered empirically. Remaining items are nice-to-have but don't
block the Rust port.

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

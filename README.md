# l3tc-prod

**A production-ready learned lossless compressor built on L3TC's ideas.**

Target: a zstd alternative that uses a small RWKV-based language model
with arithmetic coding to achieve better compression ratios than
classical codecs, at single-stream speeds that are practical for general
use.

## Status

**Phase 4 complete; moving to Phase 5/6/7 (tooling + release).**
Default tier: enwik6 actual coded ratio **0.1699** at **~131
KB/s compress** (Phase 4c NEON polish). Opt-in high-ratio tier:
L3TC-3.2M at **0.1337 ratio, 25.95 KB/s** (Phase 4d port, beats
Python L3TC-3.2M's reported entropy bound 0.1309 at 0.1275).
Forward pass bit-identical to Python L3TC; 86% of the 200K
entropy-bound gap closed. Phase 4e (distillation for speed)
was explored and closed as failed: the 1-layer 96-hidden
student experiment cleared the pipeline end-to-end but missed
both targets (0.2871 ratio at 1.12× speedup, not ≤0.195 at
≥2×) — the structural speed ceiling at vocab 16384 × hidden 96
turned out to be layer-independent. See
[`docs/phase_4e_findings.md`](docs/phase_4e_findings.md). The
hybrid classical fallback originally scoped for Phase 4 was
moved to Phase 8. The project's two goals in
[`CLAUDE.md`](CLAUDE.md) remain met on the enwik corpus and the
focus shifts to turning the runtime into a shippable open
source tool.

- ✅ **Phase 0 — Reproduce L3TC with solid engineering foundations.**
  Built a Python benchmark harness, reproduced L3TC-200K and L3TC-3.2M on
  enwik6 (matching paper ratios within noise), measured classical
  compressors on the same corpus, identified framework overhead as the
  dominant cost in L3TC's Python implementation (~97% of runtime on the
  200K model).
- ✅ **Phase 1 — Rust inference runtime.** Rewrote L3TC's forward pass,
  tokenizer wrapper, arithmetic coder, and codec glue in Rust. Hit
  byte-identical round trip on the full 1 MB enwik6 corpus (including
  Persian/Arabic interlanguage links via the raw-fallback path) and
  **beat Python L3TC-200K by 6.43×** on single-stream CPU throughput.
- ✅ **Phase 2 — Ratio tuning and throughput polish.** Default segment
  size 2048 → 4096 for better ratio, serial head matvec (rayon overhead
  was hurting), session pooling experiments, comparative benchmarks vs
  all classical compressors. Final: **89 KB/s compress, 6.9× Python**,
  ratio 0.2060 on enwik6 (best ratio of any compressor in the
  benchmark, 27% better than bzip2-9).
- ✅ **Phase 2.5 — Aggressive speed optimizations (partial).** Shipped
  hand-tuned NEON `matvec_96x96` for the 12 block projections per token
  (2.5a) and INT8 per-column quantization of the 1.5 M-element head
  weight with a widening-AXPY matvec (2.5b). Vectorized cum_freqs
  (2.5c) was attempted and deferred — the simple prefilter killed
  autovectorization and the top-K approach carries too much ratio
  risk for the now-shrunk upside. Final: **116 KB/s compress, 7.4×
  Python**, ratio 0.2061 (essentially unchanged).
- ✅ **Phase 3 — File format, streaming, binary input support.** v3
  format with magic bytes + CRC32 integrity trailer, streaming
  encode, streaming decode for raw-store, `FLAG_RAW_STORE` for
  non-UTF-8 input, full enwik8 baseline (0.2166, 110.65 KB/s).
- ✅ **Phase 4a — Implementation diff vs Python.** Proved our
  forward pass is bit-identical to Python's (max L_inf 3.81e-05),
  matched the paper's entropy bound (we hit 0.1632 at seg 4096 vs
  Python's reported 0.1665). Discovered the paper's reported
  ratio is the theoretical entropy bound, not actual coded bytes.
- ✅ **Phase 4b1-2 — Varint headers + unk extraction.** File format
  v4 with LEB128 segment headers (−1.13 pp), binary-search unk
  extraction replaced raw-fallback subdivision entirely (−2.48 pp).
  enwik6 actual coded ratio **0.1699 at 119 KB/s** — 86% of the
  gap to the entropy bound closed, speed budget intact, 0 raw-
  fallback segments remaining on enwik6.
- ✅ **Phase 4b enwik8 confirmation.** 100 MB round trip at
  ratio **0.1793**, **113.77 KB/s compress**, 113.13 KB/s
  decompress. Phase 4b wins generalize to 100× scale. See
  [`docs/phase_4b_findings.md`](docs/phase_4b_findings.md).
- ✅ **Phase 4c — CPU speed polish.** NEON `exp_f32x4`,
  FFN K/V matvecs on the NEON 96×96 kernel, and NEON
  `quantize_exps_to_freqs`. +7% compress / +8% decompress over
  4b2 → **126.8 / 128.2 KB/s** at unchanged ratio. Added
  `profile` / `audit` / `entropy-bound` debug subcommands.
- ✅ **Phase 4d — L3TC-3.2M port (opt-in tier).** Runtime made
  dimension-agnostic (per-block `intermediate_size`), 3.2M
  checkpoint loads end-to-end. Entropy bound **0.1275** (0.34 pp
  better than Python's 0.1309), actual ratio **0.1337** at
  **25.95 KB/s compress** — 2.4× faster than the Python 3.2M
  reference, 5× slower than our 200K. Ships as an opt-in
  high-ratio tier alongside 200K default. See
  [`docs/phase_4d_findings.md`](docs/phase_4d_findings.md).
- ❌ **Phase 4e — Distillation for compression speed (closed,
  failed).** Built the full pipeline: `dump-teacher` CLI + v2
  format + rayon-parallelized top-K softmax dumps,
  `scripts/distill_l3tc.py` PyTorch training loop with
  pure-PyTorch WKV monkey-patch for MPS/CPU. Ran 4e3: 1-layer
  96-hidden student distilled from the 3.2M teacher on the
  first 5 MB of enwik8, 2 epochs on MPS. Result: **ratio
  0.2871 at 146 KB/s** — missed both decision criteria. The
  structural ceiling: `cum_freqs` and the head matvec both
  scale with vocab (16384), not with layers, so halving the
  layer count only yielded 1.12× speedup, not 2×. The 200K
  stays as default, the 3.2M stays as opt-in. See
  [`docs/phase_4e_findings.md`](docs/phase_4e_findings.md).
- ⏳ **Phase 5 — RWKV-v7 architecture upgrade (still enwik8).**
  Replace RWKV-v4-HiRA with RWKV-v7 at the same 200K parameter
  budget, same training corpus. Apples-to-apples architecture
  comparison: better forward pass, unchanged data.
- ⏳ **Phase 6 — Multi-platform release builds.** macOS-arm64,
  macOS-x86_64, linux-x86_64 release artifacts + GitHub Actions.
  Requires Phase 7.
- ⏳ **Phase 7 — Cross-platform numeric contract.** Deterministic
  integer-only or locked-f32 forward pass so that the same input
  compressed on any target produces byte-identical output.
  Prerequisite for Phase 6 and for a portable on-disk format.
- ⏳ **Phase 8 — Multi-model dispatch + specialist registry.**
  Ship several small specialist models (prose / code / JSON /
  logs / markup) and dispatch per file or per segment. Structural
  fix for the OOD trap.
- ⏳ **Phase 9 — Production hardening.** Decoder fuzzing, unsafe
  NEON audit, panic-free hot path, input-validation caps,
  CI matrix with sanitizers. Turns "clean research runtime" into
  "safe to run on untrusted input".
- ⏳ **Phase 10 — Distribution and language bindings.** C ABI,
  Python wheel, Node module, Homebrew / apt / rpm packages,
  docker image, cargo crate on crates.io. The "how do users
  actually get this" layer.
- ⏳ **Phase 11 — Broader training corpus.** Retrain L3TC-200K
  on The Pile / RedPajama / domain-mix (whatever architecture
  Phase 5 produces), same parameter budget. Does one broader
  model cover enough distributions to be the new default, or
  does the distribution zoo need Phase 8 specialist dispatch?
  Phase 11's result decides whether Phase 8 is worth doing.

Plus [`STORAGE_SERVICE_VISION.md`](STORAGE_SERVICE_VISION.md) —
exploratory writeup of a managed-storage-service productization
path that dodges most of the client-side deployment problems.
Back-burner reference, not an active phase.

See [`PHASE_0.md`](PHASE_0.md) through
[`PHASE_10.md`](PHASE_10.md) for detailed per-phase plans. See
[`CLAUDE.md`](CLAUDE.md) for the two project goals and regression
gates. See [`ANALYSIS.md`](ANALYSIS.md) for the full project
thinking. See [`DECISIONS.md`](DECISIONS.md) for the architectural
decision log (including what we tried and reversed). See
[`docs/`](docs/) for per-phase findings.

## Headline numbers (enwik6, 1 MB, round-trip verified)

| Compressor | Ratio | Compress MB/s | Decompress MB/s |
|---|---:|---:|---:|
| **l3tc-rust 200K (default, Phase 4c)** | **0.1699** | **0.131** | **0.132** |
| **l3tc-rust 3.2M (opt-in, Phase 4d)** | **0.1337** | **0.026** | **0.023** |
| Python L3TC-200K (entropy bound) | 0.1665 | 0.013 | — |
| Python L3TC-3.2M (entropy bound) | 0.1309 | 0.011 | — |
| bzip2-9 | 0.2813 | 16.67 | 35.09 |
| xz-9e | 0.2907 | 3.77 | 52.39 |
| zstd-22 | 0.3001 | 4.34 | 125.17 |
| gzip-9 | 0.3558 | 23.04 | 151.49 |

l3tc-rust has the **best ratio of any compressor in the suite**,
and the gap has widened through Phase 4 — we're now **41% better
than bzip2-9** and **43% better than zstd-22** on real coded
bytes. The Python L3TC number above is the *theoretical entropy
lower bound* (see Phase 4a findings); on the same metric
(entropy bound), l3tc-rust hits **0.1632**, which is 0.33 pp
better than Python L3TC's 0.1665. Our actual coded bytes are
within 0.34 pp of Python's reported theoretical bound — the
closest any real implementation has come in this suite.

Throughput is the remaining asymmetric tradeoff: the closest
classical compressor on speed (xz-9e) is ~32× faster.
Phase 7 (deterministic numerics) + Phase 6 (multi-platform
release builds) are the remaining distribution blockers; Phases
8/9/10 are the productionization path.

### Other corpora

- **enwik8 (100 MB, Phase 4b2 confirmation run):** ratio
  **0.1793**, compress **113.77 KB/s**, decompress **113.13
  KB/s**, round-trip byte-identical. Ratio dropped 3.73 pp vs
  Phase 3's 0.2166, tracking the 3.61 pp enwik6 drop and
  confirming the Phase 4b wins generalize to 100× scale.
- **Silesia (heterogeneous):** mixed picture. Text files
  (dickens / webster / nci) vary from "wins like enwik6" to
  "loses to raw bytes" depending on distribution. Binary files
  use the `FLAG_RAW_STORE` path and compress 1:1 plus 28 bytes
  of framing. This is the OOD failure mode that Phase 8
  (multi-model dispatch with classical fallback) addresses
  structurally. See
  [`docs/phase_3_findings.md`](docs/phase_3_findings.md) for
  the detailed numbers.

### How we got here

| phase | enwik6 ratio | compress KB/s | note |
|---|---:|---:|---|
| Phase 1 end | 0.2094 | 85.16 | first Rust round-trip |
| Phase 2 end | 0.2060 | 89.80 | ratio tuning + segment 4096 |
| Phase 2.5 end | 0.2060 | 116 | NEON blocks + INT8 head |
| Phase 4a end | 0.2060 | 117 | forward pass parity with Python |
| Phase 4b1 | 0.1947 | 114 | file format v4 (varint segments) |
| Phase 4b2 | 0.1699 | 119 | unk extraction + parallel tokenize |
| **Phase 4c** | **0.1699** | **126.8** | **NEON exp + FFN KV matvec + NEON quantize** |
| Phase 4d (opt-in 3.2M) | 0.1337 | 25.95 | L3TC-3.2M port, dimension-agnostic runtime |

## Why this project exists

L3TC (AAAI 2025) is the first learned lossless compressor that looks
like it could plausibly compete with classical codecs in production.
The paper reports 16% compression ratio on enwik9 (vs 32% for gzip,
~21% for zstd) with "megabytes per second" decoding speeds.

But the paper's speed numbers come from batched inference at batch
size 128+ on GPU. **At batch size 1 on CPU — which is how real
compression workloads actually run — L3TC drops to 11-27 KB/s**,
slower than gzip by roughly 5000×. That's the gap this project
closes.

The research is done; the engineering isn't. Closing the gap is
mostly systems work: replacing Python/PyTorch inference with a
custom runtime, designing a real file format, shipping as a single
static binary, handling adversarial inputs gracefully. None of it
is novel research. All of it is necessary for anything that claims
to be a compressor.

## Layout

```
l3tc-prod/
├── README.md              this file
├── CLAUDE.md              the two project goals + regression gates
├── ANALYSIS.md            original project thinking document
├── DECISIONS.md           architectural decision log
├── PHASE_0.md .. PHASE_11.md
│                          per-phase plans (0-4d done; 4e in
│                          progress; 5-11 roadmap past Phase 4)
├── PHASE_4C.md, PHASE_4D.md, PHASE_4E.md
│                          Phase 4 speed polish / 3.2M port /
│                          distillation plans
├── STORAGE_SERVICE_VISION.md
│                          exploratory back-burner productization writeup
├── docs/
│   ├── phase_0_findings.md
│   ├── phase_2_findings.md
│   ├── phase_3_findings.md
│   ├── phase_4a_findings.md
│   ├── phase_4b_findings.md
│   ├── phase_4d_findings.md
│   └── phase_4e_findings.md
├── bench/                 Python benchmark harness (stdlib-only)
│   ├── bench.py
│   ├── compressors.py
│   ├── corpora/           test corpora (gitignored, see download_corpora.sh)
│   └── results/           committed JSON benchmark results
├── l3tc-rust/             production Rust crate
│   ├── Cargo.toml
│   ├── README.md
│   ├── src/
│   │   ├── lib.rs         module exports + doc
│   │   ├── error.rs       thiserror-based typed errors
│   │   ├── bitio.rs       bit-level I/O for the arithmetic coder
│   │   ├── arithmetic.rs  Nayuki-style arithmetic coder
│   │   ├── tensor.rs      hand-rolled f32 linalg (matvec, LN, sigmoid, ...)
│   │   ├── checkpoint.rs  reader for the Rust-friendly binary format
│   │   ├── rwkv.rs        RWKV-v4 + HiRA forward pass
│   │   ├── tokenizer.rs   SentencePiece wrapper + raw-fallback refinement
│   │   ├── codec.rs       compress/decompress + segment parallelism
│   │   └── bin/l3tc.rs    CLI binary
│   ├── scripts/
│   │   └── convert_checkpoint.py   .pth → .bin, runs in L3TC's venv
│   ├── tests/             integration tests vs real checkpoint + SPM model
│   └── iter.sh            fast iteration loop: build + round-trip + verify
├── vendor/                (gitignored) cloned L3TC reference + RWKV-LM
└── scripts/
    ├── setup.sh           clone L3TC + RWKV-LM, set up Python venv
    ├── download_corpora.sh  download enwik8, Canterbury, Silesia
    ├── distill_l3tc.py    Phase 4e PyTorch distillation training loop
    ├── dump_python_logits.py  Phase 4a diff harness
    └── diff_logits.py     Rust vs Python logit comparison
```

## Quick start

### Compressor baselines (no Rust toolchain required)

```bash
# Download test corpora
./scripts/download_corpora.sh                # enwik8 + Canterbury
./scripts/download_corpora.sh --all          # also enwik9 + Silesia

# Run classical compressors on enwik8
python3 bench/bench.py --classical-only --corpus bench/corpora/enwik8

# Compare l3tc-rust against classical on enwik6
python3 bench/bench.py --corpus bench/corpora/enwik6 \
    --compressor l3tc-rust-200k \
    --compressor gzip-9 --compressor bzip2-9 \
    --compressor xz-9e --compressor zstd-22
```

### Rust crate (production path)

```bash
cd l3tc-rust
cargo build --release
cargo test --release                          # 34 unit tests

# End-to-end round trip on 50 KB of Wikipedia
./iter.sh

# Or explicit CLI
./target/release/l3tc compress /path/to/input.txt -o out.l3tc --verify --time
./target/release/l3tc decompress out.l3tc -o back.txt --time

# Debug subcommands
./target/release/l3tc entropy-bound --input enwik6 --segment-bytes 4096
./target/release/l3tc audit --input enwik6         # per-source byte breakdown
./target/release/l3tc profile --input enwik6       # per-phase timing breakdown
./target/release/l3tc dump-logits --input enwik6 -o logits.bin
./target/release/l3tc dump-teacher --input enwik6 -o teacher.bin --top-k 64
```

### Integration tests against the real checkpoint

Requires `./scripts/setup.sh` to clone L3TC and `python
l3tc-rust/scripts/convert_checkpoint.py --input
vendor/L3TC/checkpoints/l3tc_checkpoints/l3tc_200k_bpe16k_c999_checkpoint0019.pth
--config vendor/L3TC/config/l3tc/l3tc_200k.py --output
l3tc-rust/checkpoints/l3tc_200k.bin` to produce the Rust-friendly
checkpoint. Then:

```bash
cd l3tc-rust
cargo test --release --test end_to_end -- --ignored --test-threads=1 --nocapture
```

## Decisions (key ones)

- **D1 — Target:** general-purpose lossless compressor, zstd alternative
- **D2 — Language:** Rust for the production implementation
- **D3 — Inference runtime:** originally "candle first, rwkv.cpp as
  fallback", but reversed in Phase 1 — we hand-rolled the tensor math
  because the model is small (2 layers × 96 dim) and a framework's
  abstractions cost more than they save. See D3 and D8 in
  [`DECISIONS.md`](DECISIONS.md).
- **D4 — Training data:** stick with enwik8 through Phase 5 (v7
  architecture upgrade); broader corpus (The Pile / RedPajama /
  domain mix) is its own Phase 11 after the architecture test
- **D5 — RWKV version:** match L3TC (v4 + HiRA) for Phase 0–4,
  upgrade to RWKV-7 in Phase 5 as an apples-to-apples
  architecture comparison on the same enwik8 corpus
- **D7 — Benchmark harness:** stdlib-only Python (no numpy, no pandas),
  shells out to compressors and measures wall time + rusage + byte counts

See [`DECISIONS.md`](DECISIONS.md) for the full list with reasoning and
(where applicable) what we changed our mind about.

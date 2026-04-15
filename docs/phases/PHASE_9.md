# Phase 9 — Production hardening (fuzz, security, panic budget)

**Status as of 2026-04-15:** in progress. Fuzz infrastructure
set up, first round of bugs found and fixed. Long-duration
fuzz runs (48h each) pending.

## The problem

The decoder assumes well-formed input. Our tests cover "inputs our own encoder produced" but not "arbitrary bytes an attacker crafted". A compressor is a natural attack surface: format parsing, length-field allocations, tight numeric loops, unsafe intrinsics.

## Deliverables

1. **Decoder fuzzing.** `cargo-fuzz` harness targeting `decompress_bytes` + each parser in isolation (`read_header`, `read_varint`, `ArithmeticDecoder`, `decode_segment`, `decode_writer`). 24-48 hours per harness, zero crashes, zero ASan/UBSan findings.

2. **Unsafe NEON audit.** Line-by-line review of `matvec_96x96_neon` in `src/tensor.rs`. Pre/post conditions documented, bounds verified, debug assertions added.

3. **Panic-free hot path.** Replace `.unwrap()`, `.expect()`, unchecked indexing in the decoder with explicit bounds checks returning typed errors.

4. **Input validation budget.** Explicit caps on segment count (10M), segment body length (16 MB), unk count (64K), unk payload (64 KB), total allocated memory (N x file size). Reject before allocating.

5. **Integer overflow review.** `checked_*` ops on all untrusted-input paths (AC arithmetic, cum_freqs accumulator, varint decoder).

6. **CI hardening.** Miri on unsafe code, ASan+UBSan on linux-x86_64, `clippy::unwrap_used` and `clippy::panic` as errors in the library crate, weekly fuzz runs.

7. **Release signing + SBOM.** minisign/cosign signatures, `cargo sbom` for every release.

## Progress

### Completed (2026-04-15)

**Fuzz infrastructure:**
- `cargo-fuzz` with 3 harnesses in `l3tc-rust/fuzz/`:
  - `fuzz_checkpoint_parse` — binary checkpoint parser
  - `fuzz_arithmetic_decoder` — AC decoder with random freq tables
  - `fuzz_decompress_header` — checkpoint parser via from_bytes
- All three clean after 60s smoke tests (9.3M + 1.5M + 9.5M runs)
- Requires nightly: `cargo +nightly fuzz run <target>`

**Bugs found and fixed by fuzzing:**
1. **Checkpoint OOM** — `HashMap::with_capacity(n_tensors)` where
   n_tensors was read from malformed input without bounds check.
   Fix: cap n_tensors against remaining input bytes.
2. **Checkpoint integer overflow** — `shape.iter().product()`
   panicked on crafted tensor dimensions that overflow usize.
   Fix: `checked_mul` with explicit error return.

**Safety hardening applied:**
- Varint decoder: `checked_shl` prevents shift overflow
  (`codec.rs:read_varint`)
- cum_freqs accumulator: `saturating_add` + runtime fallback
  replaces `debug_assert` (`codec.rs:logits_to_cum_freqs_scratch`)
- NEON unsafe: `assert!` replaces `debug_assert!` for shape
  checks — fires in release builds (`tensor.rs:matvec_96x96_neon`)
- Checkpoint: ndim bounded against remaining bytes

### Remaining

- Long-duration fuzz runs (48h per harness) — schedule overnight
- Input validation budgets (segment count, body length, unk caps)
- Panic-free hot path audit (grep for unwrap/expect/indexing)
- CI integration (Miri on unsafe, ASan/UBSan on Linux)
- Release signing + SBOM (low priority, do at Phase 6)

## Success criteria

- 48+ hours of `cargo-fuzz` per harness with zero crashes and zero sanitizer findings
- No `.unwrap()`/`.expect()`/unchecked indexing in the library hot path
- Every input-derived length field has an explicit cap
- Miri clean on non-aarch64 scalar path
- Release artifacts have SHA256 + signatures + SBOM

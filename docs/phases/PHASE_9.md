# Phase 9 — Production hardening (fuzz, security, panic budget)

**Status:** back-burner. Turns "clean research runtime" into "safe to run on untrusted input from strangers". Not urgent while single-user on local machines; becomes critical for file-sharing workflows, backup tools, or a storage service accepting uploads.

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

## Success criteria

- 48+ hours of `cargo-fuzz` per harness with zero crashes and zero sanitizer findings
- No `.unwrap()`/`.expect()`/unchecked indexing in the library hot path
- Every input-derived length field has an explicit cap
- Miri clean on non-aarch64 scalar path
- Release artifacts have SHA256 + signatures + SBOM

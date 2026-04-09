# Phase 9 — Production hardening (fuzz, security, panic budget)

**Status:** back-burner. The work that turns "clean research
runtime" into "safe to run on untrusted input from strangers".

**The problem.** The current decoder assumes its input is
well-formed. Our 34 unit tests pass and the round trip is
byte-identical on every corpus we've tested, but "inputs our
own encoder produced" is a much weaker guarantee than "arbitrary
bytes an attacker crafted". A compressor is a natural attack
surface — file formats parsed on arbitrary input, length fields
used for buffer allocation, tight numeric loops, unsafe
intrinsics. Every one of those is a place a bad file can crash,
hang, or own the decoder.

Not an immediate concern because we're single-user on local
machines. Becomes an immediate concern the moment l3tc-rust is
part of a file-sharing workflow, a backup tool run on
user-provided data, or a storage service accepting uploads.

## Concrete deliverables

1. **Decoder fuzzing.** `cargo-fuzz` harness targeting the full
   `decompress_bytes` path plus each parser in isolation:
   - `read_header` / `read_segment_meta` / `read_segment_meta_v3`
   - `read_varint`
   - `ArithmeticDecoder::new` + `decode_symbol`
   - `decode_segment` (the tokenizer decode side)
   - `decode_writer` (streaming raw-store path with adversarial
     trailer positions)

   Target: 24-48 hours of fuzzing per harness, zero crashes,
   zero ASan/UBSan findings. Corpus: seed with real compressed
   files, plus hand-crafted edge cases (zero-length segments,
   maximum segment counts, truncated CRCs).

2. **Unsafe NEON audit.** Line-by-line review of the
   `matvec_96x96_neon` function in `src/tensor.rs` with explicit
   pre/post conditions documented as comments. Verify bounds on
   all pointer arithmetic, confirm alignment assumptions, add
   debug assertions in the `#[cfg(debug_assertions)]` path.
   Consider whether Miri can validate the intrinsics under
   `cfg(not(target_arch = "aarch64"))` via a scalar emulation.

3. **Panic-free hot path.** Audit the decoder for `.unwrap()`,
   `.expect()`, and panicking indexing (`[i]` on a slice with
   untrusted `i`). Replace with explicit bounds checks that
   return `Error::BadCheckpoint`. The hot path should never
   panic on adversarial input.

4. **Input validation budget.** Declare explicit maximum values
   for:
   - segment count (current u32::MAX sentinel means implicit;
     add a sanity cap like 10M segments per file)
   - segment body length (currently unchecked; cap at maybe
     16 MB to match the format's realistic use)
   - unk count per segment (cap at 64K)
   - individual unk payload (cap at 64 KB)
   - raw-fallback payload (cap at segment body limit)
   - total allocated memory per decompress call (cap at N× the
     file size to detect decompression-bomb attacks)

   Reject anything that exceeds these with `Error::BadCheckpoint`
   before allocating. Decompression bombs are a class of attack
   where a tiny compressed file expands to terabytes; our format
   has multiple length fields that could be abused the same way.

5. **Integer overflow review.** Especially on the arithmetic
   coder's `(low, high)` arithmetic, the cum_freqs accumulator,
   and the varint decoder. Prefer `checked_*` ops on any
   untrusted-input path; `wrapping_*` only where the wrap is
   intentional and documented.

6. **CI matrix hardening.** Add to the existing CI:
   - Miri on unsafe code in `tensor.rs`
   - `cargo test --release --target x86_64-unknown-linux-gnu`
     under ASan + UBSan (requires nightly)
   - Clippy warnings as errors, especially `clippy::unwrap_used`
     and `clippy::panic` in the library crate (allow them in
     tests and the CLI)
   - Weekly fuzz run on CI with the seed corpus

7. **Release signing + SBOM.** Sign release binaries with
   minisign or cosign. Generate a software bill of materials
   (`cargo sbom`) for every release. Enables downstream consumers
   to verify provenance.

## Success criteria

- 48+ hours of `cargo-fuzz` on each harness with zero crashes
  and zero sanitizer findings
- No `.unwrap()` / `.expect()` / unchecked indexing in the
  library hot path (tests and CLI exempt)
- Every input-derived length field has an explicit cap with a
  clear error on overflow
- Miri clean on the non-aarch64 scalar path
- Release artifacts have SHA256 + minisign signatures + SBOM

## Non-goals

- Formal verification (out of scope; we're aiming for "no
  trivial crashes", not "provably correct")
- Side-channel resistance (this is a compressor, not a crypto
  library; constant-time ops are not a goal)
- Windows / macOS sanitizer coverage beyond what's easy — Linux
  is the primary fuzzing target because the toolchain is
  mature

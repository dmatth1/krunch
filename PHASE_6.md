# Phase 6 — Multi-platform release builds and distribution

**Goal:** ship pre-built binaries for the three platforms users
actually run (macOS-arm64, macOS-x86_64, linux-x86_64) without
requiring them to clone the repo and install a Rust toolchain.

This was originally scoped as 3d in `PHASE_3.md` but was deferred
because Phase 3's main work (file format hardening, streaming,
binary input support) is more user-facing and Phase 4's ratio
work is higher value per unit time.

**Hard prerequisite: Phase 7** (cross-platform numeric contract).
Multi-platform release builds are pointless if a file compressed
on macOS-arm64 doesn't decompress identically on x86_64-linux.
Our current forward pass uses libm `f32::exp` and has no
guarantees about cross-target bit-identity. Phase 7 is the
numerics work that makes the on-disk format actually portable.
Do not ship 6 until 7 lands.

---

## Tasks

1. **Add a `release-portable` cargo profile** that disables
   `target-cpu=native` so the binary runs on any CPU of the
   target architecture, not just the build machine. Keep the
   existing `release` profile (with `target-cpu=native`) as the
   "fastest possible local build" option for development and
   benchmarking.
2. **Cross-compile the three targets:**
   - `aarch64-apple-darwin`
   - `x86_64-apple-darwin`
   - `x86_64-unknown-linux-gnu`
   The `aarch64` build needs to keep the NEON intrinsics intact
   (they're already gated on `target_arch = "aarch64"`).
3. **Measure each artifact** on a representative machine and
   record the throughput delta vs the `target-cpu=native` build.
   Acceptable delta: ≤10% on aarch64-apple-darwin (NEON should
   still fire), ≤25% on x86_64 targets where the autovectorizer
   has less specific knowledge.
4. **GitHub Actions workflow** that:
   - Triggers on tag push (`v*.*.*`)
   - Builds all three targets
   - Runs `cargo test --release --lib` on each
   - Attaches the binaries to the GitHub release
   - Generates a SHA256 manifest
5. **Documentation:**
   - Update README with installation instructions for each
     platform
   - Add a "verify the binary" section pointing at the SHA256
     manifest
6. **SentencePiece dependency note:** the `sentencepiece` crate
   wraps the C++ library, which must be present at build time.
   The release workflow needs to install it on each runner
   (`brew install sentencepiece` on macOS,
   `apt install libsentencepiece-dev` on Debian/Ubuntu).
   Document this in the README so users building from source on
   other systems know what to install.

---

## Success criteria (Phase 6 exit)

- Three release binaries build cleanly from a fresh checkout on
  the matching CI runner
- Each binary's throughput on enwik6 is within the per-platform
  budget noted above
- A tagged release on GitHub has all three artifacts attached
  with a verifiable SHA256 manifest
- README has working installation instructions for each platform

## Non-goals

- Cargo crate publication (different release channel, separate
  consideration)
- Homebrew / apt package metadata (community-driven, not
  upstream's job)
- Windows support (sentencepiece-rs wrapper is shaky on Windows;
  defer until there is real demand)
- Static linking against musl (out of scope; the `gnu` target
  is fine for the artifact users we care about)

# Phase 6 — Multi-platform release builds and distribution

**Status:** not started. Depends on Phase 7 (cross-platform numeric contract).

**Goal:** ship pre-built binaries for macOS-arm64, macOS-x86_64, and linux-x86_64 without requiring users to clone the repo and install a Rust toolchain. Originally scoped as Phase 3d, deferred because ratio work was higher value.

## Hard prerequisite: Phase 7

Multi-platform release builds are pointless if a file compressed on macOS-arm64 doesn't decompress identically on x86_64-linux. The current forward pass uses libm `f32::exp` with no cross-target bit-identity guarantees. Phase 7 solves this. Do not ship Phase 6 until Phase 7 lands.

## Requirements

1. **`release-portable` cargo profile** disabling `target-cpu=native`. Keep `release` as the fast-local-build option.
2. **Three cross-compiled targets:** `aarch64-apple-darwin`, `x86_64-apple-darwin`, `x86_64-unknown-linux-gnu`. NEON intrinsics must stay active on aarch64.
3. **Throughput budget:** aarch64-apple-darwin <= 10% slower than native, x86_64 targets <= 25% slower.
4. **GitHub Actions workflow:** triggers on `v*.*.*` tag push, builds all three targets, runs `cargo test --release --lib`, attaches binaries + SHA256 manifest to the release.
5. **SentencePiece note:** the `sentencepiece` crate wraps C++; CI runners need `brew install sentencepiece` (macOS) or `apt install libsentencepiece-dev` (Debian/Ubuntu).
6. **README** updated with installation instructions per platform.

For universal text compression, portability is table stakes -- users need to compress on one platform and decompress on another.

## Success criteria

- Three release binaries build cleanly from a fresh checkout
- Each binary's throughput within the per-platform budget
- Tagged GitHub release with all artifacts + SHA256 manifest
- README has working install instructions

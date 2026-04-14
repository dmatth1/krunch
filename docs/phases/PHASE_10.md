# Phase 10 — Distribution and language bindings

**Status:** back-burner. Depends on Phases 6-9 (portable builds, deterministic numerics, specialist dispatch, hardening). This is the distribution layer on top.

**Goal:** turn "clone our repo and cargo build" into "pip install l3tc" or "brew install l3tc". For universal text compression to be useful, it has to be accessible from any language and any package manager people already use.

## Deliverables (priority order)

1. **C ABI layer** (`src/c_api.rs`). `cbindgen`-generated header. ~10-15 `extern "C"` functions: open compressor with model path, compress/decompress bytes, error codes, close. Foundation for all other bindings.

2. **Python wheel** via PyO3. Pre-built manylinux + macOS wheels on PyPI. This is how most users will discover the tool.

3. **Cargo crate** on crates.io. Split into `l3tc-core` (library) and `l3tc-cli` (binary).

4. **Homebrew formula** for macOS. Ships pre-built binary from Phase 6 release artifacts.

5. **Debian/Ubuntu packages** (`.deb`) via `cargo-deb`.

6. **Node.js module** via napi-rs (if demand exists).

7. **Docker image** on `ghcr.io/dmatth1/l3tc`. Tagged per release.

8. **Documentation site** (mdBook) with quickstart, API reference, benchmarks, file format spec. GitHub Pages.

## Why this comes last

Every item consumes what Phases 6-9 produce: portable binaries (6), numeric stability for a stable format claim (7), specialist models to ship alongside (8), safety guarantees for arbitrary input (9). Shipping distribution before those is how you get "l3tc crashed on a normal file" bug reports.

## Success criteria

- `pip install l3tc` works on macOS-arm64, macOS-x86_64, linux-x86_64-manylinux
- `brew install l3tc` works
- `cargo install l3tc` works via crates.io
- Every release tag triggers CI that builds binaries, wheels, and container images
- Documentation site has a quickstart that works without cloning the repo

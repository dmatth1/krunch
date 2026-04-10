# Phase 10 — Distribution and language bindings

**Status:** back-burner. The work that turns "you can clone our
repo and cargo build it" into "you can `pip install l3tc` or
`brew install l3tc` and use it in any language you already code
in". Nothing shipped here matters until Phases 6-9 are done
(multi-platform builds, deterministic numerics, hardening,
dispatch) — this is the distribution layer on top.

**The problem.** Right now l3tc-rust is a Rust binary and a Rust
library crate. Using it from anywhere else means cloning the
repo, installing the Rust toolchain, and building from source.
That's fine for research collaborators and terrible for anyone
who just wants to compress a file.

## Concrete deliverables

1. **C ABI layer** (`src/c_api.rs` in the library crate, or a
   separate `l3tc-c` crate). `cbindgen`-generated header. API
   surface: open a compressor with a model path, compress bytes
   to bytes, decompress bytes to bytes, error codes, close.
   Opaque handles for the model and session. Keep it small —
   maybe 10-15 functions, all `#[no_mangle]` `extern "C"`.

   This is the foundation for every language binding. Do it
   first, make it stable, and build everything else on top.

2. **Python wheel** via PyO3 (or raw cffi on the C ABI). API:

   ```python
   import l3tc
   compressed = l3tc.compress(b"hello world")
   original = l3tc.decompress(compressed)
   ```

   Pre-built manylinux and macOS wheels published to PyPI. This
   is how most users will actually discover and try the tool.

3. **Node.js module** via napi-rs. Same API shape as Python.
   Publish to npm. Supports JavaScript server-side workflows,
   electron apps, etc.

4. **Go wrapper** via cgo consuming the C ABI. Only if there's
   demonstrated demand; cgo is painful and the Go ecosystem has
   its own preferences.

5. **Homebrew formula** for macOS. `brew install l3tc`. Ships
   the pre-built binary from the Phase 6 release artifacts.

6. **Debian / Ubuntu packages** (`.deb`) via `cargo-deb` or a
   hand-written debian/ directory. `apt install l3tc`.

7. **RPM** for Fedora / RHEL / Rocky. Similar shape.

8. **Nix flake / derivation**. `nix run github:you/ltec` works
   out of the box.

9. **Docker image** published to `ghcr.io/dmatth1/l3tc`. Tag
   per release; `latest` tracks the most recent stable.

10. **Cargo crate** published to crates.io. The library is
    already structured correctly; just need a `cargo publish`
    with a stable API contract. Splits into `l3tc-core` (the
    library) and `l3tc-cli` (the binary) at publication time.

11. **Documentation site** (mdBook) with quickstart, API
    reference, benchmarks, and the file format spec. Hosted on
    GitHub Pages from the repo.

## Why this comes last

Every item in this list consumes the thing Phases 6-9 produce:

- Phase 6 gives us portable release binaries to package.
- Phase 7 gives us the numeric stability to claim the format is
  stable enough to publish.
- Phase 8 gives us the specialist models that ship alongside
  the binary.
- Phase 9 gives us the safety guarantees that make "run this on
  arbitrary input" a defensible claim.

Shipping distribution before those ship is how you get "I
installed l3tc from apt and it crashed on a normal file" bug
reports. Worse than not shipping at all.

## Success criteria

- `pip install l3tc` works on macOS-arm64, macOS-x86_64,
  linux-x86_64-manylinux
- `brew install l3tc` works
- `cargo install l3tc` works via crates.io
- `npm install l3tc` works (if Node binding ships)
- Every release tag triggers a GitHub Actions workflow that
  builds binaries, wheels, and container images and publishes
  them
- The documentation site has a quickstart that a new user can
  follow without cloning the repo

## Non-goals

- Windows support (defer unless there's demand)
- iOS / Android bindings (the use case for an on-device text
  compressor is narrow; defer)
- WASM build (interesting for browser demos but not a real
  workflow)
- A GUI (the whole point of a compressor is that it slots into
  a file pipeline, not that it has a UI)

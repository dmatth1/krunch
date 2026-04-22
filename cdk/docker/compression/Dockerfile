# Compression worker image.
#
# Stage 1: Rust builder. Pulls the l3tc source from the repo root
# (CodeBuild context is `.`), installs the bzip3 / sentencepiece C
# libraries the hybrid dispatcher depends on, and produces a release
# binary at /l3tc.
#
# Stage 2: Python runtime. Thin slim-python with boto3 + zstd CLI
# (legacy codec=zstd_fallback path) and the l3tc binary copied from
# stage 1. Runtime libs (libsentencepiece0, libbz3) are installed so
# the binary loads; the heavier build toolchain from stage 1 is
# discarded, keeping the final image under ~250 MB.
#
# When `codec=hybrid` is set in a dataset's metadata, the worker
# shells out to `/usr/local/bin/l3tc hybrid-compress`; when missing,
# the worker falls back to raw zstd with a WARN log line so older
# deployments of this image still function.

# -----------------------------------------------------------------------------
# Stage 1: Rust builder
# -----------------------------------------------------------------------------
# Pull the Rust image via AWS's Docker Hub mirror on public.ecr.aws to
# dodge the anonymous Docker Hub rate limit (429 Too Many Requests)
# that killed the compression-image build on 2026-04-22. Same upstream
# image, mirrored by AWS — zero functional difference.
# 1-bookworm floats to the latest Rust 1.x stable. Pinning to 1.82
# broke on 2026-04-22 because cargo's dep resolver pulled in
# clap_builder 4.6.0 (edition 2024 in its Cargo.toml) which 1.82
# can't parse. Latest-stable avoids re-breaking every time an indirect
# dep bumps its MSRV.
FROM public.ecr.aws/docker/library/rust:1-bookworm AS rust-builder

# bzip3 headers aren't in Debian Bookworm yet; build from upstream.
# Pin to a specific tag so CodeBuild reruns are reproducible.
# Match the version libbzip3-sys 0.5.1+1.5.3 bundles — the crate's
# build.rs links against whatever headers we install here.
ARG BZIP3_VERSION=1.5.3
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git pkg-config \
        libsentencepiece-dev \
        libclang-dev \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${BZIP3_VERSION} \
        https://github.com/kspalaiologos/bzip3.git /tmp/bzip3 \
 && cd /tmp/bzip3 \
 && cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Release . \
 && make -j"$(nproc)" install \
 && rm -rf /tmp/bzip3

ENV BZIP3_INCLUDE_DIR=/usr/include \
    BZIP3_LIB_DIR=/usr/lib

WORKDIR /src
COPY l3tc-rust /src/l3tc-rust
# Build the release binary. `--no-default-features` drops the
# install-models machinery (ureq/tar/sha2) which the server-side
# worker doesn't need, shaving ~3 MB off the binary.
RUN cd l3tc-rust \
 && cargo build --release --bin l3tc \
 && strip target/release/l3tc \
 && cp target/release/l3tc /l3tc

# -----------------------------------------------------------------------------
# Stage 2: Python runtime
# -----------------------------------------------------------------------------
FROM public.ecr.aws/docker/library/python:3.11-slim

# Flush stdout on every line so CloudWatch sees progress live. Without
# this, Python buffers stdout when the FD is a pipe.
ENV PYTHONUNBUFFERED=1

# Runtime dependencies:
#   - zstd: legacy codec=zstd_fallback path (pre-hybrid metadata).
#   - ca-certificates: S3 + SQS TLS.
#   - libsentencepiece0: dynamically loaded by the l3tc binary's
#     SentencePiece wrapper.
#   - libstdc++6 / libgcc-s1: C++ runtime for libsentencepiece + bzip3.
# bzip3 is built from source in stage 1; we copy the shared library
# into /usr/lib so it's available at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        zstd ca-certificates libsentencepiece0 libstdc++6 libgcc-s1 \
 && rm -rf /var/lib/apt/lists/*

# The l3tc binary + bzip3 shared library both come from stage 1.
# bzip3 1.5.3 installs `libbzip3.so.*` (earlier versions used
# `libbz3.so.*`); glob both so we survive future renames. ldconfig
# rebuilds the linker cache so the newly-placed libs are visible
# without having to drop LD_LIBRARY_PATH on the PID.
COPY --from=rust-builder /l3tc /usr/local/bin/l3tc
COPY --from=rust-builder /usr/lib/libbz* /usr/lib/
RUN ldconfig

WORKDIR /app

RUN pip install --no-cache-dir boto3==1.35.0

COPY cdk/docker/compression/compression_worker.py /app/compression_worker.py

ENTRYPOINT ["python", "/app/compression_worker.py"]

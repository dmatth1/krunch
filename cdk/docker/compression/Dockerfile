# Compression worker container for ECS Fargate.
#
# Wraps the l3tc-rust binary + a small Python wrapper that reads
# SQS messages and orchestrates download → compress → upload → delete raw
# → update DDB.

FROM rust:1-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake pkg-config libsentencepiece-dev build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy the Rust crate source (l3tc-rust, same as in the archived project).
COPY l3tc-rust/Cargo.toml l3tc-rust/Cargo.lock /build/
COPY l3tc-rust/src/ /build/src/
# `l3tc-rust/benches/` and `tests/` not needed for the binary.

RUN cargo build --release --bin l3tc

# ---- runtime image ----
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsentencepiece-dev ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir boto3==1.35.0

COPY --from=builder /build/target/release/l3tc /usr/local/bin/l3tc

COPY cdk/docker/compression/compression_worker.py /app/compression_worker.py

ENTRYPOINT ["python", "/app/compression_worker.py"]

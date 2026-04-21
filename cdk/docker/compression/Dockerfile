# Spike 1 compression worker — zstd-only. No Rust build needed.
# In a later spike we'll multi-stage-build the l3tc Rust binary in here
# to serve codec=l3tc datasets.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    zstd ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir boto3==1.35.0

COPY cdk/docker/compression/compression_worker.py /app/compression_worker.py

ENTRYPOINT ["python", "/app/compression_worker.py"]

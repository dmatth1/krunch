# Training container for AWS Batch.
#
# Built from the repo root (docker build -f cdk/docker/training/Dockerfile .)
# so that we can COPY in the scripts/ and vendor/ directories.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates cmake pkg-config \
    libsentencepiece-dev sentencepiece \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps. We pin a small set rather than copying the full
# vendor/L3TC/requirements.txt because many of those are PyTorch CUDA
# that wants a specific wheel index.
COPY cdk/docker/training/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy training pipeline + reference model code.
COPY scripts/ /app/scripts/
COPY vendor/L3TC/ /app/vendor/L3TC/

# Copy the entrypoint that wraps scripts/train_l3tc_phase11.py for
# the Batch-style "read from S3, train, write to S3" flow.
COPY cdk/docker/training/train_entrypoint.sh /app/train_entrypoint.sh
RUN chmod +x /app/train_entrypoint.sh

ENTRYPOINT ["/app/train_entrypoint.sh"]

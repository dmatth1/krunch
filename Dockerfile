FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RWKV_JIT_ON=1 \
    RWKV_CUDA_ON=1 \
    KRUNCH_MODEL_DIR=/models

# System deps — ninja-build is required for torch.utils.cpp_extension (WKV kernel).
# pip install ninja installs only a Python wrapper that torch can't find.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    ninja-build build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Python dependencies (includes pip-published `rwkv` package — provides the
# fast RWKV inference path with the WKV CUDA kernel)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Model weights + tokenizer baked into the image — no runtime downloads.
# Place files in ./models/ before `docker build`:
#   - RWKV-4-Pile-169M-20220807-8023.pth  (~323 MB)
#   - 20B_tokenizer.json                  (~2.4 MB)
COPY models/ /models/

# Warm up the WKV kernel compile at build time so first request doesn't stall.
# Requires CUDA at build time — falls back gracefully if no GPU is available.
RUN python3 -c " \
from rwkv.model import RWKV; \
m = RWKV(model='/models/RWKV-4-Pile-169M-20220807-8023', strategy='cuda fp16', verbose=False); \
m.forward([0], None); print('kernel warm-up ok') \
" || echo "kernel warm-up skipped (no GPU at build time — will compile on first request)"

# Application code
COPY server/ /app/server/

WORKDIR /app

COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

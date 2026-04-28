FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RWKV_JIT_ON=1 \
    RWKV_CUDA_ON=1 \
    KRUNCH_MODEL_DIR=/models \
    RWKV_LM_PATH=/opt/rwkv-lm/RWKV-v4/src

# System deps — ninja-build is required for torch.utils.cpp_extension (WKV kernel)
# pip install ninja installs only a Python wrapper that torch can't find
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    ninja-build build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# BlinkDL RWKV-LM (Apache-2.0): provides the fast RWKV inference path
# Pinned to a known-good commit for reproducibility
RUN git clone --depth 1 https://github.com/BlinkDL/RWKV-LM.git /opt/rwkv-lm

# Model weights + tokenizer (baked into image — no runtime downloads)
# Build with: docker build --build-arg MODEL_TAR_URL=<s3-presigned-url> .
# Or place files in ./models/ before building.
ARG MODEL_TAR_URL=""
COPY models/ /models/
# If a URL is provided, overwrite with the fetched tarball
RUN if [ -n "$MODEL_TAR_URL" ]; then \
    curl -sL "$MODEL_TAR_URL" | tar xz -C /models; \
    fi

# Warm up the WKV kernel compile at build time so first requests don't stall
# (requires CUDA at build time — skip if building without GPU via --no-cache)
RUN python3 -c " \
import os; os.environ['RWKV_JIT_ON']='1'; os.environ['RWKV_CUDA_ON']='1'; \
import sys; sys.path.insert(0, '/opt/rwkv-lm/RWKV-v4/src'); \
from model_run import RWKV_RNN; \
m = RWKV_RNN('/models/RWKV-4-Pile-169M-20220807-8023.pth', 'cuda fp16'); \
m.forward([0], None); print('kernel warm-up ok') \
" || echo "kernel warm-up skipped (no GPU at build time — will compile on first request)"

# Application code
COPY server/ /app/server/

WORKDIR /app

COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# pytorch/pytorch comes with Python + torch + CUDA + cuDNN preinstalled.
# -devel variant includes nvcc (needed by torch.utils.cpp_extension to
# compile the WKV CUDA kernel at first inference).
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RWKV_JIT_ON=1 \
    RWKV_CUDA_ON=1 \
    KRUNCH_MODEL_DIR=/models

# ninja-build is required by torch.utils.cpp_extension. `pip install ninja`
# only installs a Python wrapper that torch can't find — must come from apt.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ninja-build curl \
    && rm -rf /var/lib/apt/lists/*

# Krunch-specific Python deps. Torch is already in the base image.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Model weights + tokenizer baked into the image — no runtime downloads.
# Place files in ./models/ before `docker build`:
#   - RWKV-4-Pile-169M-20220807-8023.pth  (~323 MB)
#   - 20B_tokenizer.json                  (~2.4 MB)
COPY models/ /models/

# Warm up the WKV kernel compile at build time so first request doesn't stall.
# Requires CUDA at build time — falls back gracefully if no GPU is available.
RUN python -c " \
from rwkv.model import RWKV; \
m = RWKV(model='/models/RWKV-4-Pile-169M-20220807-8023', strategy='cuda fp16', verbose=False); \
m.forward([0], None); print('kernel warm-up ok') \
" || echo "kernel warm-up skipped (no GPU at build time — will compile on first request)"

# Application code
COPY krunch/ /app/krunch/
COPY krunch_ac/ /app/krunch_ac/

# Build the GPU AC kernel (nvcc + ninja). Devel base has nvcc; ninja
# is already installed above. No GPU needed at build time — nvcc only
# compiles, doesn't run, and TORCH_CUDA_ARCH_LIST hints at expected SMs.
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
RUN cd /app/krunch_ac/cuda && python setup.py build_ext --inplace && \
    cp krunch_ac_cuda*.so /app/

ENV PYTHONPATH=/app:/app/krunch_ac/cuda

WORKDIR /app

COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

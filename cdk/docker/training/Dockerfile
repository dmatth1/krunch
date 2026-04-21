# GPU training container for AWS Batch on g5.xlarge (NVIDIA A10G).
#
# Uses the official PyTorch CUDA runtime image so CUDA + cuDNN match
# the PyTorch build. The ECS-GPU-optimized AMI (auto-selected by
# Batch when we set instance class to G5) ships the NVIDIA runtime
# hooks that expose the GPU to the container at runtime.
#
# Built from the repo root:
#   docker build -f cdk/docker/training/Dockerfile .

# devel (not runtime) image: includes nvcc + CUDA headers so torch's
# cpp_extension.load() can JIT-build the RWKV WKV CUDA kernel at first
# import. The -runtime image omits nvcc and makes every training run
# fail with "No such file: wkv_op.cpp" → "nvcc: command not found".
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# awscli for S3 sync in the entrypoint; rest are build essentials in
# case any Python wheel needs to compile. sentencepiece wheel is
# prebuilt on PyPI for linux-x86_64 so we don't need libsentencepiece.
RUN apt-get update && apt-get install -y --no-install-recommends \
    awscli git curl ca-certificates zstd tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY cdk/docker/training/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# --- site-packages stubs for deepspeed + pkuseg ---
# Multiple vendor/L3TC/models/RWKV_V4/*.py files do
# `from deepspeed.ops.adam import FusedAdam` at module top. Real
# deepspeed requires nvcc + CUDA_HOME at IMPORT time, which the
# runtime image lacks. The trainer
# (scripts/train_l3tc_phase11.py) uses torch.optim.AdamW, never
# FusedAdam, so the imports are vestigial — we stub them out so the
# modules load, then let our trainer's real optimizer code run.
# This mirrors phase11_bootstrap_helpers.sh's phase11_write_stubs.
RUN python - <<'PYEOF'
import sysconfig, pathlib
site = pathlib.Path(sysconfig.get_path("purelib"))
(site / "pkuseg").mkdir(parents=True, exist_ok=True)
(site / "pkuseg" / "__init__.py").write_text("# stub: vestigial import in L3TC datasets\n")
ds = site / "deepspeed"
(ds / "ops" / "adam").mkdir(parents=True, exist_ok=True)
(ds / "__init__.py").write_text("# stub\n")
(ds / "ops" / "__init__.py").write_text("")
(ds / "ops" / "adam" / "__init__.py").write_text(
    "class FusedAdam:\n"
    "    def __init__(self, *args, **kwargs):\n"
    "        raise RuntimeError('deepspeed stub: caller should fall back')\n"
)
print("stubs written to", site)
PYEOF

COPY scripts/ /app/scripts/
COPY vendor/L3TC/ /app/vendor/L3TC/

COPY cdk/docker/training/train_entrypoint.sh /app/train_entrypoint.sh
RUN chmod +x /app/train_entrypoint.sh

# Sanity check at build time: confirm torch sees CUDA at image build.
RUN python -c "import torch; print('torch:', torch.__version__, 'cuda_available_at_build:', torch.cuda.is_available())"

ENTRYPOINT ["/app/train_entrypoint.sh"]

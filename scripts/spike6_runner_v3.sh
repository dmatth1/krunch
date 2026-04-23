#!/bin/bash
# Spike 6 runner v3 — throughput-only, with explicit RWKV CUDA kernel loading.
# Drops P1 (ratio is already known from local: ~0.11).
# Focus: resolve the speed question — does fp16 + custom CUDA kernel
# actually deliver the 1 MB/s that ts_zip shows?

set -euxo pipefail

PY=${PY:-/opt/pytorch/bin/python3}
REPO=${REPO:-/home/ubuntu/spike6}
CORPUS=${CORPUS:-/tmp/corpus/content.bin}
S3=s3://archive-dev-archive/spike6/gpu-throughput
LOG_PATH=${LOG_PATH:-/home/ubuntu/spike6_v3.log}
UPLOAD=${UPLOAD:-1}
SHUTDOWN=${SHUTDOWN:-1}

mark() {
    local tag="$1"
    if [ "$UPLOAD" = "1" ]; then
        echo "$(date -u +%FT%TZ) $tag" | aws s3 cp - "$S3/progress_${tag}.txt" >/dev/null 2>&1 || true
    fi
    echo "=== MARK $tag ==="
}

upload() {
    local src="$1" dst="$2"
    if [ "$UPLOAD" = "1" ]; then aws s3 cp "$src" "$dst" --quiet || true; fi
}

mark V3_START

# Ensure ninja is installed (required for on-the-fly CUDA compilation)
# DLAMI PyTorch usually has it; install if not.
$PY -c "import ninja" 2>/dev/null || $PY -m pip install --quiet ninja
$PY -c "import transformers" 2>/dev/null || $PY -m pip install --quiet transformers

# Pre-flight: verify CUDA toolkit is actually present for kernel compile
$PY -c "
import torch, os, shutil
print('torch=', torch.__version__, 'cuda_available=', torch.cuda.is_available())
print('CUDA_HOME=', os.environ.get('CUDA_HOME'))
nvcc = shutil.which('nvcc') or shutil.which('nvcc', path='/usr/local/cuda/bin')
print('nvcc=', nvcc)
"

mark V3_DEPS_OK

# --- P2a: batch=1 throughput with CUDA kernel ---
"$PY" "$REPO/scripts/rwkv_kernel_throughput.py" \
    --model-id RWKV/rwkv-4-169m-pile \
    --content "$CORPUS" \
    --seq-len 1024 --batch-size 1 \
    --warmup-iters 3 --measure-iters 20 \
    --result-path /tmp/rwkv_kernel_b1.json
upload /tmp/rwkv_kernel_b1.json "$S3/rwkv_kernel_b1.json"
mark V3_B1_DONE

# --- P2b: batch=8 throughput ---
"$PY" "$REPO/scripts/rwkv_kernel_throughput.py" \
    --model-id RWKV/rwkv-4-169m-pile \
    --content "$CORPUS" \
    --seq-len 1024 --batch-size 8 \
    --warmup-iters 2 --measure-iters 10 \
    --result-path /tmp/rwkv_kernel_b8.json
upload /tmp/rwkv_kernel_b8.json "$S3/rwkv_kernel_b8.json"
mark V3_B8_DONE

# Upload log
upload "$LOG_PATH" "$S3/spike6_v3.log"
mark V3_ALL_DONE

if [ "$SHUTDOWN" = "1" ]; then
    sudo shutdown -h +1
fi

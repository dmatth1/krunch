#!/bin/bash
# Spike 6 runner v4 — RWKV-LM with custom WKV CUDA kernel.
# Bypasses HF transformers entirely.

set -euxo pipefail

PY=${PY:-/opt/pytorch/bin/python3}
REPO=${REPO:-/home/ubuntu/spike6}
CORPUS=${CORPUS:-/tmp/corpus/content.bin}
BUNDLE_DIR=${BUNDLE_DIR:-/home/ubuntu/rwkv_bundle}
S3=s3://archive-dev-archive/spike6/gpu-throughput
UPLOAD=${UPLOAD:-1}
SHUTDOWN=${SHUTDOWN:-1}

mark() {
    local tag="$1"
    if [ "$UPLOAD" = "1" ]; then
        echo "$(date -u +%FT%TZ) $tag" | aws s3 cp - "$S3/progress_V4_${tag}.txt" >/dev/null 2>&1 || true
    fi
    echo "=== MARK V4_$tag ==="
}

upload() {
    local src="$1" dst="$2"
    if [ "$UPLOAD" = "1" ]; then aws s3 cp "$src" "$dst" --quiet || true; fi
}

mark START

# Deps: tokenizers (for 20B_tokenizer.json), ninja (for cpp_extension.load)
$PY -c "import tokenizers, ninja" 2>/dev/null || $PY -m pip install --quiet tokenizers ninja

# Verify nvcc is on PATH (cpp_extension needs it to compile the kernel)
which nvcc || export PATH=/usr/local/cuda/bin:$PATH
nvcc --version || true

mark DEPS_OK

# --- batch=1 ---
"$PY" "$REPO/scripts/rwkv_lm_throughput.py" \
    --rwkv-lm-dir "$BUNDLE_DIR/RWKV-v4" \
    --model-path "$BUNDLE_DIR/RWKV-4-Pile-169M-20220807-8023" \
    --tokenizer-json "$BUNDLE_DIR/RWKV-v4/20B_tokenizer.json" \
    --content "$CORPUS" \
    --seq-len 1024 --batch-size 1 \
    --warmup-iters 3 --measure-iters 20 \
    --dtype fp16 \
    --result-path /tmp/rwkv_lm_b1.json
upload /tmp/rwkv_lm_b1.json "$S3/rwkv_lm_b1.json"
mark B1_DONE

# --- batch=8 ---
"$PY" "$REPO/scripts/rwkv_lm_throughput.py" \
    --rwkv-lm-dir "$BUNDLE_DIR/RWKV-v4" \
    --model-path "$BUNDLE_DIR/RWKV-4-Pile-169M-20220807-8023" \
    --tokenizer-json "$BUNDLE_DIR/RWKV-v4/20B_tokenizer.json" \
    --content "$CORPUS" \
    --seq-len 1024 --batch-size 8 \
    --warmup-iters 2 --measure-iters 10 \
    --dtype fp16 \
    --result-path /tmp/rwkv_lm_b8.json
upload /tmp/rwkv_lm_b8.json "$S3/rwkv_lm_b8.json"
mark B8_DONE

mark ALL_DONE

if [ "$SHUTDOWN" = "1" ]; then
    sudo shutdown -h +1
fi

#!/bin/bash
# Spike 6 runner v2 — minimal. Two measurements only:
#   P1: RWKV-4-169m zero-shot entropy ratio on 20 MB held-out chat
#   P2: RWKV-4-169m throughput on GPU (batch=1 and batch=8)
#
# Works on any DLAMI PyTorch AMI. Uses /opt/pytorch/bin/python3
# directly (no source/activate). Writes a progress sentinel to S3 at
# each step so external observers can tell exactly where any failure
# occurred.
#
# Env knobs (for local testing):
#   PY            default=/opt/pytorch/bin/python3 — pass another path to test
#   DEVICE        default=cuda — pass `cpu` to dry-run on laptop
#   MAX_BYTES     default=20000000 — pass smaller for quick local run
#   MEASURE_ITERS default=20 — pass smaller for quick local run

set -euxo pipefail

PY=${PY:-/opt/pytorch/bin/python3}
DEVICE=${DEVICE:-cuda}
MAX_BYTES=${MAX_BYTES:-20000000}
MEASURE_ITERS=${MEASURE_ITERS:-20}
REPO=${REPO:-/home/ubuntu/spike6}
CORPUS=${CORPUS:-/tmp/corpus/content.bin}
S3=s3://archive-dev-archive/spike6/gpu-throughput
LOG_PATH=${LOG_PATH:-/home/ubuntu/spike6.log}
UPLOAD=${UPLOAD:-1}   # set 0 for local testing
SHUTDOWN=${SHUTDOWN:-1}  # set 0 for local testing

mark() {
    local tag="$1"
    if [ "$UPLOAD" = "1" ]; then
        echo "$(date -u +%FT%TZ) $tag" | aws s3 cp - "$S3/progress_${tag}.txt" >/dev/null 2>&1 || true
    fi
    echo "=== MARK $tag ==="
}

upload() {
    local src="$1" dst="$2"
    if [ "$UPLOAD" = "1" ]; then
        aws s3 cp "$src" "$dst" --quiet || true
    fi
}

mark PY_OK
# Prove python + torch + CUDA are reachable before we spend time on measurements
$PY -c "
import torch
print('torch=', torch.__version__)
if '$DEVICE' == 'cuda':
    assert torch.cuda.is_available(), 'no CUDA'
    print('cuda=', torch.cuda.get_device_name(0))
elif '$DEVICE' == 'mps':
    assert torch.backends.mps.is_available(), 'no MPS'
print('device=$DEVICE  ok')
"

# Only install what's missing. transformers is the only hard dep;
# peft and deepspeed are NOT needed for the zero-shot + throughput path.
$PY -c "import transformers" 2>/dev/null || $PY -m pip install --quiet transformers

mark DEPS_OK

# --- P1: RWKV-4-169m zero-shot ratio ---
mark P1_START
"$PY" "$REPO/scripts/rwkv_zeroshot_ratio.py" \
    --model-id RWKV/rwkv-4-169m-pile \
    --content "$CORPUS" \
    --max-bytes "$MAX_BYTES" \
    --seq-len 1024 \
    --dtype bf16 --device "$DEVICE" \
    --result-path /tmp/rwkv_zeroshot.json
upload /tmp/rwkv_zeroshot.json "$S3/rwkv_zeroshot.json"
mark P1_DONE

# --- P2a: RWKV-4-169m throughput, batch=1 (realistic per-stream) ---
mark P2_START
"$PY" "$REPO/scripts/smollm2_throughput_test.py" \
    --base-model RWKV/rwkv-4-169m-pile \
    --content "$CORPUS" \
    --seq-len 1024 --batch-size 1 \
    --warmup-iters 2 --measure-iters "$MEASURE_ITERS" \
    --dtype bf16 --device "$DEVICE" \
    --result-path /tmp/rwkv_throughput_b1.json
upload /tmp/rwkv_throughput_b1.json "$S3/rwkv_throughput_b1.json"

# --- P2b: RWKV throughput batch=8 (GPU batching lever) ---
# Skip if CPU (batching hurts on CPU, we already measured this).
if [ "$DEVICE" != "cpu" ]; then
    "$PY" "$REPO/scripts/smollm2_throughput_test.py" \
        --base-model RWKV/rwkv-4-169m-pile \
        --content "$CORPUS" \
        --seq-len 1024 --batch-size 8 \
        --warmup-iters 2 --measure-iters 10 \
        --dtype bf16 --device "$DEVICE" \
        --result-path /tmp/rwkv_throughput_b8.json
    upload /tmp/rwkv_throughput_b8.json "$S3/rwkv_throughput_b8.json"
fi
mark P2_DONE

# Upload full log + finalize
upload "$LOG_PATH" "$S3/spike6.log"
mark ALL_DONE

if [ "$SHUTDOWN" = "1" ]; then
    sudo shutdown -h +1
fi

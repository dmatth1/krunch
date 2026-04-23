#!/bin/bash
# Spike 6 GPU runner — one-shot script for g5.2xlarge with DLAMI PyTorch AMI.
#
# Runs three measurements back-to-back and uploads results to S3:
#   3a. L3TC-12M v1.pth GPU throughput (resolves Track 2)
#   3b. RWKV-4-169m-pile zero-shot ratio on WildChat-English
#   3c. RWKV-4-169m-pile + LoRA fine-tuned ratio
#
# This script is idempotent for the install steps and emits progress
# sentinels to S3 so we can tell from outside exactly where any
# failure occurred. Self-terminates on completion.

set -euxo pipefail

S3_OUT="s3://archive-dev-archive/spike6/gpu-throughput"
MODEL_S3="s3://archive-dev-archive/spike5/wildchat-en/models"
CORPUS_S3="s3://archive-dev-archive/spike6/corpus"

mark() {
    # Write a sentinel to S3 so outside observers can see progress.
    local tag="$1"
    echo "$(date -u +%FT%TZ) $tag" | aws s3 cp - "$S3_OUT/progress_${tag}.txt" 2>/dev/null || true
}

# 2-hour watchdog — independent of the main script succeeding.
(sleep 7200 && sudo shutdown -h +1) &

mark START

# The DLAMI PyTorch AMI already has: CUDA 12.x, torch 2.7, transformers.
# We only need to add: sentencepiece (tokenizer) + peft (LoRA).
source /opt/pytorch/bin/activate 2>/dev/null || source /home/ubuntu/pytorch/bin/activate 2>/dev/null || true
python3 -m pip install --quiet sentencepiece peft deepspeed 2>&1 | tail -3 || true

mark DEPS_OK

# Stage code + data
mkdir -p /home/ubuntu/spike6 /tmp/model /tmp/corpus
cd /home/ubuntu/spike6

# Pull code bundle (uploaded by laptop before launch)
aws s3 cp "$S3_OUT/code.tar.gz" /tmp/code.tar.gz
tar xzf /tmp/code.tar.gz -C /home/ubuntu/spike6
ls /home/ubuntu/spike6

# Pull model artifacts
aws s3 cp "$MODEL_S3/v1.pth" /tmp/model/v1.pth --quiet
aws s3 cp "$MODEL_S3/v1.tokenizer.model" /tmp/model/v1.tokenizer.model --quiet

# Pull corpus (raw NDJSON for Track 2, content stream for Track 3)
aws s3 cp "$CORPUS_S3/wildchat_en_200mb.ndjson" /tmp/corpus/wildchat_en_200mb.ndjson --quiet
aws s3 cp "$CORPUS_S3/wildchat_en_content.content.bin" /tmp/corpus/content.bin --quiet

ls -lh /tmp/model /tmp/corpus

mark STAGED

# --- Track 2: L3TC-12M throughput (3a) ---
mark TRACK2_START
python3 scripts/gpu_throughput_test.py \
    --checkpoint /tmp/model/v1.pth \
    --tokenizer /tmp/model/v1.tokenizer.model \
    --corpus /tmp/corpus/wildchat_en_200mb.ndjson \
    --num-layers 4 --vocab-size 16384 --hidden-size 384 \
    --intermediate-size 1024 --rwkv-rank 4 --ctx-len 2048 \
    --chunk-size-bytes 65536 --max-chunks 100 --warmup-iters 3 \
    --dtype bf16 --device cuda \
    --repo-root /home/ubuntu/spike6 \
    --result-path /tmp/l3tc_throughput.json 2>&1 | tee /tmp/l3tc_throughput.log
aws s3 cp /tmp/l3tc_throughput.json "$S3_OUT/l3tc_throughput.json"
aws s3 cp /tmp/l3tc_throughput.log "$S3_OUT/l3tc_throughput.log"
mark TRACK2_DONE

# --- Track 3b: RWKV-4-169m zero-shot ratio ---
mark TRACK3B_START
python3 scripts/rwkv_zeroshot_ratio.py \
    --model-id RWKV/rwkv-4-169m-pile \
    --content /tmp/corpus/content.bin \
    --max-bytes 20000000 \
    --seq-len 1024 \
    --dtype bf16 --device cuda \
    --result-path /tmp/rwkv_zeroshot.json 2>&1 | tee /tmp/rwkv_zeroshot.log
aws s3 cp /tmp/rwkv_zeroshot.json "$S3_OUT/rwkv_zeroshot.json"
aws s3 cp /tmp/rwkv_zeroshot.log "$S3_OUT/rwkv_zeroshot.log"
mark TRACK3B_DONE

# --- Track 3c: RWKV-4-169m + LoRA fine-tune + ratio ---
mark TRACK3C_START
python3 scripts/rwkv_lora_train.py \
    --base-model RWKV/rwkv-4-169m-pile \
    --content-stream /tmp/corpus/content.bin \
    --train-val-split 0.90 \
    --seq-len 1024 --batch-size 8 \
    --epochs 2 --epoch-length 5000 \
    --lr 1e-4 --lora-r 16 --lora-alpha 32 \
    --eval-chunks 100 \
    --dtype bf16 --device cuda \
    --output-dir /tmp/rwkv_lora_adapter \
    --result-path /tmp/rwkv_lora.json 2>&1 | tee /tmp/rwkv_lora.log
tar czf /tmp/rwkv_lora_adapter.tar.gz -C /tmp rwkv_lora_adapter
aws s3 cp /tmp/rwkv_lora.json "$S3_OUT/rwkv_lora.json"
aws s3 cp /tmp/rwkv_lora.log "$S3_OUT/rwkv_lora.log"
aws s3 cp /tmp/rwkv_lora_adapter.tar.gz "$S3_OUT/rwkv_lora_adapter.tar.gz"
mark TRACK3C_DONE

mark ALL_DONE
sudo shutdown -h +1

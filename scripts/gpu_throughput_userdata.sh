#!/bin/bash
# User-data script for g5.xlarge GPU throughput test.
# Runs once on boot, uploads result to S3, terminates instance.
# Safe to re-run (idempotent).

set -euxo pipefail

# Hard timeout: terminate instance after 2 hours no matter what, to
# avoid cost runaway if something hangs.
(sleep 7200 && shutdown -h now) &

exec > >(tee /var/log/gpu-test.log | logger -t gpu-test) 2>&1

S3_PREFIX="s3://archive-dev-archive/spike6/gpu-throughput"
MODEL_S3="s3://archive-dev-archive/spike5/wildchat-en/models"
CORPUS_S3="s3://archive-dev-archive/spike6/corpus"

apt-get update -y
apt-get install -y python3-pip python3-venv awscli git

# Clone the repo (public)
cd /root
git clone https://github.com/dmatth1/Krunch.git || (cd Krunch && git pull)
cd Krunch
ln -sf /root/Krunch /app

# Pull vendor L3TC if not present
[ -d vendor/L3TC ] || git submodule update --init --recursive || true

# Setup Python env — DLAMI has CUDA already, we just need torch + sentencepiece
python3 -m venv /opt/venv
source /opt/venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install sentencepiece numpy

# Pull model + tokenizer + corpus
mkdir -p /tmp/model /tmp/corpus
aws s3 cp "$MODEL_S3/v1.pth" /tmp/model/v1.pth
aws s3 cp "$MODEL_S3/v1.tokenizer.model" /tmp/model/v1.tokenizer.model
aws s3 cp "$CORPUS_S3/wildchat_en_200mb.ndjson" /tmp/corpus/wildchat_en_200mb.ndjson

# Find first corpus file
CORPUS_FILE=$(ls /tmp/corpus/*.ndjson 2>/dev/null | head -1)
if [ -z "$CORPUS_FILE" ]; then
    echo "FATAL: no corpus file found in /tmp/corpus"
    aws s3 cp /var/log/gpu-test.log "$S3_PREFIX/fail.log"
    shutdown -h now
    exit 1
fi
echo "using corpus: $CORPUS_FILE"

# Run the throughput test
cd /app
python scripts/gpu_throughput_test.py \
    --checkpoint /tmp/model/v1.pth \
    --tokenizer /tmp/model/v1.tokenizer.model \
    --corpus "$CORPUS_FILE" \
    --num-layers 4 --vocab-size 16384 --hidden-size 384 \
    --intermediate-size 1024 --rwkv-rank 4 --ctx-len 2048 \
    --chunk-size-bytes 65536 --max-chunks 50 \
    --dtype bf16 \
    --result-path /tmp/gpu_result.json 2>&1 | tee /tmp/gpu_run.log

# Also try fp16 for comparison
python scripts/gpu_throughput_test.py \
    --checkpoint /tmp/model/v1.pth \
    --tokenizer /tmp/model/v1.tokenizer.model \
    --corpus "$CORPUS_FILE" \
    --num-layers 4 --vocab-size 16384 --hidden-size 384 \
    --intermediate-size 1024 --rwkv-rank 4 --ctx-len 2048 \
    --chunk-size-bytes 65536 --max-chunks 50 \
    --dtype fp16 \
    --result-path /tmp/gpu_result_fp16.json 2>&1 | tee -a /tmp/gpu_run.log

# Upload results
aws s3 cp /tmp/gpu_result.json "$S3_PREFIX/result_bf16.json"
aws s3 cp /tmp/gpu_result_fp16.json "$S3_PREFIX/result_fp16.json"
aws s3 cp /tmp/gpu_run.log "$S3_PREFIX/run.log"
aws s3 cp /var/log/gpu-test.log "$S3_PREFIX/userdata.log"

# Terminate
shutdown -h now

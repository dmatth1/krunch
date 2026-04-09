#!/bin/bash
# User data script for L3TC Phase 11 training on Spot Fleet.
# Runs automatically on every new instance launch.
#
# This script is STATELESS — it doesn't know if it's the first instance or a
# replacement after spot reclaim. It just checks S3 for the latest .pth
# checkpoint for this RUN_ID and resumes from there (or starts fresh).
#
# Placeholders substituted by launch-spot-fleet.sh at launch time:
#   PLACEHOLDER_RUN_ID, PLACEHOLDER_PASS, PLACEHOLDER_GITHUB_PAT,
#   PLACEHOLDER_S3_BUCKET, PLACEHOLDER_S3_PREFIX

set -euo pipefail
exec > /var/log/l3tc-bootstrap.log 2>&1

# === Configuration (substituted by launcher) ===
GITHUB_PAT="PLACEHOLDER_GITHUB_PAT"
S3_BUCKET="PLACEHOLDER_S3_BUCKET"
S3_PREFIX="PLACEHOLDER_S3_PREFIX"
RUN_ID="PLACEHOLDER_RUN_ID"
PASS="PLACEHOLDER_PASS"          # pass1 (enwik9 sanity) or pass2 (Pile broader)
S3_RUN_PATH="s3://${S3_BUCKET}/${S3_PREFIX}/${RUN_ID}"
S3_CORPUS_PATH="s3://${S3_BUCKET}/${S3_PREFIX}/corpora"

# === Per-pass settings ===
case "$PASS" in
    pass1)
        # Pass 1: train on enwik9 (1 GB) — same domain as the existing
        # enwik8-trained model, just 10x more data. Sanity check that
        # the cloud pipeline works end-to-end and produces a checkpoint
        # that converts cleanly into our Rust runtime format. Expected
        # to slightly improve enwik6 ratio (more in-domain data) without
        # changing speed.
        CORPUS_NAME="enwik9"
        CORPUS_S3="${S3_CORPUS_PATH}/enwik9.xz"
        TRAIN_FILE="./data/train_data/train_enwik9_bpe_16384_0.999.txt"
        EPOCHS=15
        ;;
    pass2)
        # Pass 2: train on the Pile dedup subset (~50 GB). Tests whether
        # a single 200K model can cover the broader distribution without
        # breaking the in-distribution ratio floor (enwik6 <= 0.20). The
        # ratio matrix and decision criteria are in PHASE_11.md.
        CORPUS_NAME="pile_dedup_50gb"
        CORPUS_S3="${S3_CORPUS_PATH}/pile_dedup_50gb.tar"
        TRAIN_FILE="./data/train_data/train_pile_dedup_bpe_16384_0.999.txt"
        EPOCHS=10
        ;;
    *)
        echo "ERROR: unknown PASS '${PASS}'"
        exit 1
        ;;
esac

# === System setup ===
echo "=== L3TC Phase 11 Spot Fleet instance starting ==="
echo "Run ID:      ${RUN_ID}"
echo "Pass:        ${PASS}"
echo "Corpus:      ${CORPUS_NAME}"
echo "S3 path:     ${S3_RUN_PATH}"
echo "Time:        $(date -u)"
echo "============================================"

apt-get update -qq
apt-get install -y -qq python3-venv python3-pip awscli xz-utils > /dev/null 2>&1

# === Background log uploader (so we can tail bootstrap from S3) ===
(
    while true; do
        sleep 30
        aws s3 cp /var/log/l3tc-bootstrap.log "${S3_RUN_PATH}/bootstrap.log" --quiet 2>/dev/null || true
    done
) &
BOOTSTRAP_LOG_PID=$!

# === Clone l3tc-prod (the GitHub repo is named "ltec", not "l3tc-prod") ===
cd /home/ubuntu
git clone "https://x-access-token:${GITHUB_PAT}@github.com/dmatth1/ltec.git" l3tc-prod
cd l3tc-prod
git log -1 --oneline
chown -R ubuntu:ubuntu /home/ubuntu/l3tc-prod

# === Run setup.sh to clone vendor/L3TC + RWKV-LM ===
echo "=== Running scripts/setup.sh to clone L3TC + RWKV-LM ==="
bash scripts/setup.sh

# === Python environment for L3TC training ===
cd vendor/L3TC
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
# CUDA-PyTorch matching the host CUDA. Deep Learning Base AMI ships with
# a recent CUDA; install the matching wheel index.
pip install torch --index-url https://download.pytorch.org/whl/cu121 -q
# L3TC's pinned requirements
pip install -r requirements.txt -q
# Sanity check GPU and L3TC's CUDA WKV kernel
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# === Download corpus from S3 ===
mkdir -p data/raw_text_data data/train_data data/test_data
echo "=== Downloading corpus ${CORPUS_NAME} ==="
case "$PASS" in
    pass1)
        # enwik9: stored as xz-compressed (.xz) in S3 to keep transfer small
        if ! aws s3 ls "${CORPUS_S3}" >/dev/null 2>&1; then
            echo "ERROR: enwik9 not found at ${CORPUS_S3}"
            echo "Upload it once from your local machine:"
            echo "  aws s3 cp ~/.../enwik9.xz ${CORPUS_S3}"
            echo "  (or download from http://mattmahoney.net/dc/enwik9.zip and re-pack as .xz)"
            exit 1
        fi
        aws s3 cp "${CORPUS_S3}" /tmp/enwik9.xz --quiet
        xz -d /tmp/enwik9.xz
        mv /tmp/enwik9 data/raw_text_data/enwik9
        ls -lh data/raw_text_data/enwik9
        ;;
    pass2)
        # Pile dedup: pre-extracted, pre-formatted single text file in S3.
        # The build-corpus step (one-time) creates this from
        # https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated
        # (or successor) and uploads the concatenated raw text to
        # ${CORPUS_S3}. Document the build steps in scripts/build_pile_corpus.py
        # before pass 2 is launched.
        if ! aws s3 ls "${CORPUS_S3}" >/dev/null 2>&1; then
            echo "ERROR: Pile dedup corpus not found at ${CORPUS_S3}"
            echo "Run scripts/build_pile_corpus.py locally or on a beefier"
            echo "instance to assemble it before launching pass2."
            exit 1
        fi
        aws s3 cp "${CORPUS_S3}" /tmp/pile_dedup.tar --quiet
        tar -xf /tmp/pile_dedup.tar -C data/raw_text_data/
        ls -lh data/raw_text_data/
        ;;
esac

# === Tokenize corpus with the existing L3TC SPM tokenizer ===
# We deliberately reuse the enwik8-trained SPM (vocab 16384, 0.999 coverage)
# rather than retraining the tokenizer. Per Phase 11 hard constraint #1,
# the architecture and tokenizer stay constant; only the corpus changes.
# A tokenizer retrain on the broader corpus is a separate follow-up if
# Phase 11 reveals the enwik tokenizer is the bottleneck.
echo "=== Tokenizing corpus with vocab_enwik8_bpe_16384_0.999 ==="
SPM_MODEL="dictionary/vocab_enwik8_bpe_16384_0.999/spm_enwik8_bpe_16384_0.999.model"
case "$PASS" in
    pass1)
        RAW_TEXT="data/raw_text_data/enwik9"
        ;;
    pass2)
        # Pile corpus is pre-shuffled, single text file
        RAW_TEXT="data/raw_text_data/pile_dedup.txt"
        ;;
esac

# Encode the raw text into the format L3TC's EnWikTrainDataSet expects:
# one token id per line as integers, with <s> as the BOS marker between
# segments. The L3TC reference uses scripts/preprocessor.py for this on
# enwik8; we run the same flow on the bigger corpus.
python -c "
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load('${SPM_MODEL}')
print(f'vocab size: {sp.vocab_size()}')
print(f'BOS id:     {sp.bos_id()}')
with open('${RAW_TEXT}', 'r', encoding='utf-8', errors='replace') as f_in, \
     open('${TRAIN_FILE}', 'w') as f_out:
    bytes_done = 0
    while True:
        chunk = f_in.read(1_000_000)  # 1 MB at a time to keep memory bounded
        if not chunk:
            break
        ids = sp.encode_as_ids(chunk)
        # Write one token per line, BOS-prefixed, matching L3TC format
        f_out.write('2\n')  # BOS
        for tid in ids:
            f_out.write(f'{tid}\n')
        bytes_done += len(chunk.encode('utf-8'))
        if bytes_done % (50 * 1024 * 1024) == 0:
            print(f'  tokenized {bytes_done // (1024*1024)} MB')
print(f'done: ${TRAIN_FILE}')
"
ls -lh "${TRAIN_FILE}"

# === Find latest checkpoint for this run (resume after spot reclaim) ===
RESUME_FLAG=""
LATEST_CKPT=$(aws s3 ls "${S3_RUN_PATH}/" 2>/dev/null | grep -E "checkpoint.*\.pth" | sort | tail -1 || true)
if [ -n "$LATEST_CKPT" ]; then
    CKPT_NAME=$(echo "$LATEST_CKPT" | awk '{print $4}')
    echo "Found checkpoint for run ${RUN_ID}: ${CKPT_NAME}"
    aws s3 cp "${S3_RUN_PATH}/${CKPT_NAME}" "checkpoint/${CKPT_NAME}" --quiet
    mkdir -p checkpoint
    RESUME_FLAG="--resume checkpoint/${CKPT_NAME}"
    echo "Will resume from ${CKPT_NAME}"
else
    echo "No checkpoint found for run ${RUN_ID}, starting fresh"
fi

# === Background checkpoint + log uploader (every 5 minutes) ===
(
    while true; do
        sleep 300
        # Upload the latest checkpoint we can find
        if compgen -G "checkpoint/checkpoint*.pth" > /dev/null; then
            for ckpt in checkpoint/checkpoint*.pth; do
                aws s3 cp "$ckpt" "${S3_RUN_PATH}/$(basename "$ckpt")" --quiet 2>/dev/null || true
            done
        fi
        aws s3 cp /home/ubuntu/l3tc-prod/vendor/L3TC/train.log "${S3_RUN_PATH}/train.log" --quiet 2>/dev/null || true
    done
) &
TRAIN_UPLOAD_PID=$!

# === Stop the bootstrap-log uploader, training takes over the log slot ===
kill ${BOOTSTRAP_LOG_PID} 2>/dev/null || true

# === Run L3TC training ===
# We override `train_file` via L3TC's --options mechanism so we don't have
# to commit a new config file for each pass.
echo "=== Starting L3TC training (pass ${PASS}, ${EPOCHS} epochs) ==="
date -u

python main.py \
    --config_file config/l3tc/l3tc_200k.py \
    --output_dir checkpoint \
    --device cuda \
    --options train_file="${TRAIN_FILE}" epoch=${EPOCHS} \
    ${RESUME_FLAG} \
    2>&1 | tee train.log
TRAIN_EXIT=${PIPESTATUS[0]}

echo "=== Training exited with code ${TRAIN_EXIT} ==="
date -u

# === Final upload: checkpoint + log ===
if compgen -G "checkpoint/checkpoint*.pth" > /dev/null; then
    for ckpt in checkpoint/checkpoint*.pth; do
        aws s3 cp "$ckpt" "${S3_RUN_PATH}/$(basename "$ckpt")" --quiet
    done
fi
aws s3 cp train.log "${S3_RUN_PATH}/train.log" --quiet

# Kill background uploader
kill ${TRAIN_UPLOAD_PID} 2>/dev/null || true

echo "=== Done. Spot fleet stays in maintain mode; cancel manually when ready. ==="
echo "  aws ec2 cancel-spot-fleet-requests --spot-fleet-request-ids <FLEET_ID> --terminate-instances --region us-east-1"

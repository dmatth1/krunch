#!/bin/bash
# User data script for L3TC Phase 11 training on Spot Fleet.
# Stateless: every fresh instance OR replacement after spot reclaim
# runs the same script. Resumes from latest .pth checkpoint in S3
# if present. Most setup logic lives in scripts/phase11_bootstrap_helpers.sh
# in the repo so this file stays under the 16 KB EC2 user data limit.
#
# Placeholders substituted by launch-spot-fleet.sh at launch time:
#   PLACEHOLDER_RUN_ID, PLACEHOLDER_PASS, PLACEHOLDER_GITHUB_PAT,
#   PLACEHOLDER_S3_BUCKET, PLACEHOLDER_S3_PREFIX

set -euo pipefail
exec > /var/log/l3tc-bootstrap.log 2>&1

GITHUB_PAT="PLACEHOLDER_GITHUB_PAT"
S3_BUCKET="PLACEHOLDER_S3_BUCKET"
S3_PREFIX="PLACEHOLDER_S3_PREFIX"
RUN_ID="PLACEHOLDER_RUN_ID"
PASS="PLACEHOLDER_PASS"
S3_RUN_PATH="s3://${S3_BUCKET}/${S3_PREFIX}/${RUN_ID}"
S3_CORPUS_PATH="s3://${S3_BUCKET}/${S3_PREFIX}/corpora"

case "$PASS" in
    pass1)
        CORPUS_NAME="enwik9"
        CORPUS_S3="${S3_CORPUS_PATH}/enwik9.xz"
        TRAIN_FILE="./data/train_data/train_enwik9_bpe_16384_0.999.txt"
        EPOCHS=15
        ;;
    pass2)
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

echo "=== L3TC Phase 11 instance starting ==="
echo "Run ID: ${RUN_ID}, Pass: ${PASS}, Corpus: ${CORPUS_NAME}"
echo "S3 path: ${S3_RUN_PATH}"
date -u

apt-get update -qq
apt-get install -y -qq python3-venv python3-pip awscli xz-utils > /dev/null 2>&1

# Background log uploader so bootstrap can be tailed from S3 every 30s
(
    while true; do
        sleep 30
        aws s3 cp /var/log/l3tc-bootstrap.log "${S3_RUN_PATH}/bootstrap.log" --quiet 2>/dev/null || true
    done
) &
BOOTSTRAP_LOG_PID=$!

# === Clone the repo (named "ltec" on GitHub, not "l3tc-prod") ===
cd /home/ubuntu
git clone "https://x-access-token:${GITHUB_PAT}@github.com/dmatth1/ltec.git" l3tc-prod
cd l3tc-prod
git log -1 --oneline
chown -R ubuntu:ubuntu /home/ubuntu/l3tc-prod

# === Run setup.sh to clone vendor/L3TC + RWKV-LM ===
echo "=== scripts/setup.sh ==="
bash scripts/setup.sh

# === Source the bootstrap helpers and run them ===
export S3_BUCKET S3_PREFIX S3_RUN_PATH S3_CORPUS_PATH RUN_ID PASS \
       CORPUS_NAME CORPUS_S3 TRAIN_FILE EPOCHS
source /home/ubuntu/l3tc-prod/scripts/phase11_bootstrap_helpers.sh

phase11_install_python_deps
phase11_write_stubs
phase11_check_gpu

cd /home/ubuntu/l3tc-prod/vendor/L3TC
source .venv/bin/activate
phase11_pull_spm_tokenizer

# Pass-specific corpus fetch
case "$PASS" in
    pass1) phase11_fetch_corpus_pass1 ;;
    pass2) echo "pass2 corpus fetch not yet implemented"; exit 1 ;;
esac

# Tokenize if we don't already have a pre-tokenized file from S3
if [ "${SKIP_TOKENIZE:-0}" != "1" ]; then
    phase11_tokenize_corpus "data/raw_text_data/${CORPUS_NAME}" "${TRAIN_FILE}"
fi

# === Find latest checkpoint for this run (resume after spot reclaim) ===
RESUME_FLAG=""
LATEST_CKPT=$(aws s3 ls "${S3_RUN_PATH}/" 2>/dev/null | grep -E "checkpoint.*\.pth" | sort | tail -1 || true)
if [ -n "$LATEST_CKPT" ]; then
    CKPT_NAME=$(echo "$LATEST_CKPT" | awk '{print $4}')
    echo "Found checkpoint ${CKPT_NAME}, resuming"
    mkdir -p checkpoint
    aws s3 cp "${S3_RUN_PATH}/${CKPT_NAME}" "checkpoint/${CKPT_NAME}" --quiet
    RESUME_FLAG="--resume checkpoint/${CKPT_NAME}"
fi

# === Background checkpoint + log uploader ===
(
    while true; do
        sleep 300
        if compgen -G "checkpoint/checkpoint*.pth" > /dev/null; then
            for ckpt in checkpoint/checkpoint*.pth; do
                aws s3 cp "$ckpt" "${S3_RUN_PATH}/$(basename "$ckpt")" --quiet 2>/dev/null || true
            done
        fi
        aws s3 cp /home/ubuntu/l3tc-prod/vendor/L3TC/train.log "${S3_RUN_PATH}/train.log" --quiet 2>/dev/null || true
    done
) &
TRAIN_UPLOAD_PID=$!
kill ${BOOTSTRAP_LOG_PID} 2>/dev/null || true

# === Run L3TC training ===
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

echo "=== Training exited: ${TRAIN_EXIT} ==="
date -u

# Final upload
if compgen -G "checkpoint/checkpoint*.pth" > /dev/null; then
    for ckpt in checkpoint/checkpoint*.pth; do
        aws s3 cp "$ckpt" "${S3_RUN_PATH}/$(basename "$ckpt")" --quiet
    done
fi
aws s3 cp train.log "${S3_RUN_PATH}/train.log" --quiet
kill ${TRAIN_UPLOAD_PID} 2>/dev/null || true

echo "=== Done. Cancel the spot fleet manually when ready. ==="

#!/usr/bin/env bash
# Training job entrypoint for AWS Batch.
#
# Env vars (set by the launcher Lambda in CDK):
#   CUSTOMER_ID, DATASET_ID         — identify the training target
#   BUCKET_NAME                     — S3 bucket for raw + models
#   S3_RAW_PREFIX                   — s3://bucket/<prefix> for raw NDJSON
#   S3_MODEL_PREFIX                 — s3://bucket/<prefix> for output model
#   TRIGGER                         — "initial" | "retrain"
#
# Flow:
#   1. Sync raw NDJSON from S3 → /tmp/raw/
#   2. Concatenate into a single corpus + split 80/20 train/val
#   3. Train an SPM tokenizer + the RWKV model (existing scripts)
#   4. Measure held-out ratio vs zstd-22 baseline
#   5. If our ratio beats zstd by >=2%, mark codec=l3tc; else codec=zstd_fallback
#   6. Convert .pth → .bin (Rust-compatible)
#   7. Upload model.bin + tokenizer.model + metadata.json to S3

set -euo pipefail

: "${CUSTOMER_ID:?required}"
: "${DATASET_ID:?required}"
: "${BUCKET_NAME:?required}"
: "${S3_RAW_PREFIX:?required}"
: "${S3_MODEL_PREFIX:?required}"
: "${TRIGGER:=initial}"

echo "[train] customer=${CUSTOMER_ID} dataset=${DATASET_ID} trigger=${TRIGGER}"

RAW_DIR=/tmp/raw
MODEL_DIR=/tmp/model
mkdir -p "$RAW_DIR" "$MODEL_DIR"

echo "[train] downloading raw data..."
aws s3 sync "s3://${BUCKET_NAME}/${S3_RAW_PREFIX}" "$RAW_DIR/" --only-show-errors

# Concatenate every raw NDJSON into a single corpus.
cat "$RAW_DIR"/*.ndjson > "$MODEL_DIR/corpus.txt"
corpus_bytes=$(wc -c < "$MODEL_DIR/corpus.txt")
echo "[train] corpus bytes: ${corpus_bytes}"

# 80/20 train/val split on byte offset (Phase 11 convention).
train_bytes=$(( corpus_bytes * 80 / 100 ))
head -c "$train_bytes" "$MODEL_DIR/corpus.txt" > "$MODEL_DIR/train.txt"
tail -c +$(( train_bytes + 1 )) "$MODEL_DIR/corpus.txt" > "$MODEL_DIR/val.txt"

# Determine next model version by listing existing models in S3.
existing_versions=$(aws s3 ls "s3://${BUCKET_NAME}/${S3_MODEL_PREFIX}" 2>/dev/null \
  | awk '{print $4}' | grep -oE '^v[0-9]+\.bin$' | sed 's/v//;s/\.bin//' | sort -n || true)
if [ -z "$existing_versions" ]; then
  VERSION=1
else
  VERSION=$(( $(echo "$existing_versions" | tail -1) + 1 ))
fi
echo "[train] assigning model version v${VERSION}"

# Train SPM tokenizer.
echo "[train] training SPM tokenizer..."
python /app/scripts/train_specialist_tokenizer.py \
    --input "$MODEL_DIR/train.txt" \
    --output-dir "$MODEL_DIR" \
    --vocab-size 16384 || {
  echo "[train] tokenizer training failed — aborting"
  exit 2
}

# Train RWKV-v4 2L × 96H on the corpus.
echo "[train] training RWKV-v4 model..."
python /app/scripts/train_l3tc_phase11.py \
    --train-file "$MODEL_DIR/train.txt" \
    --val-file "$MODEL_DIR/val.txt" \
    --output-dir "$MODEL_DIR/train_out" \
    --epochs 5 \
    --epoch-length 50000 \
    --batch-size 16 \
    --num-layers 2 \
    --vocab-size 16384 \
    --lr 1e-4 \
    --device cpu \
    --no-bf16 \
    --no-compile \
    --num-workers 2 || {
  echo "[train] RWKV training failed — aborting"
  exit 3
}

# Convert .pth → .bin.
echo "[train] converting to .bin..."
python /app/scripts/convert_checkpoint.py \
    --input "$MODEL_DIR/train_out/checkpoint_final.pth" \
    --output "$MODEL_DIR/v${VERSION}.bin"

# Measure held-out ratio vs zstd.
echo "[train] measuring held-out ratio vs zstd-22..."
held_out_ratio=$(python /app/scripts/measure_held_out_ratio.py \
    --val-file "$MODEL_DIR/val.txt" \
    --model "$MODEL_DIR/v${VERSION}.bin" \
    --tokenizer "$MODEL_DIR/spm.model" 2>/dev/null \
    || echo "1.0")

zstd_baseline_ratio=$(zstd --long=27 --ultra -22 --stdout --quiet "$MODEL_DIR/val.txt" \
    | wc -c | awk -v orig="$(wc -c < "$MODEL_DIR/val.txt")" '{printf "%.4f", $1/orig}')

echo "[train] held-out: ${held_out_ratio}  zstd-22: ${zstd_baseline_ratio}"

# Decide codec: fallback to zstd if our model isn't at least 2% better.
codec="zstd_fallback"
if awk -v ours="$held_out_ratio" -v zstd="$zstd_baseline_ratio" \
     'BEGIN { exit !(ours < zstd * 0.98) }'; then
  codec="l3tc"
fi
echo "[train] codec decision: ${codec}"

# Upload artifacts.
echo "[train] uploading artifacts..."
aws s3 cp "$MODEL_DIR/v${VERSION}.bin" \
    "s3://${BUCKET_NAME}/${S3_MODEL_PREFIX}v${VERSION}.bin"
aws s3 cp "$MODEL_DIR/spm.model" \
    "s3://${BUCKET_NAME}/${S3_MODEL_PREFIX}v${VERSION}.tokenizer.model"

cat > "$MODEL_DIR/metadata.json" <<EOF
{
  "version": ${VERSION},
  "corpus_bytes": ${corpus_bytes},
  "codec": "${codec}",
  "held_out_ratio": ${held_out_ratio},
  "zstd_baseline_ratio": ${zstd_baseline_ratio},
  "trigger": "${TRIGGER}",
  "trained_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF
aws s3 cp "$MODEL_DIR/metadata.json" \
    "s3://${BUCKET_NAME}/${S3_MODEL_PREFIX}v${VERSION}.metadata.json"

echo "[train] complete."

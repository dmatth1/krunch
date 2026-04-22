#!/usr/bin/env bash
# Training job entrypoint for AWS Batch.
#
# Spike 1 scope: measure whether a trained per-dataset RWKV-v4 model
# BEATS zstd-22 on held-out data. If yes, we have a product. If no,
# fall back to zstd for actual compression storage.
#
# For Spike 1 we DON'T produce the Rust .bin format (would require
# convert_checkpoint.py + config wrangling + Rust runtime in the
# Fargate worker). Instead we emit:
#   - spm.model      — tokenizer
#   - v{N}.pth       — PyTorch checkpoint (may be used by future spike)
#   - v{N}.metadata.json — has held_out_ratio vs zstd_baseline_ratio
#
# The Fargate compression worker reads codec=zstd_fallback from the
# metadata and compresses with zstd. The entropy-based `held_out_ratio`
# in metadata tells us what the L3TC model WOULD have done.
#
# Env vars (set by the launcher Lambda):
#   CUSTOMER_ID, DATASET_ID, BUCKET_NAME, S3_RAW_PREFIX, S3_MODEL_PREFIX
#   TRIGGER                           "initial" | "retrain"

# Intentionally NOT set -e — we handle failures explicitly so we always
# write a metadata.json and exit with the right code, not mid-pipeline.
set -uo pipefail

: "${CUSTOMER_ID:?required}"
: "${DATASET_ID:?required}"
: "${BUCKET_NAME:?required}"
: "${S3_RAW_PREFIX:?required}"
: "${S3_MODEL_PREFIX:?required}"
: "${TRIGGER:=initial}"

echo "[train] customer=${CUSTOMER_ID} dataset=${DATASET_ID} trigger=${TRIGGER}"

# --- Spike 2 hyperparameter overrides ---
# All have defaults matching Spike 1 so nothing breaks. Each Batch
# experiment sets only the ones it wants to change via
# containerOverrides.environment.
VOCAB_SIZE="${VOCAB_SIZE:-16384}"
NUM_LAYERS="${NUM_LAYERS:-2}"
HIDDEN_SIZE="${HIDDEN_SIZE:-96}"
CONTEXT_LEN="${CONTEXT_LEN:-2048}"
EPOCHS="${EPOCHS:-10}"
EPOCH_LENGTH="${EPOCH_LENGTH:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SAMPLE_MB="${SAMPLE_MB:-200}"  # SPM training sample size; "0" means use full corpus
MAX_PIECE_LENGTH="${MAX_PIECE_LENGTH:-16}"  # SPM max chars per token; Phase C bumps to ~256 for template absorption
NORMALIZE_VARIABLE_FIELDS="${NORMALIZE_VARIABLE_FIELDS:-0}"  # Phase C4: pre-tokenize normalization of log-variable fields (1 to enable)

echo "[train] config: vocab=${VOCAB_SIZE} num_layers=${NUM_LAYERS} ctx=${CONTEXT_LEN} epochs=${EPOCHS} epoch_len=${EPOCH_LENGTH} batch=${BATCH_SIZE} sample_mb=${SAMPLE_MB}"

RAW_DIR=/tmp/raw
MODEL_DIR=/tmp/model
mkdir -p "$RAW_DIR" "$MODEL_DIR"

echo "[train] downloading raw data from s3://${BUCKET_NAME}/${S3_RAW_PREFIX}..."
aws s3 sync "s3://${BUCKET_NAME}/${S3_RAW_PREFIX}" "$RAW_DIR/" --only-show-errors || {
  echo "[train] FATAL: s3 sync failed"
  exit 10
}

ndjson_count=$(find "$RAW_DIR" -name "*.ndjson" | wc -l)
echo "[train] raw files downloaded: ${ndjson_count}"
if [ "$ndjson_count" -eq 0 ]; then
  echo "[train] FATAL: no NDJSON files under raw/ — nothing to train on"
  exit 11
fi

# Concatenate every raw NDJSON into a single corpus.
cat "$RAW_DIR"/*.ndjson > "$MODEL_DIR/corpus.txt"
corpus_bytes=$(wc -c < "$MODEL_DIR/corpus.txt")
echo "[train] corpus bytes: ${corpus_bytes}"

# 80/20 train/val split on byte offset.
train_bytes=$(( corpus_bytes * 80 / 100 ))
head -c "$train_bytes" "$MODEL_DIR/corpus.txt" > "$MODEL_DIR/train.txt"
tail -c +$(( train_bytes + 1 )) "$MODEL_DIR/corpus.txt" > "$MODEL_DIR/val.txt"
echo "[train] train=$(wc -c < "$MODEL_DIR/train.txt")B val=$(wc -c < "$MODEL_DIR/val.txt")B"

# Phase C4b: variable-field normalization — split approach.
#
# Train SPM on NORMALIZED corpus so it learns the template skeleton
# as (relatively few) stable pieces, plus the placeholder tokens
# <TS>/<BLKID>/<IP>/etc. Then tokenize + train the RWKV model on
# the ORIGINAL (non-normalized) corpus. Variable values fall back
# to byte-fallback or short-piece tokens; skeleton matches cleanly.
#
# This is close to how zstd works (template gets a short reference,
# variable bytes are literals) but with a learned model that can
# better predict the structure around variable values.
if [ "${NORMALIZE_VARIABLE_FIELDS}" = "1" ]; then
  echo "[train] normalizing variable fields for SPM training (C4b)…"
  python /app/scripts/normalize_variable_fields.py \
      --corpus "$MODEL_DIR/train.txt" \
      --out    "$MODEL_DIR/train.norm.txt"
  python /app/scripts/normalize_variable_fields.py \
      --corpus "$MODEL_DIR/val.txt" \
      --out    "$MODEL_DIR/val.norm.txt"
  echo "[train] normalized: train=$(wc -c < "$MODEL_DIR/train.norm.txt")B val=$(wc -c < "$MODEL_DIR/val.norm.txt")B"
  # SPM trains on normalized; RWKV trains on original.
  SPM_TRAIN_CORPUS="$MODEL_DIR/train.norm.txt"
else
  SPM_TRAIN_CORPUS="$MODEL_DIR/train.txt"
fi

# Determine next model version by listing existing models in S3.
existing=$(aws s3 ls "s3://${BUCKET_NAME}/${S3_MODEL_PREFIX}" 2>/dev/null \
  | awk '{print $4}' | grep -oE '^v[0-9]+\.pth$' | sed 's/v//;s/\.pth//' | sort -n)
if [ -z "$existing" ]; then
  VERSION=1
else
  VERSION=$(( $(echo "$existing" | tail -1) + 1 ))
fi
echo "[train] assigning model version v${VERSION}"

# ------- Tokenizer -------
# train_specialist_tokenizer.py validates --domain against a fixed enum
# (prose|code|structured|logs|tabular|markup|fallback). For Spike 1 we
# default every dataset to "logs" because HDFS is the pilot corpus; a
# follow-up will plumb a DOMAIN env var through the launcher so each
# customer dataset picks its own specialist.
DOMAIN="${DOMAIN:-logs}"
# SAMPLE_MB=0 ⇒ use the full train corpus (Phase A plan target: train
# SPM on all data, not a 200 MB sample). Non-zero caps the SPM
# trainer's input to that many MB for speed.
if [ "${SAMPLE_MB}" = "0" ]; then
  SAMPLE_MB_ARG=$(( $(wc -c < "$MODEL_DIR/train.txt") / 1024 / 1024 + 1 ))
else
  SAMPLE_MB_ARG="${SAMPLE_MB}"
fi
echo "[train] training SPM tokenizer (vocab=${VOCAB_SIZE} unigram, domain=${DOMAIN}, sample_mb=${SAMPLE_MB_ARG}, max_piece_length=${MAX_PIECE_LENGTH}, spm_corpus=${SPM_TRAIN_CORPUS})..."
python /app/scripts/train_specialist_tokenizer.py \
    --domain "${DOMAIN}" \
    --corpus "$SPM_TRAIN_CORPUS" \
    --output-dir "$MODEL_DIR" \
    --vocab-size "${VOCAB_SIZE}" \
    --max-piece-length "${MAX_PIECE_LENGTH}" \
    --sample-mb "${SAMPLE_MB_ARG}" 2>&1 | tee /tmp/tokenizer.log
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  echo "[train] FATAL: tokenizer training failed (rc=$rc)"
  exit 20
fi

# ------- Tokenize corpus (raw text -> int-per-line) -------
# train_l3tc_phase11.py's L3TCTokenDataset reads one int per line
# (tokenized), not raw text. Use the just-trained SPM to encode
# train.txt + val.txt into the expected format. Mirrors
# phase11_tokenize_corpus in phase11_bootstrap_helpers.sh.
echo "[train] tokenizing corpus with trained SPM..."
python - <<PYEOF
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load("${MODEL_DIR}/spm.model")
# Stream the file in reasonably-sized text chunks so we never hold
# more than a few MB of Python string + SPM buffers at once. Loading
# the whole 1 GB train corpus into a single string blew the 16 GB
# g6.xlarge heap with std::bad_alloc in the first attempt.
CHUNK_BYTES = 2 * 1024 * 1024  # 2 MB per encode call
for src, dst in [("${MODEL_DIR}/train.txt", "${MODEL_DIR}/train.tok.txt"),
                 ("${MODEL_DIR}/val.txt",   "${MODEL_DIR}/val.tok.txt")]:
    total = 0
    with open(src, "r", encoding="utf-8", errors="replace") as fin, \
         open(dst, "w") as fout:
        fout.write("2\n")  # BOS
        buf = []
        buf_len = 0
        for line in fin:
            buf.append(line)
            buf_len += len(line)
            if buf_len >= CHUNK_BYTES:
                ids = sp.encode_as_ids("".join(buf))
                fout.write("\n".join(str(i) for i in ids) + "\n")
                total += len(ids)
                buf = []
                buf_len = 0
        if buf:
            ids = sp.encode_as_ids("".join(buf))
            fout.write("\n".join(str(i) for i in ids) + "\n")
            total += len(ids)
    print(f"  {src} -> {dst} ({total:,} tokens)")
PYEOF
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[train] FATAL: corpus tokenization failed (rc=$rc)"
  exit 21
fi

# ------- Train RWKV -------
echo "[train] training RWKV-v4 model on GPU..."
nvidia-smi 2>&1 | head -5 || echo "[train] WARN: nvidia-smi not available in container"

mkdir -p "$MODEL_DIR/train_out"

# The vendor WKV CUDA kernel uses relative paths
# ("models/RWKV_V4/cuda/wkv_op.cpp") when torch JIT-loads it at import
# time, so the trainer must run with cwd=/app/vendor/L3TC.
# This mirrors phase11_bootstrap_helpers.sh's "cd vendor/L3TC" step.
cd /app/vendor/L3TC
python /app/scripts/train_l3tc_phase11.py \
    --train-file "$MODEL_DIR/train.tok.txt" \
    --val-file "$MODEL_DIR/val.tok.txt" \
    --output-dir "$MODEL_DIR/train_out" \
    --epochs "${EPOCHS}" \
    --epoch-length "${EPOCH_LENGTH}" \
    --batch-size "${BATCH_SIZE}" \
    --num-layers "${NUM_LAYERS}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --ctx-len "${CONTEXT_LEN}" \
    --vocab-size "${VOCAB_SIZE}" \
    --lr 1e-4 \
    --device cuda \
    --no-compile \
    --num-workers 2 2>&1 | tee /tmp/train.log
cd /app
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  echo "[train] FATAL: RWKV training failed (rc=$rc)"
  exit 30
fi

# Training saves `checkpoint_latest.pth` (continuous; no explicit _final).
CHECKPOINT="$MODEL_DIR/train_out/checkpoint_latest.pth"
if [ ! -f "$CHECKPOINT" ]; then
  echo "[train] FATAL: expected checkpoint not found at $CHECKPOINT"
  ls -la "$MODEL_DIR/train_out/" 2>&1 | head
  exit 31
fi

# ------- Ratio measurement -------
echo "[train] measuring held-out ratio (entropy bound)..."
# Same cwd constraint as training: the WKV JIT extension needs
# relative paths resolvable.
cd /app/vendor/L3TC
# Capture rc of the python call specifically — not the cd that
# follows. Previous version captured $? AFTER the cd which silently
# masked every Python failure as rc=0.
held_out_ratio=$(python /app/scripts/measure_held_out_ratio.py \
    --val-file "$MODEL_DIR/val.txt" \
    --checkpoint "$CHECKPOINT" \
    --tokenizer "$MODEL_DIR/spm.model" \
    --num-layers "${NUM_LAYERS}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --ctx-len "${CONTEXT_LEN}" \
    --segment-len "${CONTEXT_LEN}" \
    --vocab-size "${VOCAB_SIZE}" 2>/tmp/measure.stderr)
rc=$?
cd /app
if [ "$rc" -ne 0 ] || [ -z "$held_out_ratio" ]; then
  echo "[train] WARN: ratio measurement failed (rc=$rc) — using sentinel 1.0"
  cat /tmp/measure.stderr 2>/dev/null | tail -20
  held_out_ratio="1.0"
fi
echo "[train] measure.stderr:"
cat /tmp/measure.stderr 2>/dev/null | tail -5 | sed 's/^/  /'

zstd_baseline_ratio=$(zstd --long=27 --ultra -22 --stdout --quiet "$MODEL_DIR/val.txt" \
    | wc -c \
    | awk -v orig="$(wc -c < "$MODEL_DIR/val.txt")" '{printf "%.6f", $1/orig}')

echo "[train] ==============================="
echo "[train]   held_out_ratio:  ${held_out_ratio}"
echo "[train]   zstd-22 ratio:   ${zstd_baseline_ratio}"
echo "[train] ==============================="

# Decide codec. Spike 1 always uses zstd_fallback for actual storage
# (we skip the .bin conversion) but records whether the L3TC model
# *would* have won.
would_have_beaten_zstd="false"
if awk -v ours="$held_out_ratio" -v zstd="$zstd_baseline_ratio" \
     'BEGIN { exit !(ours > 0 && ours < zstd * 0.98) }'; then
  would_have_beaten_zstd="true"
fi
codec="zstd_fallback"  # Spike 1: always use zstd in the worker
echo "[train] codec (storage): ${codec}"
echo "[train] would_have_beaten_zstd: ${would_have_beaten_zstd}"

# ------- Upload artifacts -------
echo "[train] uploading artifacts to s3://${BUCKET_NAME}/${S3_MODEL_PREFIX}..."
aws s3 cp "$CHECKPOINT" \
    "s3://${BUCKET_NAME}/${S3_MODEL_PREFIX}v${VERSION}.pth" --only-show-errors
aws s3 cp "$MODEL_DIR/spm.model" \
    "s3://${BUCKET_NAME}/${S3_MODEL_PREFIX}v${VERSION}.tokenizer.model" --only-show-errors

trained_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Pull bytes/token from the SPM report if present (it writes
# spm.bt_report.json with that exact field).
bytes_per_token=$(python -c "import json; d=json.load(open('${MODEL_DIR}/spm.bt_report.json')); print(d.get('bytes_per_token', d.get('bt', 'null')))" 2>/dev/null || echo "null")
cat > "$MODEL_DIR/metadata.json" <<EOF
{
  "version": ${VERSION},
  "corpus_bytes": ${corpus_bytes},
  "codec": "${codec}",
  "would_have_beaten_zstd": ${would_have_beaten_zstd},
  "held_out_ratio": ${held_out_ratio},
  "zstd_baseline_ratio": ${zstd_baseline_ratio},
  "vocab_size": ${VOCAB_SIZE},
  "num_layers": ${NUM_LAYERS},
  "hidden_size": ${HIDDEN_SIZE},
  "context_len": ${CONTEXT_LEN},
  "max_piece_length": ${MAX_PIECE_LENGTH},
  "normalize_variable_fields": ${NORMALIZE_VARIABLE_FIELDS},
  "epochs": ${EPOCHS},
  "epoch_length": ${EPOCH_LENGTH},
  "batch_size": ${BATCH_SIZE},
  "sample_mb": ${SAMPLE_MB_ARG},
  "bytes_per_token": ${bytes_per_token},
  "trigger": "${TRIGGER}",
  "trained_at": "${trained_at}",
  "spike": "${SPIKE_NAME:-spike_2}"
}
EOF
aws s3 cp "$MODEL_DIR/metadata.json" \
    "s3://${BUCKET_NAME}/${S3_MODEL_PREFIX}v${VERSION}.metadata.json" --only-show-errors

echo "[train] complete. metadata.json:"
cat "$MODEL_DIR/metadata.json"

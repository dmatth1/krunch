#!/bin/bash
# Tier 2 — local CPU end-to-end with the real RWKV-4-Pile-169M model.
#
# Validates the full neural compression path on CPU (no GPU needed).
# This catches every algorithmic bug that would otherwise burn GPU dollars.
# Speed is ~10 KB/s on CPU, so we use a tiny input (~1 KB).
#
# Prerequisites (one-time):
#   python3 -m venv /tmp/krunch-venv
#   /tmp/krunch-venv/bin/pip install constriction tokenizers zstandard numpy boto3
#   /tmp/krunch-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
#   git clone --depth 1 https://github.com/BlinkDL/RWKV-LM.git /tmp/RWKV-LM

set -euo pipefail
cd "$(dirname "$0")/.."

VENV_PY=/tmp/krunch-venv/bin/python
RWKV_LM_PATH=/tmp/RWKV-LM/RWKV-v4/src
MODELS_DIR="$(pwd)/models"

if [[ ! -x $VENV_PY ]]; then
  echo "FAIL /tmp/krunch-venv missing — see prereqs comment in this script"
  exit 1
fi

if [[ ! -d $RWKV_LM_PATH ]]; then
  echo "FAIL /tmp/RWKV-LM missing — see prereqs comment in this script"
  exit 1
fi

if [[ ! -f $MODELS_DIR/RWKV-4-Pile-169M-20220807-8023.pth ]]; then
  echo "FAIL model weights missing at $MODELS_DIR"
  exit 1
fi

echo "=== Tier 2: local CPU end-to-end roundtrip ==="

# 1 KB sample — enough to span 64KB chunk boundary trivially? No, single chunk.
# We use a 200-byte sample for a quick CPU test (model load ~30s + 5-10s compress)
SAMPLE_TXT=$(mktemp /tmp/krunch-sample.XXXXXX.txt)
SAMPLE_KRUNCH=$(mktemp /tmp/krunch-sample.XXXXXX.krunch)
SAMPLE_OUT=$(mktemp /tmp/krunch-sample.XXXXXX.out)

# Generate a deterministic 256-byte text sample (will produce ~50-80 tokens)
cat > "$SAMPLE_TXT" << 'EOF'
The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.
How vexingly quick daft zebras jump. The five boxing wizards jump quickly.
Sphinx of black quartz, judge my vow. A1 B2 C3 D4 E5 F6 G7 H8.
EOF

INPUT_BYTES=$(wc -c < "$SAMPLE_TXT")
echo "Sample: $INPUT_BYTES bytes"

COMPRESS_LOG=$(mktemp)
DECOMPRESS_LOG=$(mktemp)
trap "rm -f $SAMPLE_TXT $SAMPLE_KRUNCH $SAMPLE_OUT $COMPRESS_LOG $DECOMPRESS_LOG" EXIT

# Run compress — stdout = blob, stderr = logs (kept separate)
echo "Running compress (model load + forward pass on CPU, ~30-90s)..."
RWKV_CUDA_ON=0 RWKV_JIT_ON=1 \
  KRUNCH_MODEL_DIR="$MODELS_DIR" \
  PYTHONPATH="$(pwd)" \
  $VENV_PY -m server.cli compress \
  < "$SAMPLE_TXT" > "$SAMPLE_KRUNCH" 2>"$COMPRESS_LOG" || {
    echo "FAIL compress errored"
    cat "$COMPRESS_LOG"
    exit 1
}

COMPRESSED_BYTES=$(wc -c < "$SAMPLE_KRUNCH")
echo "Compressed: $COMPRESSED_BYTES bytes (ratio $(python3 -c "print(f'{$COMPRESSED_BYTES / $INPUT_BYTES:.3f}')"))"

# Run decompress
echo "Running decompress (sequential token-step on CPU, ~30-90s)..."
RWKV_CUDA_ON=0 RWKV_JIT_ON=1 \
  KRUNCH_MODEL_DIR="$MODELS_DIR" \
  PYTHONPATH="$(pwd)" \
  $VENV_PY -m server.cli decompress \
  < "$SAMPLE_KRUNCH" > "$SAMPLE_OUT" 2>"$DECOMPRESS_LOG" || {
    echo "FAIL decompress errored"
    cat "$DECOMPRESS_LOG"
    exit 1
}

OUTPUT_BYTES=$(wc -c < "$SAMPLE_OUT")
echo "Decompressed: $OUTPUT_BYTES bytes"

if cmp -s "$SAMPLE_TXT" "$SAMPLE_OUT"; then
  echo "PASS byte-exact roundtrip ($INPUT_BYTES bytes)"
  exit 0
else
  echo "FAIL byte mismatch:"
  diff <(xxd "$SAMPLE_TXT" | head -5) <(xxd "$SAMPLE_OUT" | head -5)
  exit 1
fi

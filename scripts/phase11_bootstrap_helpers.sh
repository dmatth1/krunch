#!/bin/bash
# Phase 11 bootstrap helpers — sourced from spot-fleet-userdata.sh.
# Lives in the repo (not embedded in EC2 user data) so the user data
# stays under the 16 KB EC2 limit.
#
# After Phase 11.5 (scripts/train_l3tc_phase11.py replacing
# vendor/L3TC/main.py), the dependency surface is much smaller:
# we only need numpy<2, torch 2.1.2+cu121, sentencepiece, and the
# pkuseg + deepspeed stubs (because rwkv_tc_hira_train.py still
# imports them at module load even though our trainer never calls
# them).
#
# Expected env vars set by the caller:
#   S3_BUCKET, S3_PREFIX, S3_RUN_PATH, S3_CORPUS_PATH, RUN_ID, PASS,
#   CORPUS_NAME, CORPUS_S3, TRAIN_FILE, EPOCHS

set -euo pipefail

# === Python venv + dependency installation ===
phase11_install_python_deps() {
    cd /home/ubuntu/l3tc-prod/vendor/L3TC

    # Wipe whatever setup.sh produced -- we don't need any of L3TC's
    # requirements.txt anymore. Our trainer imports only the model
    # class from L3TC and nothing else.
    rm -rf .venv
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip

    # numpy MUST be <2: torch 2.1.2 was compiled against numpy 1.x
    # and crashes on the 2.x ABI break.
    echo "=== installing numpy<2 ==="
    pip install "numpy<2"

    # torch 2.1.2 with CUDA 12.1, bundled CUDA libs (no separate
    # nvidia-* wheels). The bundled wheel is ~2 GB; the unbundled
    # version pulls another ~1 GB of nvidia-cufft/cusolver/cusparse
    # and saturates the network.
    echo "=== installing torch 2.1.2+cu121 ==="
    pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

    # SentencePiece for the BPE tokenizer (used by us at preprocessing
    # time, not at training time). Tiny.
    echo "=== installing sentencepiece ==="
    pip install sentencepiece

    echo "=== python deps install complete ==="
}

# === Stub pkuseg + deepspeed in site-packages ===
# Even after Phase 11.5 these are still required: L3TC's
# rwkv_tc_hira_train.py imports `pkuseg` (vestigial) and
# `from deepspeed.ops.adam import FusedAdam` (real but
# fallback-able) at module top, before our trainer can do
# anything. The stubs let the import succeed.
phase11_write_stubs() {
    local site_pkgs
    site_pkgs=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")

    mkdir -p "${site_pkgs}/pkuseg"
    echo "# stub: vestigial import in L3TC datasets" > "${site_pkgs}/pkuseg/__init__.py"

    mkdir -p "${site_pkgs}/deepspeed/ops/adam"
    echo "# stub" > "${site_pkgs}/deepspeed/__init__.py"
    echo "" > "${site_pkgs}/deepspeed/ops/__init__.py"
    cat > "${site_pkgs}/deepspeed/ops/adam/__init__.py" <<'PYEOF'
class FusedAdam:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("deepspeed stub: caller should fall back")
PYEOF
    echo "stubs written to ${site_pkgs}/{pkuseg,deepspeed}/"
}

# === GPU sanity check ===
phase11_check_gpu() {
    nvidia-smi 2>&1 | head -20 || echo "nvidia-smi failed"
    python -c "
import torch
print(f'torch version: {torch.__version__}')
print(f'cuda available: {torch.cuda.is_available()}')
print(f'cuda device count: {torch.cuda.device_count()}')
assert torch.cuda.is_available(), 'CUDA not available'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'bf16 supported: {torch.cuda.is_bf16_supported()}')
"
}

# === Pull SPM tokenizer files from S3 (not in L3TC repo) ===
phase11_pull_spm_tokenizer() {
    mkdir -p dictionary
    aws s3 cp "${S3_CORPUS_PATH}/spm_enwik8.tar.gz" /tmp/spm.tar.gz --quiet
    tar -xzf /tmp/spm.tar.gz -C dictionary/
    ls -lh dictionary/vocab_enwik8_bpe_16384_0.999/
}

# === Fetch corpus, preferring pre-tokenized file if present ===
# Sets SKIP_TOKENIZE=1 if the pre-tokenized file was found.
phase11_fetch_corpus_pass1() {
    local pretok_s3="${S3_CORPUS_PATH}/train_enwik9_bpe_16384_0.999.txt"
    if aws s3 ls "${pretok_s3}" >/dev/null 2>&1; then
        echo "=== Using pre-tokenized enwik9 from S3 ==="
        mkdir -p data/train_data
        aws s3 cp "${pretok_s3}" "${TRAIN_FILE}" --quiet
        ls -lh "${TRAIN_FILE}"
        export SKIP_TOKENIZE=1
    else
        echo "=== Downloading raw enwik9.xz and tokenizing ==="
        if ! aws s3 ls "${CORPUS_S3}" >/dev/null 2>&1; then
            echo "ERROR: enwik9 not found at ${CORPUS_S3}"
            exit 1
        fi
        mkdir -p data/raw_text_data
        aws s3 cp "${CORPUS_S3}" /tmp/enwik9.xz --quiet
        xz -d /tmp/enwik9.xz
        mv /tmp/enwik9 data/raw_text_data/enwik9
        ls -lh data/raw_text_data/enwik9
        export SKIP_TOKENIZE=0
    fi
}

# === Tokenize raw text → one-int-per-line format ===
# Only called if SKIP_TOKENIZE=0 (i.e., pre-tokenized file not in S3).
phase11_tokenize_corpus() {
    local raw_text="$1"
    local out_file="$2"
    local spm_model="dictionary/vocab_enwik8_bpe_16384_0.999/spm_enwik8_bpe_16384_0.999.model"

    mkdir -p "$(dirname "${out_file}")"
    echo "tokenizing ${raw_text} -> ${out_file}"
    python <<PYEOF
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load('${spm_model}')
print(f'vocab={sp.vocab_size()} bos={sp.bos_id()}')
with open('${raw_text}', 'r', encoding='utf-8', errors='replace') as f_in, \
     open('${out_file}', 'w') as f_out:
    text = f_in.read()
    ids = sp.encode_as_ids(text)
    f_out.write('2\n')
    for tid in ids:
        f_out.write(f'{tid}\n')
print(f'done: {len(ids)} tokens')
PYEOF
    ls -lh "${out_file}"
}

# === Carve a small held-out validation slice from the train file ===
# Takes the last `n_lines` lines as the val set. Idempotent.
phase11_make_val_split() {
    local train_file="$1"
    local val_file="$2"
    local n_lines="${3:-100000}"  # ~10 MB at our token format

    mkdir -p "$(dirname "${val_file}")"
    if [ -f "${val_file}" ]; then
        echo "val file already exists: ${val_file}"
        return 0
    fi
    tail -n "${n_lines}" "${train_file}" > "${val_file}"
    # Prepend the BOS marker so the val set starts cleanly
    echo "val file: ${val_file} ($(wc -l < ${val_file}) lines)"
}

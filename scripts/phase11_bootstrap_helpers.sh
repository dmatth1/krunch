#!/bin/bash
# Phase 11 bootstrap helpers — sourced/run from spot-fleet-userdata.sh.
# Lives in the repo (not embedded in EC2 user data) so the user data
# stays under the 16 KB EC2 limit.
#
# Expected env vars set by the caller:
#   S3_BUCKET, S3_PREFIX, S3_RUN_PATH, S3_CORPUS_PATH, RUN_ID, PASS,
#   CORPUS_NAME, CORPUS_S3, TRAIN_FILE, EPOCHS

set -euo pipefail

# === Python venv + dependency installation ===
# L3TC's pinned requirements.txt has issues we work around:
#   - pkuseg: setup.py imports numpy before build_requires installs it,
#             AND L3TC's three dataset files import pkuseg but never
#             call it. Stub it.
#   - openpyxl: not used at runtime, drop.
#   - transformers: pulls a CPU torch wheel that overwrites our CUDA
#             torch. Drop and install torch CUDA last.
#   - deepspeed (NOT in requirements.txt): every model in
#             models/RWKV_V4/*train.py imports `from deepspeed.ops.adam
#             import FusedAdam` at module top, but configure_optimizers
#             wraps the call in a try/except and falls back to
#             torch.optim.Adam. Stub the class with one that raises on
#             init -- the except catches it and the fallback runs.
#   - termcolor, scipy, ninja (NOT in requirements.txt but actually
#             needed): install via pip directly.
#
# Order matters: trimmed L3TC reqs first (these may install a CPU torch
# from transformers chain → we removed transformers anyway), then
# scipy/termcolor/ninja, then force-reinstall CUDA torch last so
# nothing can overwrite it.
phase11_install_python_deps() {
    # Hardcoded path: the userdata always clones the repo to
    # /home/ubuntu/l3tc-prod and setup.sh puts vendor/L3TC inside it.
    # `$(dirname "$0")` doesn't work when this file is sourced because
    # $0 reflects the parent script, not this file.
    cd /home/ubuntu/l3tc-prod/vendor/L3TC

    # setup.sh already created vendor/L3TC/.venv and ran the broken
    # `pip install -r requirements.txt` (which fails on pkuseg but
    # successfully installs everything else AND a CPU torch wheel
    # from the transformers dependency chain). Wipe it and start
    # fresh so we don't fight pip's resolver against the leftover
    # CPU torch + transformers state.
    rm -rf .venv
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install numpy
    grep -vE '^(pkuseg|openpyxl|transformers)\b' requirements.txt > /tmp/l3tc-req-trimmed.txt
    echo "=== trimmed L3TC requirements ==="
    cat /tmp/l3tc-req-trimmed.txt
    echo "=== installing trimmed L3TC requirements ==="
    pip install -r /tmp/l3tc-req-trimmed.txt
    echo "=== installing scipy termcolor ninja ==="
    pip install scipy termcolor ninja
    echo "=== force-reinstall torch CUDA (cu121) ==="
    pip install --force-reinstall --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
    echo "=== python deps install complete ==="
}

# === Stub pkuseg + deepspeed in site-packages ===
phase11_write_stubs() {
    local site_pkgs
    site_pkgs=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")

    mkdir -p "${site_pkgs}/pkuseg"
    echo "# stub: vestigial import in L3TC datasets" > "${site_pkgs}/pkuseg/__init__.py"

    mkdir -p "${site_pkgs}/deepspeed/ops/adam"
    echo "# stub" > "${site_pkgs}/deepspeed/__init__.py"
    echo "" > "${site_pkgs}/deepspeed/ops/__init__.py"
    cat > "${site_pkgs}/deepspeed/ops/adam/__init__.py" <<'PYEOF'
# Stub: real deepspeed not installed. L3TC's configure_optimizers
# wraps the FusedAdam call in try/except and falls back to
# torch.optim.Adam if it raises.
class FusedAdam:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "deepspeed stub: FusedAdam not available; "
            "L3TC will fall back to torch.optim.Adam"
        )
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
"
}

# === Tokenize raw text → L3TC format ===
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
print(f'vocab size: {sp.vocab_size()}, BOS id: {sp.bos_id()}')
with open('${raw_text}', 'r', encoding='utf-8', errors='replace') as f_in, \
     open('${out_file}', 'w') as f_out:
    bytes_done = 0
    while True:
        chunk = f_in.read(1_000_000)
        if not chunk:
            break
        ids = sp.encode_as_ids(chunk)
        f_out.write('2\n')
        for tid in ids:
            f_out.write(f'{tid}\n')
        bytes_done += len(chunk.encode('utf-8'))
        if bytes_done % (50 * 1024 * 1024) == 0:
            print(f'  tokenized {bytes_done // (1024*1024)} MB', flush=True)
print(f'done: ${out_file}')
PYEOF
    ls -lh "${out_file}"
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

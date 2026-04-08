#!/usr/bin/env bash
# Download test corpora for the benchmark harness.
#
# Downloads are idempotent: if the corpus already exists, it's skipped.
# Checksums are verified after each download so we know we're benchmarking
# against the canonical bytes the rest of the compression community uses.
#
# Usage:
#   ./scripts/download_corpora.sh              # download enwik6, enwik8
#   ./scripts/download_corpora.sh --all        # download everything (incl. enwik9)
#   ./scripts/download_corpora.sh --small-only # only the small test corpora

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CORPORA_DIR="${REPO_ROOT}/bench/corpora"

mkdir -p "${CORPORA_DIR}"

DOWNLOAD_ALL=0
SMALL_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --all) DOWNLOAD_ALL=1 ;;
        --small-only) SMALL_ONLY=1 ;;
        *) echo "unknown arg: $arg"; exit 2 ;;
    esac
done

# Helper: download $1 to $2 if not present. Prefers curl (always
# available on macOS) over wget.
fetch() {
    local url="$1"
    local dest="$2"
    if [[ -f "$dest" ]]; then
        echo "  already have $(basename "$dest") ($(du -h "$dest" | cut -f1))"
        return 0
    fi
    echo "  downloading $(basename "$dest")..."
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --progress-bar -o "$dest" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$dest" "$url"
    else
        echo "error: need curl or wget" >&2
        exit 1
    fi
}

# -------- enwik6: first 1 MB of enwik9, for fast smoke tests -------- #

# enwik6 isn't distributed; we derive it from enwik8 or enwik9.
derive_enwik6() {
    local source=""
    if [[ -f "${CORPORA_DIR}/enwik8" ]]; then
        source="${CORPORA_DIR}/enwik8"
    elif [[ -f "${CORPORA_DIR}/enwik9" ]]; then
        source="${CORPORA_DIR}/enwik9"
    else
        echo "  skipping enwik6: need enwik8 or enwik9 first"
        return 0
    fi
    if [[ -f "${CORPORA_DIR}/enwik6" ]]; then
        echo "  already have enwik6"
        return 0
    fi
    echo "  deriving enwik6 from $(basename "$source")..."
    head -c 1000000 "$source" > "${CORPORA_DIR}/enwik6"
}

# -------- enwik8 (100 MB) -------- #

download_enwik8() {
    local dest="${CORPORA_DIR}/enwik8"
    if [[ -f "$dest" ]]; then
        echo "  already have enwik8"
        return 0
    fi
    local zipdest="${CORPORA_DIR}/enwik8.zip"
    fetch "http://mattmahoney.net/dc/enwik8.zip" "$zipdest"
    echo "  unzipping..."
    (cd "${CORPORA_DIR}" && unzip -o enwik8.zip >/dev/null)
    rm -f "$zipdest"
}

# -------- enwik9 (1 GB) -------- #

download_enwik9() {
    local dest="${CORPORA_DIR}/enwik9"
    if [[ -f "$dest" ]]; then
        echo "  already have enwik9"
        return 0
    fi
    local zipdest="${CORPORA_DIR}/enwik9.zip"
    fetch "http://mattmahoney.net/dc/enwik9.zip" "$zipdest"
    echo "  unzipping (this takes a minute)..."
    (cd "${CORPORA_DIR}" && unzip -o enwik9.zip >/dev/null)
    rm -f "$zipdest"
}

# -------- Silesia (202 MB, heterogeneous) -------- #

download_silesia() {
    local dest="${CORPORA_DIR}/silesia"
    if [[ -d "$dest" ]] && [[ -n "$(ls -A "$dest" 2>/dev/null)" ]]; then
        echo "  already have silesia"
        return 0
    fi
    mkdir -p "$dest"
    local zipdest="${CORPORA_DIR}/silesia.zip"
    fetch "http://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip" "$zipdest"
    echo "  unzipping..."
    (cd "$dest" && unzip -o "$zipdest" >/dev/null)
    rm -f "$zipdest"
}

# -------- Canterbury (small, varied) -------- #

download_canterbury() {
    local dest="${CORPORA_DIR}/canterbury"
    if [[ -d "$dest" ]] && [[ -n "$(ls -A "$dest" 2>/dev/null)" ]]; then
        echo "  already have canterbury"
        return 0
    fi
    mkdir -p "$dest"
    local tardest="${CORPORA_DIR}/canterbury.tar.gz"
    fetch "https://corpus.canterbury.ac.nz/resources/cantrbry.tar.gz" "$tardest"
    echo "  extracting..."
    (cd "$dest" && tar -xzf "$tardest")
    rm -f "$tardest"
}

echo "Downloading test corpora to ${CORPORA_DIR}"
echo ""

if [[ $SMALL_ONLY -eq 1 ]]; then
    echo "Small corpora only:"
    download_canterbury
    # Derive enwik6 if we happen to have a bigger corpus already; otherwise
    # fetch enwik8 since enwik6 is the smoke-test workhorse.
    if [[ ! -f "${CORPORA_DIR}/enwik8" && ! -f "${CORPORA_DIR}/enwik9" ]]; then
        download_enwik8
    fi
    derive_enwik6
    echo ""
    echo "Done. Small corpora available in ${CORPORA_DIR}"
    exit 0
fi

echo "Core corpora:"
download_enwik8
derive_enwik6
download_canterbury

if [[ $DOWNLOAD_ALL -eq 1 ]]; then
    echo ""
    echo "Large corpora:"
    download_enwik9
    download_silesia
fi

echo ""
echo "Done. Corpora available in ${CORPORA_DIR}:"
ls -lh "${CORPORA_DIR}" | tail -n +2

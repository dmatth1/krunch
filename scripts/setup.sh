#!/usr/bin/env bash
# Set up the L3TC reference implementation and its Python environment.
#
# What this does:
#   1. Clones L3TC from GitHub into vendor/L3TC/
#   2. Clones BlinkDL/RWKV-LM into vendor/RWKV-LM/ (for reference)
#   3. Creates a Python venv inside vendor/L3TC/.venv
#   4. Installs PyTorch, tiktoken, SentencePiece, and L3TC's other deps
#   5. Writes vendor/L3TC/wrapper.py — a CLI shim that bench/compressors.py
#      can shell out to
#
# What this does NOT do:
#   - Download the pretrained checkpoint (requires a Google Drive link
#     and may need a browser). You have to do that manually. The script
#     prints instructions at the end.
#   - Run any benchmarks. Use bench/bench.py for that after setup.
#
# Usage:
#   ./scripts/setup.sh
#   ./scripts/setup.sh --clean   # remove existing venv first

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENDOR_DIR="${REPO_ROOT}/vendor"

CLEAN=0
for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN=1 ;;
        *) echo "unknown arg: $arg"; exit 2 ;;
    esac
done

mkdir -p "${VENDOR_DIR}"
cd "${VENDOR_DIR}"

echo "=== Cloning L3TC reference implementation ==="
if [[ ! -d L3TC ]]; then
    git clone --depth 1 \
        https://github.com/alipay/L3TC-leveraging-rwkv-for-learned-lossless-low-complexity-text-compression.git \
        L3TC
else
    echo "  L3TC already cloned; skipping. Use 'rm -rf vendor/L3TC' to force re-clone."
fi

echo ""
echo "=== Cloning BlinkDL/RWKV-LM (reference) ==="
if [[ ! -d RWKV-LM ]]; then
    git clone --depth 1 https://github.com/BlinkDL/RWKV-LM.git RWKV-LM
else
    echo "  RWKV-LM already cloned; skipping."
fi

echo ""
echo "=== Setting up Python venv ==="
VENV_DIR="${VENDOR_DIR}/L3TC/.venv"
if [[ $CLEAN -eq 1 && -d "$VENV_DIR" ]]; then
    echo "  --clean specified; removing existing venv"
    rm -rf "$VENV_DIR"
fi

# Pick a Python interpreter. L3TC's PyTorch dependency doesn't yet support
# Python 3.14, so we prefer 3.12 or 3.11 if available.
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done
if [[ -z "$PYTHON" ]]; then
    echo "error: no python3 found on PATH" >&2
    exit 1
fi

PY_VERSION="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "  using $PYTHON (Python $PY_VERSION)"
if [[ "$PY_VERSION" == "3.14" || "$PY_VERSION" == "3.13" ]]; then
    echo "  WARNING: PyTorch wheels for $PY_VERSION may not exist yet."
    echo "  If pip install fails, install Python 3.12 (pyenv install 3.12, or brew install python@3.12)"
fi

if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip wheel setuptools

echo ""
echo "=== Installing L3TC dependencies ==="
# Install PyTorch first because it's the biggest and most finicky.
# Use the CPU wheel by default; users who want CUDA can re-run with
# TORCH_CUDA=cu121 (or whichever).
TORCH_CUDA="${TORCH_CUDA:-cpu}"
if [[ "$TORCH_CUDA" == "cpu" ]]; then
    pip install --index-url https://download.pytorch.org/whl/cpu torch
else
    pip install --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" torch
fi

# L3TC's upstream requirements. We install a curated list because
# upstream requirements.txt may pin versions incompatible with our
# Python. Adjust here if upstream changes.
pip install \
    sentencepiece \
    numpy \
    einops \
    "transformers>=4.30" \
    tqdm \
    pyyaml

# Try upstream requirements.txt if it exists, but don't fail the whole
# setup if it has incompatibilities — the curated list above covers
# the essentials.
if [[ -f "${VENDOR_DIR}/L3TC/requirements.txt" ]]; then
    echo "  trying upstream requirements.txt..."
    pip install -r "${VENDOR_DIR}/L3TC/requirements.txt" || \
        echo "  upstream requirements.txt had issues; continuing with curated list"
fi

echo ""
echo "=== Writing wrapper.py ==="
# This is the shim bench/compressors.py shells out to. We write it here
# during setup so it's always in sync with where the venv lives.
cat > "${VENDOR_DIR}/L3TC/wrapper.py" << 'PYEOF'
#!/usr/bin/env python3
"""L3TC wrapper CLI — bridges bench/compressors.py to the L3TC codebase.

This script lives inside vendor/L3TC/ so it can import L3TC's modules
directly. The harness shells out to it using the venv's Python.

Subcommands:
  compress    --variant <NAME> --batch-size <N> --device <cpu|cuda> \
              --input <PATH> --output <PATH>
  decompress  (same)

The wrapper is intentionally thin: it loads the pretrained model, runs
a single file through L3TC's compress or decompress path, and exits.
All measurement is done by the harness (subprocess wall time + rusage).

STATUS: Phase 0 stub. The actual compress/decompress implementation is
filled in once we confirm L3TC runs end-to-end with the chosen
checkpoint. For now it prints a helpful error.
"""
import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="l3tc-wrapper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for cmd in ("compress", "decompress"):
        p = sub.add_parser(cmd)
        p.add_argument("--variant", default="3.2M")
        p.add_argument("--batch-size", type=int, default=1)
        p.add_argument("--device", default="cpu")
        p.add_argument("--input", required=True, type=Path)
        p.add_argument("--output", required=True, type=Path)

    args = parser.parse_args()

    print(
        "error: L3TC wrapper is a Phase 0 stub. Implementation pending once "
        "we've verified the L3TC reference runs on its test corpus. Until "
        "then, --classical-only measurements are the useful path.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
PYEOF
chmod +x "${VENDOR_DIR}/L3TC/wrapper.py"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo ""
echo "  1. Download the L3TC pretrained checkpoint from their Google Drive"
echo "     link in vendor/L3TC/README.md and save it to"
echo "     vendor/L3TC/checkpoints/checkpoint0019.pth (or similar)."
echo ""
echo "  2. Finish the Phase 0 stub in vendor/L3TC/wrapper.py — import"
echo "     L3TC's compress and decompress functions and call them with"
echo "     the CLI args. This needs to happen once we've reproduced the"
echo "     paper's enwik9 numbers with their native scripts."
echo ""
echo "  3. Run the benchmark harness:"
echo "       python3 bench/bench.py --list            # confirm availability"
echo "       python3 bench/bench.py --classical-only --corpus bench/corpora/enwik6"
echo ""
echo "Classical-only benchmarks (gzip, bzip2, xz, zstd) work without any"
echo "of the above; they use the system binaries that are already installed."

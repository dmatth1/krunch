#!/bin/bash
# Krunch installer — downloads the CLI wrapper and pre-pulls the Docker image.
# After this script, `krunch compress` is ready to use immediately.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/dmatth1/krunch/main/install.sh | sudo bash
#
# Environment overrides:
#   KRUNCH_VERSION  — git ref to install from (default: main)
#   KRUNCH_IMAGE    — Docker image to pre-pull (default: ghcr.io/dmatth1/krunch:latest)
#   INSTALL_DIR     — where to put the krunch wrapper (default: /usr/local/bin)

set -euo pipefail

KRUNCH_VERSION="${KRUNCH_VERSION:-main}"
KRUNCH_IMAGE="${KRUNCH_IMAGE:-ghcr.io/dmatth1/krunch:latest}"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"
# Override where the CLI wrapper is fetched from. Default is the public
# GitHub raw URL; testing/local builds can point at any HTTP URL or a
# local file path (file:///path/to/krunch).
KRUNCH_WRAPPER_URL="${KRUNCH_WRAPPER_URL:-https://raw.githubusercontent.com/dmatth1/krunch/${KRUNCH_VERSION}/scripts/krunch}"

echo "=== Krunch installer ==="
echo "  CLI:   ${INSTALL_DIR}/krunch"
echo "  Image: ${KRUNCH_IMAGE}"
echo

# 1. Pre-flight
command -v curl   >/dev/null || { echo "Error: curl not found";   exit 1; }
command -v docker >/dev/null || { echo "Error: docker not found"; exit 1; }

# 2. Install the CLI wrapper (curl for http(s)://, cp for file:///)
echo "[1/2] Installing krunch CLI from ${KRUNCH_WRAPPER_URL}..."
case "$KRUNCH_WRAPPER_URL" in
  file://*) cp "${KRUNCH_WRAPPER_URL#file://}" "${INSTALL_DIR}/krunch" ;;
  *)        curl -fsSL "$KRUNCH_WRAPPER_URL" -o "${INSTALL_DIR}/krunch" ;;
esac
chmod +x "${INSTALL_DIR}/krunch"
echo "      ✓ ${INSTALL_DIR}/krunch"

# 3. Pre-pull the Docker image (~3.5 GB)
echo
echo "[2/3] Pulling ${KRUNCH_IMAGE} (~3.5 GB, ~5-10 min on typical broadband)..."
docker pull "${KRUNCH_IMAGE}"
echo "      ✓ image cached"

# 4. Warm up: build the WKV CUDA kernel + torch.compile graph trace into a
#    persistent docker volume, so the first real `krunch compress` doesn't
#    pay ~60s of one-time costs. Skipped if no NVIDIA GPU is detected (the
#    user can re-run `krunch warmup` later from a GPU host).
echo
echo "[3/3] Warming up local cache (one-time, ~60-90 s)..."
if docker info --format '{{.Runtimes}}' 2>/dev/null | grep -q nvidia; then
  if "${INSTALL_DIR}/krunch" warmup; then
    echo "      ✓ cache primed"
  else
    echo "      ⚠ warmup failed — first compress will pay the kernel-build cost"
  fi
else
  echo "      ⚠ no NVIDIA runtime detected — skipping warmup"
  echo "        run \`krunch warmup\` later from a GPU host"
fi

echo
echo "=== Ready ==="
echo
echo "  krunch compress   < input  > output"
echo "  krunch decompress < input  > output"
echo
echo "For distributed jobs (AWS Batch): krunch submit --help"

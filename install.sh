#!/bin/bash
# Krunch installer — downloads the CLI wrapper and pre-pulls the Docker image.
# After this script, `krunch compress` is ready to use immediately.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/dmatth1/krunch/main/install.sh | sudo bash
#
# Environment overrides:
#   KRUNCH_VERSION  — git ref to install from (default: main)
#   KRUNCH_IMAGE    — Docker image to pre-pull (default: ghcr.io/dmatth1/krunch:v1)
#   INSTALL_DIR     — where to put the krunch wrapper (default: /usr/local/bin)

set -euo pipefail

KRUNCH_VERSION="${KRUNCH_VERSION:-main}"
KRUNCH_IMAGE="${KRUNCH_IMAGE:-ghcr.io/dmatth1/krunch:v1}"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"

echo "=== Krunch installer ==="
echo "  CLI:   ${INSTALL_DIR}/krunch"
echo "  Image: ${KRUNCH_IMAGE}"
echo

# 1. Pre-flight
command -v curl   >/dev/null || { echo "Error: curl not found";   exit 1; }
command -v docker >/dev/null || { echo "Error: docker not found"; exit 1; }

# 2. Install the CLI wrapper
echo "[1/2] Installing krunch CLI..."
curl -fsSL "https://raw.githubusercontent.com/dmatth1/krunch/${KRUNCH_VERSION}/scripts/krunch" \
  -o "${INSTALL_DIR}/krunch"
chmod +x "${INSTALL_DIR}/krunch"
echo "      ✓ ${INSTALL_DIR}/krunch"

# 3. Pre-pull the Docker image (~3.5 GB)
echo
echo "[2/2] Pulling ${KRUNCH_IMAGE} (~3.5 GB, ~5-10 min on typical broadband)..."
docker pull "${KRUNCH_IMAGE}"
echo "      ✓ image cached"

echo
echo "=== Ready ==="
echo
echo "  krunch compress   < input  > output"
echo "  krunch decompress < input  > output"
echo
echo "For distributed jobs (AWS Batch): krunch submit --help"

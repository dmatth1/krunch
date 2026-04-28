#!/bin/bash
# Krunch container entrypoint.
#
# Modes:
#   compress / decompress   — single-shot CLI; reads stdin or --in, writes stdout or --out
#   job                     — Batch job (KRUNCH_JOB_TYPE env var = compress | assemble)

set -euo pipefail

MODE="${1:-}"

case "$MODE" in
  compress|decompress)
    exec python3 -m server.cli "$@"
    ;;
  job)
    exec python3 -c "from server.job import run; run()"
    ;;
  "")
    cat << 'USAGE' >&2
Krunch — neural compression. Usage:

  Single-shot:
    docker run --gpus all -i ghcr.io/dmatth1/krunch:v1 compress   < input > output
    docker run --gpus all -i ghcr.io/dmatth1/krunch:v1 decompress < input > output

  Batch job (run by AWS Batch / k8s / etc; uses env vars):
    docker run --gpus all -e KRUNCH_JOB_TYPE=compress ... ghcr.io/dmatth1/krunch:v1 job

USAGE
    exit 1
    ;;
  *)
    echo "Unknown mode: $MODE. Use 'compress', 'decompress', or 'job'." >&2
    exit 1
    ;;
esac

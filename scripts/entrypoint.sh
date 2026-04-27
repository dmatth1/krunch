#!/bin/bash
# Krunch container entrypoint.
# Usage:
#   server           — start FastAPI server (default)
#   job              — run a Batch job (KRUNCH_JOB_TYPE=compress|assemble)

set -euo pipefail

MODE="${1:-server}"

case "$MODE" in
  server)
    exec python3 -m uvicorn server.main:app \
      --host 0.0.0.0 --port "${PORT:-8080}" --workers 1
    ;;
  job)
    exec python3 -c "from server.job import run; run()"
    ;;
  *)
    echo "Unknown mode: $MODE. Use 'server' or 'job'." >&2
    exit 1
    ;;
esac

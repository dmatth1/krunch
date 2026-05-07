#!/bin/bash
# Run the GPU pytest suite with true process isolation per test file.
#
# Why not pytest-forked or pytest-xdist: both use fork(), which
# corrupts CUDA context in the child process (CUDA driver state
# doesn't survive fork after init). This wrapper instead invokes a
# fresh `python -m pytest` per file via exec, which gives clean
# per-file process isolation without any fork-after-init issues.
#
# Cost: each file pays ~30-60s for fresh model load + WKV kernel
# JIT (mitigated on subsequent runs by torch_extensions build cache).
# Total wall ~3-5 min on T4 vs ~80s for an unisolated run.
#
# Usage (inside the published Docker image):
#   docker run --rm --gpus all -v $PWD/tests/unit/gpu:/gpu-tests \
#     --entrypoint bash ghcr.io/dmatth1/krunch:latest \
#     -c "pip install -q pytest && bash /gpu-tests/run-gpu-tests.sh"

set -uo pipefail

GPU_TEST_DIR="${1:-$(dirname "$0")/unit/gpu}"
cd "$GPU_TEST_DIR"

echo "=== GPU test suite (per-file isolation) ==="
echo "  test dir: $(pwd)"
echo

PASS=0
FAIL=0
SKIP=0
FAIL_FILES=()
for f in test_*.py; do
    out=$(python -m pytest -q --tb=line --no-header "$f" 2>&1 | tail -1)
    echo ">>> $f  — $out"
    rc=$?
    if echo "$out" | grep -qE "failed|error"; then
        FAIL=$((FAIL + 1))
        FAIL_FILES+=("$f")
    elif echo "$out" | grep -qE "no tests ran|all skipped"; then
        SKIP=$((SKIP + 1))
    else
        PASS=$((PASS + 1))
    fi
done

echo
echo "=== Summary: $PASS passed, $FAIL failed, $SKIP all-skipped (files) ==="
if [[ $FAIL -gt 0 ]]; then
    echo "Failed files:"
    printf '  %s\n' "${FAIL_FILES[@]}"
fi
exit $FAIL

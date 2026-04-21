#!/bin/bash
# Phase 14 task 12 — launch 7 specialist training runs on spot fleet.
#
# Each specialist: 2L x 96H x 16K vocab on g5.xlarge (A10G 24 GB), same
# shape as L3TC-200K. Config matches Phase 11 recipe (AdamW, cosine
# warmup 500, bf16, batch 32). Only the --train-file and --tokenizer-s3
# differ per domain.
#
# Epoch length: 500K samples * batch 32 * ctx 2048 = ~33B tokens per
# epoch. Phase 11 saw the loss asymptote by epoch 5-7; 10 epochs gives
# headroom for narrower per-domain corpora without wasting spot budget.
#
# Cost estimate: 10 epochs x ~42 min = ~7 hr/specialist on g5.xlarge
# spot ($0.30-0.40/hr) = ~$2.50-3.00/specialist x 7 = ~$18-21 total.
# On-demand fallback: $7/specialist x 7 = ~$49. Both under the
# Phase 14 $70-100 budget.
#
# Prereqs (all must be on S3 before running):
#   s3://dmatth1-bnn-checkpoints/l3tc/specialists/{domain}/corpus.npy
#   s3://dmatth1-bnn-checkpoints/l3tc/specialists/{domain}/corpus.txt
#   s3://dmatth1-bnn-checkpoints/l3tc/specialists/{domain}/val.npy
#   s3://dmatth1-bnn-checkpoints/l3tc/specialists/{domain}/spm.model
#
# These come from: task 6 (corpus split) + task 8 (per-domain
# re-tokenize to .npy) + task 11 (per-domain SPM).
#
# Usage:
#   export L3TC_GITHUB_PAT=$(cat ~/.l3tc-pat)
#   ./scripts/launch_all_specialists.sh                  # all 7, spot
#   ./scripts/launch_all_specialists.sh prose code       # subset
#   LAUNCH_MODE=ondemand ./scripts/launch_all_specialists.sh prose

set -euo pipefail

S3_SPEC_BASE="s3://dmatth1-bnn-checkpoints/l3tc/specialists"
DOMAINS_ALL="prose code structured logs tabular markup fallback"
DOMAINS="${*:-$DOMAINS_ALL}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAUNCH="$SCRIPT_DIR/launch-spot-train.sh"

if [ -z "${L3TC_GITHUB_PAT:-}" ]; then
    echo "ERROR: export L3TC_GITHUB_PAT=\$(cat ~/.l3tc-pat)"
    exit 1
fi

for domain in $DOMAINS; do
    case "$domain" in
        prose|code|structured|logs|tabular|markup|fallback) ;;
        *) echo "unknown domain: $domain"; exit 1;;
    esac
done

echo "=== Phase 14 specialist launcher ==="
echo "domains: $DOMAINS"
echo "mode:    ${LAUNCH_MODE:-spot}"
echo ""

# Train args common to all specialists.
# --epochs 10: Phase 11 showed loss asymptote by epoch 5-7 on 51 GB;
#   narrower per-domain corpora should converge faster — pad 10 for safety.
# --epoch-length 500000: matches Phase 11 (~33B tokens/epoch at batch 32
#   ctx 2048). For smaller domain corpora (logs, tabular ~1 GB) this
#   represents ~15-30 passes over the data, which is fine.
# --batch-size 32: matches L3TC-200K reference; g5.xlarge A10G fits easily.
# --grad-accum 1: no accumulation needed; speed-first.
# --num-layers 2: Phase 14 spec (shape parity with L3TC-200K).
# --vocab-size 16384: Phase 14 spec (speed enabler).
# --lr 1e-4: matches Phase 11 recipe; cosine warmup+decay handled in train script.
COMMON_ARGS="--epochs 10 --epoch-length 500000 --batch-size 32 \
    --grad-accum 1 --num-layers 2 --vocab-size 16384 --lr 1e-4"

for domain in $DOMAINS; do
    RUN_ID="phase14_${domain}_$(date +%Y%m%d_%H%M%S)"
    TRAIN_FILE="${S3_SPEC_BASE}/${domain}/corpus.txt"
    TOK_S3="${S3_SPEC_BASE}/${domain}/spm.model"

    # Pre-flight: verify the inputs exist.
    for path in "$TRAIN_FILE" "$TOK_S3"; do
        if ! aws s3 ls "$path" >/dev/null 2>&1; then
            echo "ERROR: missing S3 artifact for ${domain}: ${path}"
            echo "  tasks 6, 8, or 11 not complete for this domain yet."
            exit 2
        fi
    done

    echo "--- launching ${domain} (${RUN_ID}) ---"
    "$LAUNCH" "$RUN_ID" \
        --train-file "$TRAIN_FILE" \
        --tokenizer-s3 "$TOK_S3" \
        $COMMON_ARGS
    echo ""
done

echo ""
echo "All specialist launches submitted. Track via:"
echo "  for d in $DOMAINS; do"
echo "    aws s3 ls ${S3_SPEC_BASE%/*}/phase14_\${d}_*/train.log 2>/dev/null | tail -1"
echo "  done"

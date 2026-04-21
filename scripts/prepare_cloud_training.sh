#!/bin/bash
# Phase 14 cloud-training prep — run once before launch_all_specialists.sh.
#
# Steps:
#   1. Concat per-domain text files → corpus.txt + val.txt (95/5 split)
#   2. Upload corpus.txt + val.txt to S3
#   3. Re-upload enwik8 SPM to prose + fallback on S3 (per 2026-04-20
#      measurement: domain-trained prose/fallback SPMs were worse than
#      enwik8 on their own content)
#
# Total upload ~30 GB. Run overnight if bandwidth is the bottleneck.
#
# Usage:
#   ./scripts/prepare_cloud_training.sh              # all 7 domains
#   ./scripts/prepare_cloud_training.sh prose code   # subset
#   DRY_RUN=1 ./scripts/prepare_cloud_training.sh    # check without uploading

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SPEC_DIR="$REPO/data/specialists"
S3_BASE="s3://dmatth1-bnn-checkpoints/l3tc/specialists"
ENWIK8_SPM="$REPO/vendor/L3TC/dictionary/vocab_enwik8_bpe_16384_0.999/spm_enwik8_bpe_16384_0.999.model"
VAL_FRACTION="${VAL_FRACTION:-0.05}"  # 5% held-out
SEED="${SEED:-1204}"                   # deterministic split
SCRATCH="${SCRATCH:-/tmp/l3tc_prep}"

DOMAINS_ALL="prose code structured logs tabular markup fallback"
DOMAINS="${*:-$DOMAINS_ALL}"

mkdir -p "$SCRATCH"

for d in $DOMAINS; do
    case "$d" in prose|code|structured|logs|tabular|markup|fallback) ;;
        *) echo "unknown domain: $d"; exit 1;; esac
done

echo "=== Phase 14 cloud-training prep ==="
echo "domains:       $DOMAINS"
echo "val fraction:  $VAL_FRACTION (seed $SEED)"
echo "scratch:       $SCRATCH"
echo "dry-run:       ${DRY_RUN:-0}"
echo ""

for d in $DOMAINS; do
    echo "--- $d ---"
    DOMAIN_DIR="$SPEC_DIR/$d"
    if [ ! -d "$DOMAIN_DIR" ]; then
        echo "  skip: $DOMAIN_DIR not present"
        continue
    fi

    CORPUS="$SCRATCH/${d}_corpus.txt"
    VAL="$SCRATCH/${d}_val.txt"

    # 1. Concat all *.txt in domain dir (excluding _original/ backups
    #    and any stray SPM training logs). Use -print0 + xargs -0 because
    #    the repo path contains a space.
    n=$(find "$DOMAIN_DIR" -maxdepth 1 -type f -name "*.txt" \
        -not -name "spm_*" -not -name "_*" | wc -l)
    if [ "$n" -eq 0 ]; then
        echo "  skip: no .txt files in $DOMAIN_DIR"
        continue
    fi
    echo "  concatenating $n .txt files"
    find "$DOMAIN_DIR" -maxdepth 1 -type f -name "*.txt" \
        -not -name "spm_*" -not -name "_*" -print0 \
        | sort -z | xargs -0 cat > "$CORPUS"
    bytes=$(wc -c < "$CORPUS")
    mb=$(( bytes / 1024 / 1024 ))
    echo "  corpus: ${mb} MB"

    # 2. Deterministic train/val split by byte offset. We avoid line-random
    #    splits because they'd need the whole file in memory.
    val_bytes=$(python3 -c "import math; print(int($bytes * $VAL_FRACTION))")
    train_bytes=$(( bytes - val_bytes ))
    # Put val at the tail — simple and deterministic. For most corpora
    # the tail is interchangeable with head; Phase 11 used the same trick.
    head -c "$train_bytes" "$CORPUS" > "$SCRATCH/${d}_train.txt"
    tail -c "$val_bytes" "$CORPUS" > "$VAL"
    train_mb=$(( train_bytes / 1024 / 1024 ))
    val_mb=$(( val_bytes / 1024 / 1024 ))
    echo "  train.txt: ${train_mb} MB   val.txt: ${val_mb} MB"

    # 3. Upload.
    if [ -n "${DRY_RUN:-}" ]; then
        echo "  [DRY_RUN] skipping S3 upload"
    else
        echo "  uploading train.txt + val.txt to $S3_BASE/$d/"
        aws s3 cp "$SCRATCH/${d}_train.txt" "$S3_BASE/$d/corpus.txt" --no-progress
        aws s3 cp "$VAL" "$S3_BASE/$d/val.txt" --no-progress
    fi

    # 4. For prose + fallback, replace the domain-trained SPM on S3 with
    #    enwik8. Per 2026-04-20 measurement the domain SPMs were worse
    #    than enwik8 on their own content for these two.
    if [ "$d" = "prose" ] || [ "$d" = "fallback" ]; then
        if [ -n "${DRY_RUN:-}" ]; then
            echo "  [DRY_RUN] would upload $ENWIK8_SPM → $S3_BASE/$d/spm.model"
        else
            echo "  swapping $d spm.model → enwik8 SPM on S3"
            aws s3 cp "$ENWIK8_SPM" "$S3_BASE/$d/spm.model" --no-progress
        fi
    fi

    # 5. Clean up scratch (keep train.txt until upload confirmed; clean
    #    corpus.txt concat which is no longer needed).
    rm -f "$CORPUS"

    echo ""
done

echo "=== prep complete ==="
if [ -z "${DRY_RUN:-}" ]; then
    echo "verify with: aws s3 ls $S3_BASE/ --recursive | grep -E 'corpus|val|spm'"
    echo "then launch: ./scripts/launch_all_specialists.sh"
fi

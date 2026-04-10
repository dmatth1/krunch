#!/bin/bash
# On-demand variant of the spot fleet launcher. Same userdata, same
# everything else — just runs `aws ec2 run-instances` directly so we
# bypass spot capacity / quota issues. Slightly more expensive (~$1.20/hr
# vs ~$0.80/hr) but for a 30-60 min Phase 11 run the difference is
# pennies and on-demand has none of the spot pain.
#
# Usage:
#   L3TC_GITHUB_PAT=ghp_... ./scripts/launch-ondemand.sh [PASS] [RUN_ID]

set -euo pipefail

REGION="us-east-1"
S3_BUCKET="dmatth1-bnn-checkpoints"
S3_PREFIX="l3tc"
KEY_NAME="swarm-ec2"
SG_ID="sg-0af8b62d12cf4272c"
IAM_PROFILE_ARN="arn:aws:iam::584956668248:instance-profile/bnn-s3-access"
# g5.xlarge: A10G 24 GB, 4 vCPU, 16 GB RAM. Plenty for batch=8 training
# (~5-6 GB VRAM). g5.2xlarge was only needed for the old bootstrap issues.
INSTANCE_TYPE="g5.xlarge"

if [ -z "${L3TC_GITHUB_PAT:-}" ]; then
    echo "ERROR: L3TC_GITHUB_PAT not set. export L3TC_GITHUB_PAT=\$(cat ~/.l3tc-pat)"
    exit 1
fi

PASS="${1:-pass1}"
RUN_ID="${2:-phase11_${PASS}_$(date +%Y%m%d_%H%M%S)_od}"

echo "=== L3TC Phase 11 On-Demand Launcher ==="
echo "Pass:   ${PASS}"
echo "Run ID: ${RUN_ID}"

# Pre-baked AMI with everything installed: torch 2.1.2+cu121, numpy<2,
# sentencepiece, tqdm, ninja, pkuseg+deepspeed stubs, WKV CUDA kernel
# compiled, SPM tokenizer, enwik9 pre-tokenized corpus. Validated with
# 100-step bf16 canary at 23.3 it/s on A10G.
# Fallback to Deep Learning Base AMI if the baked AMI is not available.
BAKED_AMI="ami-07a4fc98c4ed4e19e"
if aws ec2 describe-images --image-ids "$BAKED_AMI" --region "$REGION" --query 'Images[0].State' --output text 2>/dev/null | grep -q available; then
    AMI="$BAKED_AMI"
    echo "AMI: ${AMI} (pre-baked l3tc-phase11-base-v1)"
else
    AMI=$(aws ec2 describe-images --region "$REGION" --owners amazon \
        --filters \
            "Name=name,Values=Deep Learning Base AMI with Single CUDA (Ubuntu 22.04)*" \
            "Name=state,Values=available" \
            "Name=architecture,Values=x86_64" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' --output text)
    echo "AMI: ${AMI} (Deep Learning Base, fallback — baked AMI not ready)"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
USERDATA_SCRIPT="${SCRIPT_DIR}/spot-fleet-userdata.sh"
USERDATA=$(sed \
    -e "s|PLACEHOLDER_RUN_ID|${RUN_ID}|g" \
    -e "s|PLACEHOLDER_PASS|${PASS}|g" \
    -e "s|PLACEHOLDER_GITHUB_PAT|${L3TC_GITHUB_PAT}|g" \
    -e "s|PLACEHOLDER_S3_BUCKET|${S3_BUCKET}|g" \
    -e "s|PLACEHOLDER_S3_PREFIX|${S3_PREFIX}|g" \
    "$USERDATA_SCRIPT" | base64)

INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --iam-instance-profile "Arn=${IAM_PROFILE_ARN}" \
    --user-data "$USERDATA" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=l3tc-phase11-${RUN_ID}},{Key=l3tc-run-id,Value=${RUN_ID}},{Key=l3tc-pass,Value=${PASS}}]" \
    --query 'Instances[0].InstanceId' --output text)

echo ""
echo "============================================"
echo "  Instance:  ${INSTANCE_ID} (${INSTANCE_TYPE} on-demand)"
echo "  Run ID:    ${RUN_ID}"
echo "  S3 path:   s3://${S3_BUCKET}/${S3_PREFIX}/${RUN_ID}/"
echo ""
echo "  Monitor:"
echo "    aws s3 cp s3://${S3_BUCKET}/${S3_PREFIX}/${RUN_ID}/bootstrap.log - | tail -30"
echo "    aws s3 cp s3://${S3_BUCKET}/${S3_PREFIX}/${RUN_ID}/train.log - | tail -30"
echo ""
echo "  Find IP:"
echo "    aws ec2 describe-instances --instance-ids ${INSTANCE_ID} --region ${REGION} --query 'Reservations[0].Instances[0].PublicIpAddress' --output text"
echo ""
echo "  Terminate:"
echo "    aws ec2 terminate-instances --instance-ids ${INSTANCE_ID} --region ${REGION}"
echo "============================================"

echo "${INSTANCE_ID}" > "/tmp/l3tc-ondemand-${RUN_ID}.txt"

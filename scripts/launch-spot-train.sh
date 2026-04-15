#!/bin/bash
# Launch a spot fleet for L3TC training with arbitrary args.
# Self-healing: spot reclaim → replacement resumes from S3 checkpoint.
#
# Usage:
#   export L3TC_GITHUB_PAT=$(cat ~/.l3tc-pat)
#   ./scripts/launch-spot-train.sh <RUN_ID> <TRAIN_ARGS...>
#
# Example:
#   ./scripts/launch-spot-train.sh exp_12l_32k \
#       --num-layers 12 --vocab-size 32768 \
#       --train-file s3://dmatth1-bnn-checkpoints/l3tc/corpora/train_pile_32k.txt \
#       --tokenizer-s3 s3://dmatth1-bnn-checkpoints/l3tc/corpora/tokenizer_pile_32k/spm_pile_bpe_32768.model \
#       --epochs 20 --epoch-length 500000 --batch-size 8 --grad-accum 4

set -euo pipefail

REGION="us-east-1"
S3_BUCKET="dmatth1-bnn-checkpoints"
S3_PREFIX="l3tc"
KEY_NAME="swarm-ec2"
SG_ID="sg-0af8b62d12cf4272c"
IAM_PROFILE_ARN="arn:aws:iam::584956668248:instance-profile/bnn-s3-access"
BAKED_AMI="ami-07a4fc98c4ed4e19e"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Default instance type. Override with INSTANCE_TYPE env var for
# larger models that need more VRAM (e.g., INSTANCE_TYPE=g6e.xlarge
# for 48 GB VRAM models).
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"

if [ -z "${L3TC_GITHUB_PAT:-}" ]; then
    echo "ERROR: L3TC_GITHUB_PAT not set."
    exit 1
fi
if [ $# -lt 2 ]; then
    echo "Usage: $0 <RUN_ID> <TRAIN_ARGS...>"
    echo "  INSTANCE_TYPE=g6e.xlarge $0 ...  # for larger models"
    exit 1
fi

RUN_ID="$1"; shift
TRAIN_ARGS="$*"
S3_RUN="s3://${S3_BUCKET}/${S3_PREFIX}/${RUN_ID}"

echo "=== L3TC Spot Training ==="
echo "Run ID: ${RUN_ID}"
echo "Args:   ${TRAIN_ARGS}"
echo "S3:     ${S3_RUN}"

# Resolve AMI
AMI="$BAKED_AMI"
aws ec2 describe-images --image-ids "$AMI" --region "$REGION" --query 'Images[0].State' --output text 2>/dev/null | grep -q available || {
    AMI=$(aws ec2 describe-images --region "$REGION" --owners amazon \
        --filters "Name=name,Values=Deep Learning Base AMI with Single CUDA (Ubuntu 22.04)*" \
        "Name=state,Values=available" "Name=architecture,Values=x86_64" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' --output text)
}
echo "AMI: ${AMI}"

# Build userdata from template
TEMPLATE="${SCRIPT_DIR}/spot-train-userdata.sh.template"
USERDATA=$(sed \
    -e "s|__GITHUB_PAT__|${L3TC_GITHUB_PAT}|g" \
    -e "s|__RUN_ID__|${RUN_ID}|g" \
    -e "s|__S3_RUN__|${S3_RUN}|g" \
    -e "s|__TRAIN_ARGS__|${TRAIN_ARGS}|g" \
    "$TEMPLATE" | base64)

SIZE_RAW=$(sed -e "s|__GITHUB_PAT__|${L3TC_GITHUB_PAT}|g" -e "s|__RUN_ID__|${RUN_ID}|g" -e "s|__S3_RUN__|${S3_RUN}|g" -e "s|__TRAIN_ARGS__|${TRAIN_ARGS}|g" "$TEMPLATE" | wc -c)
echo "userdata: ${SIZE_RAW} bytes"
if [ "$SIZE_RAW" -gt 12000 ]; then
    echo "WARNING: userdata is large (${SIZE_RAW} bytes). EC2 limit is 16384 base64."
fi

# Single instance type — no diversification. Pick the right one
# for your model's VRAM needs:
#   g5.xlarge  (A10G 24GB)  — 6L models, batch 8-12
#   g6e.xlarge (L40S 48GB)  — 12L models, larger batches
echo "Instance: ${INSTANCE_TYPE}"
LAUNCH_SPECS="{\"ImageId\":\"${AMI}\",\"InstanceType\":\"${INSTANCE_TYPE}\",\"KeyName\":\"${KEY_NAME}\",\"SecurityGroups\":[{\"GroupId\":\"${SG_ID}\"}],\"IamInstanceProfile\":{\"Arn\":\"${IAM_PROFILE_ARN}\"},\"BlockDeviceMappings\":[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":200,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}],\"UserData\":\"${USERDATA}\",\"TagSpecifications\":[{\"ResourceType\":\"instance\",\"Tags\":[{\"Key\":\"Name\",\"Value\":\"l3tc-${RUN_ID}\"},{\"Key\":\"l3tc-run-id\",\"Value\":\"${RUN_ID}\"}]}]}"

# Submit fleet
FLEET_ID=$(echo "{\"IamFleetRole\":\"arn:aws:iam::584956668248:role/aws-ec2-spot-fleet-tagging-role\",\"TargetCapacity\":1,\"SpotPrice\":\"3.50\",\"TerminateInstancesWithExpiration\":false,\"Type\":\"maintain\",\"AllocationStrategy\":\"capacityOptimized\",\"LaunchSpecifications\":[${LAUNCH_SPECS}]}" \
    | aws ec2 request-spot-fleet --region "$REGION" --spot-fleet-request-config file:///dev/stdin --query 'SpotFleetRequestId' --output text)

echo ""
echo "============================================"
echo "  Spot Fleet: ${FLEET_ID}"
echo "  Run ID:     ${RUN_ID}"
echo "  S3:         ${S3_RUN}/"
echo ""
echo "  Monitor:"
echo "    aws s3 cp ${S3_RUN}/train.log - | tail -30"
echo ""
echo "  Cancel:"
echo "    aws ec2 cancel-spot-fleet-requests --spot-fleet-request-ids ${FLEET_ID} --terminate-instances --region ${REGION}"
echo "============================================"
echo "${FLEET_ID}" > "/tmp/l3tc-fleet-${RUN_ID}.txt"

#!/bin/bash
# Launch a self-healing Spot Fleet for L3TC Phase 11 training.
# The fleet maintains exactly 1 g6e.xlarge instance.
# When spot reclaims, a replacement auto-launches and resumes from the
# latest .pth checkpoint in S3.
#
# Usage:
#   L3TC_GITHUB_PAT=ghp_... ./scripts/launch-spot-fleet.sh [PASS] [RUN_ID]
#
#   PASS:    pass1 (enwik9 sanity check) or pass2 (Pile dedup broader corpus)
#            Defaults to pass1.
#   RUN_ID:  optional explicit run ID. Without it, generates a timestamped one.
#            With it, resumes an existing run by reading checkpoints from
#            s3://dmatth1-bnn-checkpoints/l3tc/<RUN_ID>/.
#
# Reuses the bnn S3 bucket with an l3tc/ prefix so we don't have to create
# new IAM roles, security groups, or buckets. The bnn-s3-access instance
# profile already has read/write to s3://dmatth1-bnn-checkpoints/* so it
# transparently covers the l3tc/ prefix.
#
# Monitor:
#   aws s3 cp s3://dmatth1-bnn-checkpoints/l3tc/<RUN_ID>/train.log - | tail -50
# Cancel:
#   aws ec2 cancel-spot-fleet-requests --spot-fleet-request-ids <FLEET_ID> \
#       --terminate-instances --region us-east-1

set -euo pipefail

REGION="us-east-1"
S3_BUCKET="dmatth1-bnn-checkpoints"
S3_PREFIX="l3tc"
KEY_NAME="swarm-ec2"
SG_ID="sg-0af8b62d12cf4272c"
IAM_PROFILE_ARN="arn:aws:iam::584956668248:instance-profile/bnn-s3-access"
# Diversify across instance types to dodge per-AZ spot capacity drops.
# All four are NVIDIA single-GPU spots in the 24-48 GB VRAM range and
# all are massive overkill for a 200K-param model — capacityOptimized
# picks whichever has stock at launch. The training script doesn't
# care which we land on. Order is roughly cheapest-first.
INSTANCE_TYPES=("g5.xlarge" "g6.xlarge" "g5.2xlarge" "g6e.xlarge")

# === Validate inputs ===
if [ -z "${L3TC_GITHUB_PAT:-}" ]; then
    echo "ERROR: L3TC_GITHUB_PAT environment variable is not set."
    echo ""
    echo "The userdata script needs a GitHub Personal Access Token to clone"
    echo "the l3tc-prod repo on the EC2 instance. Set it before launching:"
    echo ""
    echo "  export L3TC_GITHUB_PAT=ghp_yourtokenhere"
    echo "  ./scripts/launch-spot-fleet.sh pass1"
    echo ""
    echo "Use a fine-grained PAT scoped to the dmatth1/l3tc-prod repo with"
    echo "read-only contents access. The token is base64-encoded into the"
    echo "instance userdata at launch time and never written to disk in"
    echo "this repo."
    exit 1
fi

PASS="${1:-pass1}"
case "$PASS" in
    pass1|pass2) ;;
    *)
        echo "ERROR: PASS must be 'pass1' or 'pass2', got '${PASS}'"
        exit 1
        ;;
esac

# Use provided run ID or generate a new one
if [ -n "${2:-}" ]; then
    RUN_ID="$2"
    echo "=== L3TC Phase 11 Spot Fleet Launcher (resuming) ==="
else
    RUN_ID="phase11_${PASS}_$(date +%Y%m%d_%H%M%S)"
    echo "=== L3TC Phase 11 Spot Fleet Launcher (fresh) ==="
fi
echo "Pass:   ${PASS}"
echo "Run ID: ${RUN_ID}"

# === Find latest Deep Learning AMI ===
AMI=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters \
        "Name=name,Values=Deep Learning Base AMI with Single CUDA (Ubuntu 22.04)*" \
        "Name=state,Values=available" \
        "Name=architecture,Values=x86_64" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text)
echo "AMI: ${AMI}"

# === Read and prepare user data script ===
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
USERDATA_SCRIPT="${SCRIPT_DIR}/spot-fleet-userdata.sh"

if [ ! -f "$USERDATA_SCRIPT" ]; then
    echo "ERROR: ${USERDATA_SCRIPT} not found"
    exit 1
fi

# Substitute placeholders in userdata. The PAT is injected at launch
# time and base64-encoded into the EC2 user data, never committed to git.
USERDATA=$(sed \
    -e "s|PLACEHOLDER_RUN_ID|${RUN_ID}|g" \
    -e "s|PLACEHOLDER_PASS|${PASS}|g" \
    -e "s|PLACEHOLDER_GITHUB_PAT|${L3TC_GITHUB_PAT}|g" \
    -e "s|PLACEHOLDER_S3_BUCKET|${S3_BUCKET}|g" \
    -e "s|PLACEHOLDER_S3_PREFIX|${S3_PREFIX}|g" \
    "$USERDATA_SCRIPT" | base64)

# === Build LaunchSpecifications: one per instance type for diversification ===
LAUNCH_SPECS=""
for itype in "${INSTANCE_TYPES[@]}"; do
    if [ -n "$LAUNCH_SPECS" ]; then
        LAUNCH_SPECS="${LAUNCH_SPECS},"
    fi
    LAUNCH_SPECS="${LAUNCH_SPECS}$(cat <<EOF
        {
            "ImageId": "${AMI}",
            "InstanceType": "${itype}",
            "KeyName": "${KEY_NAME}",
            "SecurityGroups": [{"GroupId": "${SG_ID}"}],
            "IamInstanceProfile": {"Arn": "${IAM_PROFILE_ARN}"},
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "VolumeSize": 100,
                        "VolumeType": "gp3",
                        "DeleteOnTermination": true
                    }
                }
            ],
            "UserData": "${USERDATA}",
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": "l3tc-phase11-${RUN_ID}"},
                        {"Key": "l3tc-run-id", "Value": "${RUN_ID}"},
                        {"Key": "l3tc-pass", "Value": "${PASS}"}
                    ]
                }
            ]
        }
EOF
)"
done

# === Create Spot Fleet request config ===
FLEET_CONFIG=$(cat <<EOF
{
    "IamFleetRole": "arn:aws:iam::584956668248:role/aws-ec2-spot-fleet-tagging-role",
    "TargetCapacity": 1,
    "SpotPrice": "2.00",
    "TerminateInstancesWithExpiration": false,
    "Type": "maintain",
    "AllocationStrategy": "capacityOptimized",
    "LaunchSpecifications": [
${LAUNCH_SPECS}
    ]
}
EOF
)

# === Verify spot fleet role exists (created already by bnn launcher) ===
echo "Checking spot fleet role..."
aws iam get-role --role-name aws-ec2-spot-fleet-tagging-role > /dev/null 2>&1 || {
    echo "ERROR: aws-ec2-spot-fleet-tagging-role not found."
    echo "It should have been created by bnn's launcher already."
    echo "Run: aws iam create-role --role-name aws-ec2-spot-fleet-tagging-role ..."
    exit 1
}

# === Submit fleet request ===
echo "Submitting Spot Fleet request..."
FLEET_ID=$(echo "$FLEET_CONFIG" | aws ec2 request-spot-fleet \
    --region "$REGION" \
    --spot-fleet-request-config file:///dev/stdin \
    --query 'SpotFleetRequestId' \
    --output text)

echo ""
echo "============================================"
echo "  Spot Fleet: ${FLEET_ID}"
echo "  Run ID:     ${RUN_ID}"
echo "  Pass:       ${PASS}"
echo "  Instances:  ${INSTANCE_TYPES[*]} (capacityOptimized)"
echo "  S3 path:    s3://${S3_BUCKET}/${S3_PREFIX}/${RUN_ID}/"
echo ""
echo "  Monitor training:"
echo "    aws s3 cp s3://${S3_BUCKET}/${S3_PREFIX}/${RUN_ID}/train.log - | tail -50"
echo ""
echo "  Tail bootstrap log (before training starts):"
echo "    aws s3 cp s3://${S3_BUCKET}/${S3_PREFIX}/${RUN_ID}/bootstrap.log - | tail -50"
echo ""
echo "  Find instance IP:"
echo "    aws ec2 describe-instances --region ${REGION} --filters 'Name=tag:l3tc-run-id,Values=${RUN_ID}' 'Name=instance-state-name,Values=running' --query 'Reservations[0].Instances[0].PublicIpAddress' --output text"
echo ""
echo "  Cancel fleet:"
echo "    aws ec2 cancel-spot-fleet-requests --spot-fleet-request-ids ${FLEET_ID} --terminate-instances --region ${REGION}"
echo "============================================"

# Save fleet info for later reference
echo "${FLEET_ID}" > "/tmp/l3tc-fleet-${RUN_ID}.txt"
echo "Fleet ID saved to /tmp/l3tc-fleet-${RUN_ID}.txt"

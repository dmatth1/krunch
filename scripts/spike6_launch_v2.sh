#!/bin/bash
# Spike 6 launch orchestrator v2 — run from laptop.
# Narrowed scope: P1 (ratio) + P2 (throughput) only.

set -euo pipefail

REGION=us-east-1
AMI=ami-09d0a18beb02cc7d4  # Deep Learning OSS PyTorch 2.7 (Ubuntu 22.04)
INSTANCE_TYPE=g5.2xlarge
KEY_NAME=swarm-ec2
SG_ID=sg-06762942403f0f3fb
SUBNET_ID=subnet-0f57e97d8411d102d  # us-east-1d
IAM_PROFILE=archive-dev-training-BatchComputeInstanceProfile3F53073B-Xf3fwba3pnGI
IAM_ROLE=archive-dev-training-BatchComputeInstanceProfileRol-Emf4dvQx0XMy
S3_PREFIX=s3://archive-dev-archive/spike6/gpu-throughput

cd "$(dirname "$0")/.."  # repo root

echo "=== 1. Bundle code ==="
tar czf /tmp/spike6_code_v2.tar.gz \
    scripts/rwkv_zeroshot_ratio.py \
    scripts/smollm2_throughput_test.py \
    scripts/spike6_runner_v2.sh
ls -lh /tmp/spike6_code_v2.tar.gz

echo "=== 2. Upload code + clear old progress sentinels ==="
# Clear old progress markers so outside observer doesn't get confused
aws s3 rm "$S3_PREFIX/" --recursive --exclude "*" --include "progress_*" 2>&1 | tail -3 || true
aws s3 cp /tmp/spike6_code_v2.tar.gz "$S3_PREFIX/code.tar.gz" --quiet

echo "=== 3. Ensure IAM S3 policy on compute role ==="
aws iam put-role-policy \
    --role-name "$IAM_ROLE" \
    --policy-name KrunchSpike6S3 \
    --policy-document '{
        "Version":"2012-10-17",
        "Statement":[{
            "Effect":"Allow",
            "Action":["s3:GetObject","s3:PutObject","s3:ListBucket"],
            "Resource":["arn:aws:s3:::archive-dev-archive","arn:aws:s3:::archive-dev-archive/*"]
        }]
    }' > /dev/null

echo "=== 4. Ensure SSH ingress from my IP ==="
MYIP=$(curl -s https://api.ipify.org)
aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp --port 22 --cidr "${MYIP}/32" \
    2>/dev/null || echo "  (rule already exists)"

echo "=== 5. Prep user-data ==="
# User-data is minimal: stage, kick off runner in background, let it self-terminate.
cat > /tmp/spike6_userdata_v2.sh <<'USERDATA'
#!/bin/bash
set -euxo pipefail
exec > /var/log/spike6.log 2>&1

# Early-fail sentinel: prove user-data even ran at all
echo "$(date -u +%FT%TZ) USERDATA_START" | aws s3 cp - s3://archive-dev-archive/spike6/gpu-throughput/progress_USERDATA_START.txt

# Stage code
mkdir -p /home/ubuntu/spike6
aws s3 cp s3://archive-dev-archive/spike6/gpu-throughput/code.tar.gz /tmp/code.tar.gz
tar xzf /tmp/code.tar.gz -C /home/ubuntu/spike6
chmod +x /home/ubuntu/spike6/scripts/spike6_runner_v2.sh

# Stage corpus
mkdir -p /tmp/corpus
aws s3 cp s3://archive-dev-archive/spike6/corpus/wildchat_en_content.content.bin /tmp/corpus/content.bin --quiet

# Sentinel: everything staged, about to kick off runner
echo "$(date -u +%FT%TZ) USERDATA_STAGED" | aws s3 cp - s3://archive-dev-archive/spike6/gpu-throughput/progress_USERDATA_STAGED.txt

# Kick off runner in a detached shell so user-data can complete cleanly.
# Runner handles all measurements + S3 uploads + shutdown itself.
setsid nohup /home/ubuntu/spike6/scripts/spike6_runner_v2.sh > /var/log/spike6_runner.log 2>&1 < /dev/null &
disown
USERDATA
ls -lh /tmp/spike6_userdata_v2.sh

echo "=== 6. Launch $INSTANCE_TYPE ==="
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --subnet-id "$SUBNET_ID" \
    --iam-instance-profile "Name=$IAM_PROFILE" \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}' \
    --instance-initiated-shutdown-behavior terminate \
    --user-data file:///tmp/spike6_userdata_v2.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=Krunch},{Key=Spike,Value=6},{Key=Purpose,Value=p1-p2-gpu-run},{Key=AutoTerminate,Value=true}]" \
    --query 'Instances[0].InstanceId' --output text)

echo "instance: $INSTANCE_ID"
echo "monitor:  aws s3 ls $S3_PREFIX/"
echo "$INSTANCE_ID" > /tmp/spike6_instance_id_v2

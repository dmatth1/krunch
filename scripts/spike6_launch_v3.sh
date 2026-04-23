#!/bin/bash
# Spike 6 launch v3 — throughput-only, with CUDA kernel.
set -euo pipefail

REGION=us-east-1
AMI=ami-09d0a18beb02cc7d4
INSTANCE_TYPE=g5.2xlarge
KEY_NAME=swarm-ec2
SG_ID=sg-06762942403f0f3fb
SUBNET_ID=subnet-0f57e97d8411d102d
IAM_PROFILE=archive-dev-training-BatchComputeInstanceProfile3F53073B-Xf3fwba3pnGI
S3_PREFIX=s3://archive-dev-archive/spike6/gpu-throughput

cd "$(dirname "$0")/.."

echo "=== 1. Bundle ==="
tar czf /tmp/spike6_code_v3.tar.gz \
    scripts/rwkv_kernel_throughput.py \
    scripts/spike6_runner_v3.sh
aws s3 cp /tmp/spike6_code_v3.tar.gz "$S3_PREFIX/code_v3.tar.gz" --quiet

echo "=== 2. Clear old progress sentinels ==="
aws s3 rm "$S3_PREFIX/" --recursive --exclude "*" --include "progress_*" 2>&1 | tail -3 || true

echo "=== 3. Prep user-data ==="
cat > /tmp/spike6_userdata_v3.sh <<'USERDATA'
#!/bin/bash
set -euxo pipefail
exec > /var/log/spike6_v3.log 2>&1
echo "$(date -u +%FT%TZ) USERDATA_START" | aws s3 cp - s3://archive-dev-archive/spike6/gpu-throughput/progress_V3_USERDATA_START.txt
mkdir -p /home/ubuntu/spike6
aws s3 cp s3://archive-dev-archive/spike6/gpu-throughput/code_v3.tar.gz /tmp/code.tar.gz
tar xzf /tmp/code.tar.gz -C /home/ubuntu/spike6
chmod +x /home/ubuntu/spike6/scripts/spike6_runner_v3.sh
mkdir -p /tmp/corpus
aws s3 cp s3://archive-dev-archive/spike6/corpus/wildchat_en_content.content.bin /tmp/corpus/content.bin --quiet
echo "$(date -u +%FT%TZ) USERDATA_STAGED" | aws s3 cp - s3://archive-dev-archive/spike6/gpu-throughput/progress_V3_USERDATA_STAGED.txt
setsid nohup /home/ubuntu/spike6/scripts/spike6_runner_v3.sh > /var/log/spike6_v3_runner.log 2>&1 < /dev/null &
disown
USERDATA

echo "=== 4. Launch ==="
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --subnet-id "$SUBNET_ID" \
    --iam-instance-profile "Name=$IAM_PROFILE" \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}' \
    --instance-initiated-shutdown-behavior terminate \
    --user-data file:///tmp/spike6_userdata_v3.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=Krunch},{Key=Spike,Value=6},{Key=Purpose,Value=v3-cuda-kernel},{Key=AutoTerminate,Value=true}]" \
    --query 'Instances[0].InstanceId' --output text)

echo "instance: $INSTANCE_ID"
echo "$INSTANCE_ID" > /tmp/spike6_instance_id_v3

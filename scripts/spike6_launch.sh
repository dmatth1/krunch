#!/bin/bash
# Spike 6 launch orchestrator — run from laptop.
#
# Steps:
#   1. Bundle repo scripts + vendor/L3TC to a tarball
#   2. Upload tarball to S3
#   3. Ensure IAM role has S3 read/write on archive-dev-archive
#   4. Ensure SG ingress allows SSH from my IP (for debugging only)
#   5. Launch g5.2xlarge with DLAMI PyTorch AMI
#   6. User-data runs scripts/spike6_gpu_runner.sh
#   7. Instance self-terminates when done (or at 2h watchdog)

set -euo pipefail

REGION=us-east-1
AMI=ami-09d0a18beb02cc7d4  # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04)
INSTANCE_TYPE=g5.2xlarge
KEY_NAME=swarm-ec2
SG_ID=sg-06762942403f0f3fb
SUBNET_ID=subnet-0e4f032207e25b74d
IAM_PROFILE=archive-dev-training-BatchComputeInstanceProfile3F53073B-Xf3fwba3pnGI
IAM_ROLE=archive-dev-training-BatchComputeInstanceProfileRol-Emf4dvQx0XMy
S3_PREFIX=s3://archive-dev-archive/spike6/gpu-throughput

cd "$(dirname "$0")/.."  # repo root

echo "=== 1. Bundle code ==="
tar czf /tmp/spike6_code.tar.gz \
    scripts/gpu_throughput_test.py \
    scripts/rwkv_zeroshot_ratio.py \
    scripts/rwkv_lora_train.py \
    scripts/smollm2_throughput_test.py \
    scripts/smollm2_lora_train.py \
    scripts/measure_held_out_ratio.py \
    scripts/train_l3tc_phase11.py \
    scripts/spike6_gpu_runner.sh \
    scripts/preprocess_chat.py \
    vendor/L3TC
ls -lh /tmp/spike6_code.tar.gz

echo "=== 2. Upload code + userdata to S3 ==="
aws s3 cp /tmp/spike6_code.tar.gz "$S3_PREFIX/code.tar.gz" --quiet

echo "=== 3. Ensure IAM S3 policy ==="
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
# User-data: download + run the staged runner script. Keeps user-data
# small + survives most cloud-init quirks.
cat > /tmp/spike6_userdata.sh <<'USERDATA'
#!/bin/bash
set -euxo pipefail
exec > /var/log/spike6.log 2>&1
cd /home/ubuntu
aws s3 cp s3://archive-dev-archive/spike6/gpu-throughput/code.tar.gz /tmp/code.tar.gz
mkdir -p /home/ubuntu/spike6
tar xzf /tmp/code.tar.gz -C /home/ubuntu/spike6
chmod +x /home/ubuntu/spike6/scripts/spike6_gpu_runner.sh
# Run as ubuntu user in background; instance will self-terminate.
sudo -u ubuntu nohup /home/ubuntu/spike6/scripts/spike6_gpu_runner.sh > /var/log/spike6_runner.log 2>&1 &
USERDATA
ls -lh /tmp/spike6_userdata.sh

echo "=== 6. Launch g5.2xlarge ==="
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --subnet-id "$SUBNET_ID" \
    --iam-instance-profile "Name=$IAM_PROFILE" \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}' \
    --instance-initiated-shutdown-behavior terminate \
    --user-data file:///tmp/spike6_userdata.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=Krunch},{Key=Spike,Value=6},{Key=Purpose,Value=three-track-gpu-run},{Key=AutoTerminate,Value=true}]" \
    --query 'Instances[0].InstanceId' --output text)

echo "instance: $INSTANCE_ID"
echo "region: $REGION"
echo "monitor: aws s3 ls $S3_PREFIX/"
echo "ssh: (wait ~90s for boot, then ssh -i ~/.ssh/swarm-ec2.pem ubuntu@\$IP)"
echo "$INSTANCE_ID" > /tmp/spike6_instance_id

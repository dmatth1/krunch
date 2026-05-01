#!/bin/bash
# Tier 3 — single-shot roundtrip on a g5.xlarge spot.
#
# Validates the shipped artifact: pulls ghcr.io/dmatth1/krunch:latest,
# installs the krunch CLI, runs `krunch compress` + `krunch decompress`
# on a 100 MB WildChat sample. Same exact path as a real user, just
# automated end-to-end.
#
# Gates (tightened 2026-04-30): ratio ≤ 0.11, compress AND decompress
# avg ≥ 200 KB/s, byte-exact roundtrip.
#
# Cost: ~$0.30/hr × ~10 min ≈ $0.05 (image pull + run, no build).
# Set KRUNCH_LOCAL_BUILD=1 to build from source instead of pulling
# (useful before ghcr.io publish exists, or to test a local diff).

set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="g5.xlarge"

# These are author-specific by default — override via env to point at your
# own AWS resources. Required: an SSH key pair, a security group, an S3
# bucket the instance can read/write, and an IAM instance profile that
# grants access to that bucket.
KEY_NAME="${KRUNCH_KEY_PAIR:?KRUNCH_KEY_PAIR env var required (your EC2 key pair name)}"
SECURITY_GROUP="${KRUNCH_SG:?KRUNCH_SG env var required (security group name)}"
S3_BUCKET="${KRUNCH_S3_BUCKET:?KRUNCH_S3_BUCKET env var required (bucket the test reads/writes)}"
INSTANCE_PROFILE="${KRUNCH_INSTANCE_PROFILE:?KRUNCH_INSTANCE_PROFILE env var required (IAM profile with S3 access to KRUNCH_S3_BUCKET)}"

S3_PREFIX="krunch-tier3"
SAMPLE_LOCAL="data/spike6/wildchat_en_content.content.bin"
SAMPLE_LIMIT_MB="${KRUNCH_SAMPLE_MB:-100}"
LOCAL_BUILD="${KRUNCH_LOCAL_BUILD:-0}"
KRUNCH_IMAGE_TAG="${KRUNCH_IMAGE:-ghcr.io/dmatth1/krunch:latest}"
TEST_TAG="$(date +%Y%m%d-%H%M%S)"
S3_BASE="s3://${S3_BUCKET}/${S3_PREFIX}/${TEST_TAG}"
RESULT_LOCAL="/tmp/krunch-tier3-result-${TEST_TAG}.json"

echo "=== Tier 3 single-shot Docker roundtrip — ${TEST_TAG} ==="
echo "  region:       ${REGION}"
echo "  instance:     ${INSTANCE_TYPE} spot"
echo "  S3 base:      ${S3_BASE}"
echo "  image:        ${KRUNCH_IMAGE_TAG}  (build locally: ${LOCAL_BUILD})"
echo "  sample limit: ${SAMPLE_LIMIT_MB} MB"

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
[[ -f $SAMPLE_LOCAL ]] || { echo "FAIL sample missing: $SAMPLE_LOCAL"; exit 1; }

AMI_ID=$(aws ec2 describe-images --region "$REGION" --owners amazon \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04) *" \
  --query "sort_by(Images, &CreationDate)[-1].ImageId" --output text)
[[ $AMI_ID =~ ^ami- ]] || { echo "FAIL DLAMI lookup failed"; exit 1; }
echo "  AMI:          $AMI_ID"

# ---------------------------------------------------------------------------
# 1. Upload artifacts to S3
# ---------------------------------------------------------------------------
echo
echo "[1/5] Uploading repo + sample to S3..."

SAMPLE_BYTES=$(( SAMPLE_LIMIT_MB * 1024 * 1024 ))
head -c "$SAMPLE_BYTES" "$SAMPLE_LOCAL" | aws s3 cp --quiet - "${S3_BASE}/sample.bin"
echo "  uploaded ${SAMPLE_LIMIT_MB} MB sample (install.sh + wrapper come from public repo)"

# If local-build mode requested, also upload the repo tarball + ensure the
# model bundle is present (used by the user-data fallback path below).
REPO_TAR=""
if [[ $LOCAL_BUILD == 1 ]]; then
  REPO_TAR=/tmp/krunch-repo-${TEST_TAG}.tar.gz
  tar czf "$REPO_TAR" \
    --exclude=models --exclude=data --exclude=.git --exclude=node_modules \
    --exclude=__pycache__ --exclude=cdk.out \
    -C "$(pwd)/.." "$(basename "$(pwd)")"
  aws s3 cp --quiet "$REPO_TAR" "${S3_BASE}/repo.tar.gz"
  aws s3 ls "s3://${S3_BUCKET}/krunch/rwkv_bundle.tar.gz" >/dev/null \
    || { echo "FAIL model bundle missing for local-build"; exit 1; }
  echo "  + repo.tar.gz $(du -sh "$REPO_TAR" | cut -f1) (LOCAL_BUILD=1)"
fi

# ---------------------------------------------------------------------------
# 2. Build user-data — QUOTED heredoc (literal string, no Mac-side expansion)
#    Then sed-substitute the few placeholders we want injected.
# ---------------------------------------------------------------------------
USER_DATA=$(mktemp)
cat > "$USER_DATA" << 'USERDATA_EOF'
#!/bin/bash
set -e
exec > /var/log/krunch-tier3.log 2>&1
echo "TIER3_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"

S3_BASE="__S3_BASE__"
TEST_TAG="__TEST_TAG__"
REGION="__REGION__"
LOCAL_BUILD="__LOCAL_BUILD__"
KRUNCH_IMAGE_TAG="__KRUNCH_IMAGE_TAG__"
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)

cd /tmp
echo "FETCH_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
aws s3 cp "${S3_BASE}/sample.bin" sample.bin

if [[ "$LOCAL_BUILD" == "1" ]]; then
  # Build path — pull repo + model bundle, build image locally
  aws s3 cp "${S3_BASE}/repo.tar.gz"                                  repo.tar.gz
  aws s3 cp "s3://__S3_BUCKET__/krunch/rwkv_bundle.tar.gz"            rwkv_bundle.tar.gz
  mkdir -p /tmp/extract
  tar xzf repo.tar.gz
  tar xzf rwkv_bundle.tar.gz -C /tmp/extract
  KRUNCH_DIR=$(find /tmp -maxdepth 2 -name 'krunch' -type d | head -1)
  mkdir -p "${KRUNCH_DIR}/models"
  cp /tmp/extract/RWKV-4-Pile-169M-20220807-8023.pth "${KRUNCH_DIR}/models/"
  cp /tmp/extract/RWKV-v4/20B_tokenizer.json          "${KRUNCH_DIR}/models/"
  echo "FETCH_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"

  cd "${KRUNCH_DIR}"
  echo "BUILD_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  set -o pipefail
  docker build -t "${KRUNCH_IMAGE_TAG}" . 2>&1 | tee /tmp/docker-build.log | tail -40
  BUILD_RC=${PIPESTATUS[0]}
  if [[ $BUILD_RC -ne 0 ]]; then
    echo "BUILD_FAILED $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$BUILD_RC"
    aws s3 cp /tmp/docker-build.log "${S3_BASE}/build.log"
    aws s3 cp /var/log/krunch-tier3.log "${S3_BASE}/setup.log"
    python3 -c 'import json; json.dump({"all_gates_pass": False, "error": "docker build failed"}, open("/tmp/result.json", "w"))'
    aws s3 cp /tmp/result.json "${S3_BASE}/result.json"
    aws ec2 terminate-instances --instance-ids "${INSTANCE_ID}" --region "${REGION}"
    exit $BUILD_RC
  fi
  echo "BUILD_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  KRUNCH_WRAPPER_SRC="${KRUNCH_DIR}/scripts/krunch"
else
  # Pull path — exactly the user UX: curl install.sh from the public repo
  # and pipe to bash. install.sh handles the docker pull + wrapper install.
  echo "INSTALL_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  KRUNCH_IMAGE="${KRUNCH_IMAGE_TAG}" \
    bash -c "curl -fsSL https://raw.githubusercontent.com/dmatth1/krunch/main/install.sh | bash"
  echo "INSTALL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

# install.sh already put the wrapper at /usr/local/bin/krunch. For the
# local-build path we install it manually here.
if [[ "$LOCAL_BUILD" == "1" ]]; then
  install -m 0755 "$KRUNCH_WRAPPER_SRC" /usr/local/bin/krunch
fi
export KRUNCH_IMAGE="${KRUNCH_IMAGE_TAG}"

INPUT_BYTES=$(wc -c < /tmp/sample.bin)
echo "Input: ${INPUT_BYTES} bytes"

echo "COMPRESS_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
T0=$(date +%s.%N)
krunch compress < /tmp/sample.bin > /tmp/sample.krunch
T1=$(date +%s.%N)
COMPRESS_S=$(echo "$T1 - $T0" | bc)
COMPRESSED_BYTES=$(wc -c < /tmp/sample.krunch)
echo "COMPRESS_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) elapsed=${COMPRESS_S}s out=${COMPRESSED_BYTES}b"

echo "DECOMPRESS_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
T2=$(date +%s.%N)
krunch decompress < /tmp/sample.krunch > /tmp/sample.out
T3=$(date +%s.%N)
DECOMPRESS_S=$(echo "$T3 - $T2" | bc)
RECOVERED_BYTES=$(wc -c < /tmp/sample.out)
echo "DECOMPRESS_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) elapsed=${DECOMPRESS_S}s out=${RECOVERED_BYTES}b"

if cmp -s /tmp/sample.bin /tmp/sample.out; then BYTE_EXACT="true"; else BYTE_EXACT="false"; fi

# Build result JSON via Python (avoids floating-point shell math issues)
INPUT_BYTES=$INPUT_BYTES \
COMPRESSED_BYTES=$COMPRESSED_BYTES \
RECOVERED_BYTES=$RECOVERED_BYTES \
COMPRESS_S=$COMPRESS_S \
DECOMPRESS_S=$DECOMPRESS_S \
BYTE_EXACT=$BYTE_EXACT \
TEST_TAG=$TEST_TAG \
python3 -c '
import os, json
input_bytes  = int(os.environ["INPUT_BYTES"])
compressed   = int(os.environ["COMPRESSED_BYTES"])
recovered    = int(os.environ["RECOVERED_BYTES"])
compress_s   = float(os.environ["COMPRESS_S"])
decompress_s = float(os.environ["DECOMPRESS_S"])
byte_exact   = os.environ["BYTE_EXACT"] == "true"
ratio = compressed / input_bytes
ckb   = input_bytes / 1024 / compress_s
dkb   = input_bytes / 1024 / decompress_s
result = {
    "tag": os.environ["TEST_TAG"],
    "input_bytes": input_bytes,
    "compressed_bytes": compressed,
    "recovered_bytes": recovered,
    "ratio": round(ratio, 5),
    "compress_seconds": round(compress_s, 2),
    "decompress_seconds": round(decompress_s, 2),
    "compress_kb_s": round(ckb, 1),
    "decompress_kb_s": round(dkb, 1),
    "byte_exact": byte_exact,
    "gates": {
        "ratio_lte_0_11": ratio <= 0.11,
        "compress_kb_s_gte_200": ckb >= 200,
        "decompress_kb_s_gte_200": dkb >= 200,
        "byte_exact": byte_exact,
    },
}
result["all_gates_pass"] = all(result["gates"].values())
with open("/tmp/result.json", "w") as f: json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
'

aws s3 cp /tmp/result.json "${S3_BASE}/result.json"
aws s3 cp /var/log/krunch-tier3.log "${S3_BASE}/setup.log"
echo "TIER3_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"

aws ec2 terminate-instances --instance-ids "${INSTANCE_ID}" --region "${REGION}"
USERDATA_EOF

sed -i.bak \
  -e "s|__S3_BASE__|${S3_BASE}|g" \
  -e "s|__S3_BUCKET__|${S3_BUCKET}|g" \
  -e "s|__TEST_TAG__|${TEST_TAG}|g" \
  -e "s|__REGION__|${REGION}|g" \
  -e "s|__LOCAL_BUILD__|${LOCAL_BUILD}|g" \
  -e "s|__KRUNCH_IMAGE_TAG__|${KRUNCH_IMAGE_TAG}|g" \
  "$USER_DATA"
rm -f "${USER_DATA}.bak"

# ---------------------------------------------------------------------------
# 3. Launch instance — spot by default, on-demand if KRUNCH_ON_DEMAND=1
#    (useful when spot capacity is unavailable in your AZ)
# ---------------------------------------------------------------------------
ON_DEMAND="${KRUNCH_ON_DEMAND:-0}"
if [[ "$ON_DEMAND" == "1" ]]; then
  MARKET_ARGS=""
  MARKET_LABEL="on-demand"
else
  MARKET_ARGS='--instance-market-options MarketType=spot,SpotOptions={SpotInstanceType=one-time}'
  MARKET_LABEL="spot"
fi

echo
echo "[2/5] Launching ${INSTANCE_TYPE} (${MARKET_LABEL}) instance..."
# shellcheck disable=SC2086  -- intentional word-splitting on $MARKET_ARGS
INSTANCE_ID=$(aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-groups "$SECURITY_GROUP" \
  --iam-instance-profile "Name=${INSTANCE_PROFILE}" \
  $MARKET_ARGS \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=120,VolumeType=gp3,DeleteOnTermination=true}' \
  --user-data "file://${USER_DATA}" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=krunch-tier3-${TEST_TAG}},{Key=Project,Value=krunch}]" \
  --query "Instances[0].InstanceId" --output text)

echo "  instance: $INSTANCE_ID"
echo "  ssh:      ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@\$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
echo "  watch:    aws ssm start-session --target $INSTANCE_ID --document-name AWS-StartInteractiveCommand --parameters command='tail -f /var/log/krunch-tier3.log'"

# ---------------------------------------------------------------------------
# 4. Poll for result
# ---------------------------------------------------------------------------
echo
echo "[3/5] Waiting for ${S3_BASE}/result.json (timeout 45 min for first build)..."
DEADLINE=$(( $(date +%s) + 2700 ))
while (( $(date +%s) < DEADLINE )); do
  if aws s3 cp --quiet "${S3_BASE}/result.json" "$RESULT_LOCAL" 2>/dev/null; then
    echo
    echo "[4/5] Result:"
    cat "$RESULT_LOCAL"
    break
  fi
  sleep 30
  printf "."
done

if [[ ! -f $RESULT_LOCAL ]]; then
  echo
  echo "FAIL no result after 45 min"
  echo "  Logs:  aws s3 cp ${S3_BASE}/setup.log -"
  echo "  Term:  aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
  exit 1
fi

# ---------------------------------------------------------------------------
# 5. Cleanup
# ---------------------------------------------------------------------------
echo
echo "[5/5] Cleanup..."
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
  --query "TerminatingInstances[0].CurrentState.Name" --output text 2>&1 | head -1 || true
aws s3 rm --quiet --recursive "${S3_BASE}" 2>&1 | tail -3 || true
rm -f "$REPO_TAR" "$USER_DATA"

ALL_PASS=$(/usr/bin/python3 -c "import json; print(json.load(open('$RESULT_LOCAL'))['all_gates_pass'])")
if [[ $ALL_PASS == "True" ]]; then
  echo
  echo "=== ALL GATES PASS ✓ ==="
  exit 0
else
  echo
  echo "=== GATE FAILURE — see $RESULT_LOCAL ==="
  exit 1
fi

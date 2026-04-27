# Krunch AWS CDK deployer

Deploys one g5.xlarge spot EC2 instance running the krunch Docker image.
Works on a fresh AWS account — uses the default VPC, no pre-existing infra needed.

## Prerequisites

- AWS CLI configured (`aws configure`)
- Node.js 18+
- `npm install` run in this directory

## Quickstart

```bash
# One-time per account/region
npx cdk bootstrap

# Deploy (~3 min: instance launch + Docker pull + model load)
npx cdk deploy

# Outputs:
#   KrunchEndpoint  = http://<public-ip>:8080
#   WatchLogs       = aws ssm start-session ... (stream init log)
```

## Run the roundtrip test

```bash
python3 ../../scripts/roundtrip_test.py \
  --url $(aws cloudformation describe-stacks \
    --stack-name KrunchStack \
    --query "Stacks[0].Outputs[?OutputKey=='KrunchEndpoint'].OutputValue" \
    --output text) \
  --file ../../data/spike6/wildchat_en_content.content.bin \
  --limit-mb 100
```

Expected results (A10G fp16):
- Ratio: ~0.11 (beats zstd by ~34%)
- Compress: ≥300 KB/s
- Decompress: ≥100 KB/s
- Byte-exact: PASS

## Tear down

```bash
npx cdk destroy
```

The instance is spot-priced: ~$1.01/hr on-demand, ~$0.30/hr spot.
Destroy when done to avoid idle charges.

## Customization

Edit `bin/app.ts`:

```typescript
new KrunchStack(app, "KrunchStack", {
  // Larger GPU for throughput testing
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.G5, ec2.InstanceSize.X2LARGE),

  // On-demand if spot availability is low in your region
  spot: false,

  // Add your IP to enable SSH (SSM works without this)
  sshAllowedCidr: "203.0.113.0/32",
});
```

## Multi-worker distributed test

To test N-worker parallel compression:

```bash
# Deploy N instances by running N stacks with different names
for i in 1 2 3; do
  npx cdk deploy --context instanceSuffix=$i KrunchStack-$i &
done
wait

# Then split your corpus across the N endpoints
```

Or spin up additional instances manually and point them at different
input chunks — each worker is fully independent.

## Logs

The init log streams to `/var/log/krunch-init.log`. Watch it via SSM:

```bash
aws ssm start-session \
  --target <InstanceId from Outputs> \
  --document-name AWS-StartInteractiveCommand \
  --parameters command="tail -f /var/log/krunch-init.log"
```

Sentinel lines to watch for:
- `KRUNCH_INIT_START` — instance booted
- `KRUNCH_READY` — server accepting requests
- `KRUNCH_INIT_DONE` — init script finished cleanly

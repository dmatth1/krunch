import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as iam from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";

export interface KrunchStackProps extends cdk.StackProps {
  /** Number of GPU worker instances. Default: 2 */
  workerCount?: number;
  /** GPU instance type per worker. Default: g5.xlarge (A10G, 16 GB VRAM) */
  workerInstanceType?: ec2.InstanceType;
  /** Use spot pricing for workers. Default: true */
  spot?: boolean;
  /** Docker image for workers. Default: ghcr.io/dmatth1/krunch:v1 */
  workerImage?: string;
  /** S3 bucket name the orchestrator + workers may read/write. Optional. */
  s3BucketName?: string;
  /** Allow SSH to orchestrator from this CIDR. Default: disabled (use SSM) */
  sshAllowedCidr?: string;
}

/**
 * Krunch v1 distributed deployment.
 *
 * Architecture:
 *   Internet → Orchestrator (t3.medium, public IP, port 8080)
 *                  ↓ fan-out via private IPs
 *           Worker-0  Worker-1  ... Worker-N  (g5.xlarge spot, port 8080)
 *
 * Orchestrator accepts /compress and /decompress.
 * Workers handle actual GPU compression; orchestrator splits + reassembles.
 *
 * Quickstart:
 *   npm install
 *   npx cdk bootstrap   # once per account/region
 *   npx cdk deploy
 *
 * Run roundtrip test:
 *   ENDPOINT=$(aws cloudformation describe-stacks --stack-name KrunchStack \
 *     --query "Stacks[0].Outputs[?OutputKey=='KrunchEndpoint'].OutputValue" \
 *     --output text)
 *   python3 ../../scripts/roundtrip_test.py --url $ENDPOINT \
 *     --file ../../data/spike6/wildchat_en_content.content.bin
 */
export class KrunchStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: KrunchStackProps = {}) {
    super(scope, id, props);

    const workerCount = props.workerCount ?? 2;
    const workerType =
      props.workerInstanceType ??
      ec2.InstanceType.of(ec2.InstanceClass.G5, ec2.InstanceSize.XLARGE);
    const useSpot = props.spot ?? true;
    const workerImage = props.workerImage ?? "ghcr.io/dmatth1/krunch:v1";
    const port = 8080;

    // ---------------------------------------------------------------------------
    // VPC — default VPC so a fresh account needs zero networking setup
    // ---------------------------------------------------------------------------
    const vpc = ec2.Vpc.fromLookup(this, "Vpc", { isDefault: true });

    // ---------------------------------------------------------------------------
    // IAM roles
    // ---------------------------------------------------------------------------
    const basePolicy = [
      iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSSMManagedInstanceCore"),
    ];

    const workerRole = new iam.Role(this, "WorkerRole", {
      assumedBy: new iam.ServicePrincipal("ec2.amazonaws.com"),
      managedPolicies: basePolicy,
    });

    const orchestratorRole = new iam.Role(this, "OrchestratorRole", {
      assumedBy: new iam.ServicePrincipal("ec2.amazonaws.com"),
      managedPolicies: basePolicy,
    });

    // Grant S3 access if a bucket is provided
    if (props.s3BucketName) {
      const s3Arn = `arn:aws:s3:::${props.s3BucketName}`;
      const s3Policy = new iam.PolicyStatement({
        actions: ["s3:GetObject", "s3:PutObject", "s3:DeleteObject",
                  "s3:HeadObject", "s3:ListBucket"],
        resources: [s3Arn, `${s3Arn}/*`],
      });
      workerRole.addToPolicy(s3Policy);
      orchestratorRole.addToPolicy(s3Policy);
    }

    // ---------------------------------------------------------------------------
    // Security groups
    // ---------------------------------------------------------------------------
    const workerSg = new ec2.SecurityGroup(this, "WorkerSg", {
      vpc,
      description: "krunch GPU workers — inbound from orchestrator only",
      allowAllOutbound: true,
    });

    const orchSg = new ec2.SecurityGroup(this, "OrchestratorSg", {
      vpc,
      description: "krunch orchestrator — public inbound on port 8080",
      allowAllOutbound: true,
    });

    // Orchestrator reachable from internet
    orchSg.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(port), "krunch API");

    // Workers only reachable from orchestrator
    workerSg.addIngressRule(orchSg, ec2.Port.tcp(port), "orchestrator fan-out");

    if (props.sshAllowedCidr) {
      orchSg.addIngressRule(
        ec2.Peer.ipv4(props.sshAllowedCidr), ec2.Port.tcp(22), "SSH"
      );
    }

    // ---------------------------------------------------------------------------
    // AMI — Deep Learning AMI: Docker + CUDA + nvidia-container-toolkit
    // ---------------------------------------------------------------------------
    const dlAmi = ec2.MachineImage.lookup({
      name: "Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04) *",
      owners: ["amazon"],
    });

    const ubuntuAmi = ec2.MachineImage.lookup({
      name: "ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*",
      owners: ["099720109477"], // Canonical
    });

    // ---------------------------------------------------------------------------
    // Workers
    // ---------------------------------------------------------------------------
    const workerInstances: ec2.Instance[] = [];

    for (let i = 0; i < workerCount; i++) {
      const workerUserData = ec2.UserData.forLinux();
      workerUserData.addCommands(
        "set -euo pipefail",
        `exec > /var/log/krunch-worker-${i}.log 2>&1`,
        `echo "WORKER_${i}_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"`,
        "for j in $(seq 1 30); do docker info && break || sleep 5; done",
        `docker pull ${workerImage}`,
        `docker run -d --restart=unless-stopped \\`,
        `  --gpus all --name krunch-worker \\`,
        `  -p ${port}:${port} \\`,
        `  -e RWKV_CUDA_ON=1 -e RWKV_JIT_ON=1 \\`,
        `  ${workerImage}`,
        `echo "WORKER_${i}_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"`,
      );

      const worker = new ec2.Instance(this, `Worker${i}`, {
        instanceType: workerType,
        machineImage: dlAmi,
        vpc,
        securityGroup: workerSg,
        role: workerRole,
        userData: workerUserData,
        blockDevices: [{
          deviceName: "/dev/sda1",
          volume: ec2.BlockDeviceVolume.ebs(100, {
            volumeType: ec2.EbsDeviceVolumeType.GP3,
          }),
        }],
      });

      if (useSpot) {
        (worker.node.defaultChild as ec2.CfnInstance).addPropertyOverride(
          "InstanceMarketOptions",
          { MarketType: "spot", SpotOptions: { SpotInstanceType: "one-time" } }
        );
      }

      workerInstances.push(worker);
    }

    // ---------------------------------------------------------------------------
    // Orchestrator (CPU — t3.medium, no GPU needed)
    // ---------------------------------------------------------------------------
    const workerUrls = workerInstances
      .map((w) => `http://${w.instancePrivateIp}:${port}`)
      .join(",");

    const orchUserData = ec2.UserData.forLinux();
    orchUserData.addCommands(
      "set -euo pipefail",
      "exec > /var/log/krunch-orchestrator.log 2>&1",
      "echo 'ORCH_START'",

      // Install deps (Ubuntu 22.04, no pre-installed Docker)
      "apt-get update -q",
      "apt-get install -y -q python3.11 python3-pip python3.11-venv curl",
      "python3.11 -m venv /opt/krunch-venv",
      "/opt/krunch-venv/bin/pip install -q fastapi uvicorn httpx boto3",

      // Write orchestrator launch script
      "mkdir -p /opt/krunch",
      `cat > /opt/krunch/start.sh << 'SCRIPT'
#!/bin/bash
export KRUNCH_WORKERS="${workerUrls}"
cd /opt/krunch
exec /opt/krunch-venv/bin/python -m uvicorn server.orchestrator:app \\
  --host 0.0.0.0 --port ${port} --workers 4
SCRIPT`,
      "chmod +x /opt/krunch/start.sh",

      // Pull server code from the Docker image (reuse the Python modules)
      "docker pull ghcr.io/dmatth1/krunch:v1 2>/dev/null || true",
      "docker create --name krunch-tmp ghcr.io/dmatth1/krunch:v1 2>/dev/null && \
       docker cp krunch-tmp:/app/server /opt/krunch/server && \
       docker rm krunch-tmp || \
       echo 'Docker not available, install server code manually'",

      // Systemd service
      `cat > /etc/systemd/system/krunch-orchestrator.service << 'UNIT'
[Unit]
Description=Krunch Orchestrator
After=network.target

[Service]
ExecStart=/opt/krunch/start.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT`,
      "systemctl daemon-reload",
      "systemctl enable --now krunch-orchestrator",

      // Wait for readyz
      `for i in $(seq 1 60); do curl -sf http://localhost:${port}/readyz && echo 'ORCH_READY' && break || sleep 5; done`,
      "echo 'ORCH_DONE'",
    );

    const orchestrator = new ec2.Instance(this, "Orchestrator", {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MEDIUM),
      machineImage: ubuntuAmi,
      vpc,
      securityGroup: orchSg,
      role: orchestratorRole,
      userData: orchUserData,
    });

    // Orchestrator waits for workers to be running (CDK dependency, not a runtime wait)
    workerInstances.forEach((w) => orchestrator.node.addDependency(w));

    // ---------------------------------------------------------------------------
    // Outputs
    // ---------------------------------------------------------------------------
    new cdk.CfnOutput(this, "KrunchEndpoint", {
      value: `http://${orchestrator.instancePublicIp}:${port}`,
      description: "Krunch API — POST /compress, /decompress",
    });

    new cdk.CfnOutput(this, "OrchestratorInstanceId", {
      value: orchestrator.instanceId,
      description: "Orchestrator instance ID (SSM access)",
    });

    new cdk.CfnOutput(this, "WorkerCount", {
      value: String(workerCount),
      description: "Number of GPU workers",
    });

    new cdk.CfnOutput(this, "WatchOrchestratorLogs", {
      value: `aws ssm start-session --target ${orchestrator.instanceId} --document-name AWS-StartInteractiveCommand --parameters command="journalctl -fu krunch-orchestrator"`,
    });
  }
}

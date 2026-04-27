import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as iam from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";

export interface KrunchStackProps extends cdk.StackProps {
  /** EC2 instance type. Must be GPU-capable. Default: g5.xlarge (A10G, 16 GB VRAM) */
  instanceType?: ec2.InstanceType;
  /** Use spot pricing. Default: true */
  spot?: boolean;
  /** Docker image to run. Default: ghcr.io/dmatth1/krunch:v1 */
  image?: string;
  /** Port the server listens on inside the container. Default: 8080 */
  containerPort?: number;
  /** Allow SSH access. Provide your public IP as "x.x.x.x/32". Default: disabled */
  sshAllowedCidr?: string;
}

/**
 * Krunch v1 reference deployer.
 *
 * Deploys one GPU-backed EC2 instance running the krunch Docker image.
 * Outputs: KrunchEndpoint — the URL to hit /compress and /decompress.
 *
 * Quickstart:
 *   npm install
 *   npx cdk bootstrap   # once per AWS account/region
 *   npx cdk deploy
 *
 * To run a roundtrip test after deploy:
 *   python3 ../../scripts/roundtrip_test.py \
 *     --url $(aws cloudformation describe-stacks \
 *       --stack-name KrunchStack \
 *       --query "Stacks[0].Outputs[?OutputKey=='KrunchEndpoint'].OutputValue" \
 *       --output text) \
 *     --file ../../data/spike6/wildchat_en_content.content.bin
 */
export class KrunchStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: KrunchStackProps = {}) {
    super(scope, id, props);

    const instanceType =
      props.instanceType ??
      ec2.InstanceType.of(ec2.InstanceClass.G5, ec2.InstanceSize.XLARGE);
    const useSpot = props.spot ?? true;
    const image = props.image ?? "ghcr.io/dmatth1/krunch:v1";
    const port = props.containerPort ?? 8080;

    // ---------------------------------------------------------------------------
    // VPC — use default VPC so a fresh account needs zero networking setup
    // ---------------------------------------------------------------------------
    const vpc = ec2.Vpc.fromLookup(this, "Vpc", { isDefault: true });

    // ---------------------------------------------------------------------------
    // Security group
    // ---------------------------------------------------------------------------
    const sg = new ec2.SecurityGroup(this, "KrunchSg", {
      vpc,
      description: "krunch compression server",
      allowAllOutbound: true,
    });

    sg.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(port),
      "krunch API"
    );

    if (props.sshAllowedCidr) {
      sg.addIngressRule(
        ec2.Peer.ipv4(props.sshAllowedCidr),
        ec2.Port.tcp(22),
        "SSH"
      );
    }

    // ---------------------------------------------------------------------------
    // IAM role (minimal — ECR pull is public for ghcr.io; add S3 if needed)
    // ---------------------------------------------------------------------------
    const role = new iam.Role(this, "KrunchRole", {
      assumedBy: new iam.ServicePrincipal("ec2.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "AmazonSSMManagedInstanceCore" // SSM Session Manager instead of SSH
        ),
      ],
    });

    // ---------------------------------------------------------------------------
    // AMI — Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04)
    // Docker + CUDA + nvidia-container-toolkit all preinstalled.
    // ---------------------------------------------------------------------------
    const ami = ec2.MachineImage.lookup({
      name: "Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04) *",
      owners: ["amazon"],
    });

    // ---------------------------------------------------------------------------
    // User data: pull image + run container
    // ---------------------------------------------------------------------------
    const userData = ec2.UserData.forLinux();
    userData.addCommands(
      "set -euo pipefail",
      "exec > /var/log/krunch-init.log 2>&1",

      // Signal start
      `echo "KRUNCH_INIT_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"`,

      // Wait for Docker daemon
      "for i in $(seq 1 30); do docker info && break || sleep 5; done",

      // Pull image (public ghcr.io, no auth needed)
      `docker pull ${image}`,

      // Run container
      `docker run -d --restart=unless-stopped \\`,
      `  --gpus all \\`,
      `  --name krunch \\`,
      `  -p ${port}:${port} \\`,
      `  -e RWKV_CUDA_ON=1 \\`,
      `  -e RWKV_JIT_ON=1 \\`,
      `  ${image}`,

      // Wait for readyz
      `for i in $(seq 1 60); do`,
      `  curl -sf http://localhost:${port}/readyz && echo "KRUNCH_READY $(date -u +%Y-%m-%dT%H:%M:%SZ)" && break`,
      `  sleep 5`,
      `done`,

      `echo "KRUNCH_INIT_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"`,
    );

    // ---------------------------------------------------------------------------
    // EC2 instance
    // Spot is requested via a launch template override on the underlying CfnInstance.
    // ec2.Instance doesn't expose spot natively; we patch the L1 after creation.
    // ---------------------------------------------------------------------------
    const instance = new ec2.Instance(this, "KrunchInstance", {
      instanceType,
      machineImage: ami,
      vpc,
      securityGroup: sg,
      role,
      userData,
      blockDevices: [
        {
          deviceName: "/dev/sda1",
          volume: ec2.BlockDeviceVolume.ebs(100, {
            volumeType: ec2.EbsDeviceVolumeType.GP3,
          }),
        },
      ],
    });

    // Patch spot request onto the underlying CfnInstance
    if (useSpot) {
      const cfn = instance.node.defaultChild as ec2.CfnInstance;
      (cfn as any).addPropertyOverride("InstanceMarketOptions", {
        MarketType: "spot",
        SpotOptions: { SpotInstanceType: "one-time" },
      });
    }

    // ---------------------------------------------------------------------------
    // Outputs
    // ---------------------------------------------------------------------------
    new cdk.CfnOutput(this, "KrunchEndpoint", {
      value: `http://${instance.instancePublicIp}:${port}`,
      description: "Krunch API endpoint — /compress, /decompress, /healthz",
    });

    new cdk.CfnOutput(this, "InstanceId", {
      value: instance.instanceId,
      description: "EC2 instance ID (for SSM Session Manager access)",
    });

    new cdk.CfnOutput(this, "WatchLogs", {
      value: `aws ssm start-session --target ${instance.instanceId} --document-name AWS-StartInteractiveCommand --parameters command="tail -f /var/log/krunch-init.log"`,
      description: "Stream init logs via SSM (no SSH key needed)",
    });
  }
}

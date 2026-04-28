import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as iam from "aws-cdk-lib/aws-iam";
import * as batch from "aws-cdk-lib/aws-batch";
import * as s3 from "aws-cdk-lib/aws-s3";
import { Construct } from "constructs";

export interface KrunchStackProps extends cdk.StackProps {
  /** Max number of concurrent GPU workers. Default: 10 */
  maxWorkers?: number;
  /** GPU instance type. Default: g5.xlarge (A10G, 16 GB VRAM) */
  instanceType?: ec2.InstanceType;
  /** Docker image. Default: ghcr.io/dmatth1/krunch:v1 */
  image?: string;
  /**
   * S3 bucket for temp parts + compressed output.
   * Created fresh if not provided.
   */
  s3BucketName?: string;
  /**
   * Use spot instances. Default: true.
   * Set to false for on-demand (higher cost, no interruption risk).
   * Both queues are always created; this controls the default priority.
   */
  spot?: boolean;
}

/**
 * Krunch v1 — AWS Batch deployment.
 *
 * Architecture:
 *   krunch submit --source s3://... --dest s3://...
 *     → Batch array job  (N compress tasks, each reads one byte range)
 *     → Batch single job (1 assemble task, stitches parts → final blob)
 *
 * No always-on orchestrator. Batch handles scheduling, spot retry,
 * and scaling to maxWorkers in parallel.
 *
 * Quickstart:
 *   npm install && npx cdk bootstrap && npx cdk deploy
 *
 *   python3 ../../scripts/krunch_cli.py submit \
 *     --source s3://my-bucket/logs/data.jsonl \
 *     --dest   s3://my-bucket/logs/data.krunch \
 *     --workers 4
 *
 * Cold-start note:
 *   First job on a fresh compute environment takes ~15 min (EC2 launch +
 *   Docker image pull). Subsequent jobs on warm instances: ~1-2 min.
 *   To eliminate cold pull time, bake the image into a custom AMI and
 *   set the imageId on the compute environment.
 */
export class KrunchStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: KrunchStackProps = {}) {
    super(scope, id, props);

    const maxWorkers = props.maxWorkers ?? 10;
    const instanceType =
      props.instanceType ??
      ec2.InstanceType.of(ec2.InstanceClass.G5, ec2.InstanceSize.XLARGE);
    const image = props.image ?? "ghcr.io/dmatth1/krunch:v1";
    const preferSpot = props.spot ?? true;

    // ---------------------------------------------------------------------------
    // VPC — default VPC, no NAT gateway needed (Batch uses public subnets)
    // ---------------------------------------------------------------------------
    const vpc = ec2.Vpc.fromLookup(this, "Vpc", { isDefault: true });

    // ---------------------------------------------------------------------------
    // S3 bucket — temp parts + compressed output
    // ---------------------------------------------------------------------------
    const bucket = props.s3BucketName
      ? s3.Bucket.fromBucketName(this, "Bucket", props.s3BucketName)
      : new s3.Bucket(this, "KrunchBucket", {
          removalPolicy: cdk.RemovalPolicy.RETAIN,
          lifecycleRules: [{
            // Auto-delete orphaned parts after 3 days
            prefix: "*.parts/",
            expiration: cdk.Duration.days(3),
          }],
        });

    // ---------------------------------------------------------------------------
    // IAM
    // ---------------------------------------------------------------------------
    const jobRole = new iam.Role(this, "JobRole", {
      assumedBy: new iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
      inlinePolicies: {
        S3Access: new iam.PolicyDocument({
          statements: [new iam.PolicyStatement({
            actions: ["s3:GetObject", "s3:PutObject", "s3:DeleteObject",
                      "s3:HeadObject"],
            resources: [`${bucket.bucketArn}/*`],
          }), new iam.PolicyStatement({
            actions: ["s3:ListBucket"],
            resources: [bucket.bucketArn],
          })],
        }),
      },
    });

    const instanceRole = new iam.Role(this, "InstanceRole", {
      assumedBy: new iam.ServicePrincipal("ec2.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "service-role/AmazonEC2ContainerServiceforEC2Role"
        ),
        iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSSMManagedInstanceCore"),
      ],
    });

    const instanceProfile = new iam.CfnInstanceProfile(this, "InstanceProfile", {
      roles: [instanceRole.roleName],
    });

    // ---------------------------------------------------------------------------
    // Security group
    // ---------------------------------------------------------------------------
    const sg = new ec2.SecurityGroup(this, "BatchSg", {
      vpc,
      description: "krunch Batch workers — outbound only",
      allowAllOutbound: true,
    });

    const sharedComputeProps = {
      type: "MANAGED" as const,
      state: "ENABLED",
    };
    const sharedResources = {
      instanceTypes: [instanceType.toString()],
      minvCpus: 0,
      maxvCpus: maxWorkers * 4, // g5.xlarge = 4 vCPUs
      subnets: vpc.publicSubnets.map((s) => s.subnetId),
      securityGroupIds: [sg.securityGroupId],
      instanceRole: instanceProfile.attrArn,
      // Override imageId with a custom AMI (pre-pulled image) to eliminate cold-pull time
      // imageId: "ami-0xxxxxxxxxxxxxxxxx",
    };

    // ---------------------------------------------------------------------------
    // Batch compute environments — spot (cheap) + on-demand (reliable)
    // Both created; job queue prefers whichever `spot` prop selects.
    // ---------------------------------------------------------------------------
    const spotEnv = new batch.CfnComputeEnvironment(this, "SpotEnv", {
      ...sharedComputeProps,
      computeResources: {
        ...sharedResources,
        type: "SPOT",
        allocationStrategy: "SPOT_CAPACITY_OPTIMIZED",
        bidPercentage: 60,
        tags: { Name: "krunch-spot-worker" },
      },
    });

    const onDemandEnv = new batch.CfnComputeEnvironment(this, "OnDemandEnv", {
      ...sharedComputeProps,
      computeResources: {
        ...sharedResources,
        type: "EC2",
        allocationStrategy: "BEST_FIT_PROGRESSIVE",
        tags: { Name: "krunch-ondemand-worker" },
      },
    });

    // ---------------------------------------------------------------------------
    // Job queue — preferred env first, fallback to the other
    // ---------------------------------------------------------------------------
    const [primary, fallback] = preferSpot
      ? [spotEnv, onDemandEnv]
      : [onDemandEnv, spotEnv];

    const jobQueue = new batch.CfnJobQueue(this, "JobQueue", {
      state: "ENABLED",
      priority: 10,
      computeEnvironmentOrder: [
        { order: 1, computeEnvironment: primary.ref },
        { order: 2, computeEnvironment: fallback.ref },
      ],
    });

    // ---------------------------------------------------------------------------
    // Job definitions
    // ---------------------------------------------------------------------------
    const containerProps = {
      image,
      command: ["job"],             // entrypoint.sh job mode
      jobRoleArn: jobRole.roleArn,
      resourceRequirements: [
        { type: "VCPU",   value: "4" },
        { type: "MEMORY", value: "16384" },
        { type: "GPU",    value: "1" },
      ],
      environment: [
        { name: "RWKV_CUDA_ON", value: "1" },
        { name: "RWKV_JIT_ON",  value: "1" },
      ],
      mountPoints: [],
      volumes: [],
    };

    const compressJobDef = new batch.CfnJobDefinition(this, "CompressJobDef", {
      type: "container",
      containerProperties: containerProps,
      retryStrategy: { attempts: 2 },  // retry once on spot interruption
      timeout: { attemptDurationSeconds: 3600 },
    });

    // Assemble job: CPU only, no GPU needed
    const assembleJobDef = new batch.CfnJobDefinition(this, "AssembleJobDef", {
      type: "container",
      containerProperties: {
        ...containerProps,
        resourceRequirements: [
          { type: "VCPU",   value: "2" },
          { type: "MEMORY", value: "8192" },
          // no GPU
        ],
      },
      retryStrategy: { attempts: 2 },
      timeout: { attemptDurationSeconds: 1800 },
    });

    // ---------------------------------------------------------------------------
    // Outputs (read by krunch_cli.py)
    // ---------------------------------------------------------------------------
    new cdk.CfnOutput(this, "JobQueueArn", {
      value: jobQueue.ref,
      description: "Batch job queue — pass to krunch submit --job-queue",
    });

    new cdk.CfnOutput(this, "CompressJobDef", {
      value: compressJobDef.ref,
      description: "Batch job definition for compress array tasks",
    });

    new cdk.CfnOutput(this, "AssembleJobDef", {
      value: assembleJobDef.ref,
      description: "Batch job definition for assemble task",
    });

    new cdk.CfnOutput(this, "BucketName", {
      value: bucket.bucketName,
      description: "S3 bucket for compressed output and temp parts",
    });

    new cdk.CfnOutput(this, "SubmitExample", {
      value: [
        "python3 scripts/krunch_cli.py submit",
        `  --source s3://${bucket.bucketName}/input/data.jsonl`,
        `  --dest   s3://${bucket.bucketName}/output/data.krunch`,
        "  --workers 4",
      ].join(" \\\n"),
      description: "Example krunch submit command",
    });
  }
}

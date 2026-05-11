import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as iam from "aws-cdk-lib/aws-iam";
import * as batch from "aws-cdk-lib/aws-batch";
import * as logs from "aws-cdk-lib/aws-logs";
import * as s3 from "aws-cdk-lib/aws-s3";
import { Construct } from "constructs";

export interface KrunchStackProps extends cdk.StackProps {
  /** Max number of concurrent GPU workers. Default: 4 (matches the
   * default fresh-account AWS service quota of 16 vCPU for On-Demand
   * G+VT instances in us-east-1; 4× g5.xlarge = 16 vCPU). Override
   * higher only if your AWS quota allows. */
  maxWorkers?: number;
  /** GPU instance type. Default: g5.xlarge (A10G, 16 GB VRAM).
   * If overriding, also set `vcpusPerInstance` to match (the default
   * 4 is correct for g5.xlarge / g6.xlarge / g6e.xlarge). */
  instanceType?: ec2.InstanceType;
  /** vCPUs per instance — used for ASG cap math. Default: 4 (xlarge). */
  vcpusPerInstance?: number;
  /** Docker image. Default: ghcr.io/dmatth1/krunch:latest */
  image?: string;
  /**
   * Optional pre-baked AMI (with the krunch image already pulled) to
   * eliminate the ~3.5 GB image-pull tax on cold-start. The AMI must
   * be Amazon-Linux-2 / ECS-AL2023-NVIDIA compatible (so the ECS
   * agent registers with the Batch CE) and must have the krunch
   * image present in the docker cache at AMI-build time.
   *
   * If unset, Batch picks the latest ECS GPU AMI and pulls the image
   * on first run. See the V1_PLAN backlog item "Custom AMI with
   * pre-pulled image".
   */
  imageId?: string;
  /**
   * S3 bucket for temp parts + compressed output.
   * Created fresh if not provided.
   */
  s3BucketName?: string;
  /**
   * CloudWatch retention for `/aws/batch/job` logs. Default: 30 days
   * (cuts indefinite log accumulation). Set to `INFINITE` only if
   * compliance/audit requires it.
   */
  logRetention?: logs.RetentionDays;
}

/**
 * Krunch v1 — AWS Batch deployment.
 *
 * Architecture:
 *   krunch plan --target aws-batch ... > job.json
 *   jq .main job.json     | aws batch submit-job ...   (N-task array, GPU)
 *   jq .finalize job.json | aws batch submit-job ...   (1-task CPU, dependsOn array)
 *
 * Same flow for --mode compress and --mode decompress; the rendered
 * spec sets KRUNCH_MODE per submission via containerOverrides.
 *
 * No always-on orchestrator. Batch handles scheduling and scaling
 * to maxWorkers in parallel.
 *
 * Quickstart:
 *   npm install && npx cdk bootstrap && npx cdk deploy
 *   # Then: krunch plan --target aws-batch ...  (see deploy/aws-cdk/README.md)
 *
 * Cold-start note:
 *   First job on a fresh CE: ~13.5 min wall (T4 measurement 2026-05-07)
 *   = EC2 launch + 3.5 GB image pull + model load + WKV-kernel JIT.
 *   Subsequent jobs on warm instances: ~30 s. Pass `imageId` to use a
 *   pre-baked AMI and skip the image-pull portion.
 */
export class KrunchStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: KrunchStackProps = {}) {
    super(scope, id, props);

    const maxWorkers = props.maxWorkers ?? 4;
    const instanceType =
      props.instanceType ??
      ec2.InstanceType.of(ec2.InstanceClass.G5, ec2.InstanceSize.XLARGE);
    const vcpusPerInstance = props.vcpusPerInstance ?? 4;
    const image = props.image ?? "ghcr.io/dmatth1/krunch:latest";
    const logRetention = props.logRetention ?? logs.RetentionDays.ONE_MONTH;

    // Stack-level tags propagate to all taggable resources for cost
    // allocation + identification in the Resource Groups console.
    cdk.Tags.of(this).add("Project", "krunch");
    cdk.Tags.of(this).add("ManagedBy", "cdk");

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
            // Auto-delete orphaned partial blobs after 3 days.
            //
            // krunch workers (krunch/job.py) tag every part upload
            // with `lifecycle=temp` (via krunch/url_io.py:write tags=).
            // S3 lifecycle's `prefix` filter is literal-prefix only —
            // a glob like `*.parts/` matches NOTHING. Using `tagFilters`
            // is the correct way to target the per-job `<output>.parts/<i>`
            // paths that have variable output prefixes.
            tagFilters: { lifecycle: "temp" },
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
      description: "krunch Batch workers - outbound only",
      allowAllOutbound: true,
    });

    const sharedComputeProps = {
      type: "MANAGED" as const,
      state: "ENABLED",
    };
    const sharedResources = {
      instanceTypes: [instanceType.toString()],
      minvCpus: 0,
      maxvCpus: maxWorkers * vcpusPerInstance,
      subnets: vpc.publicSubnets.map((s) => s.subnetId),
      securityGroupIds: [sg.securityGroupId],
      instanceRole: instanceProfile.attrArn,
      ...(props.imageId ? { imageId: props.imageId } : {}),
    };

    // ---------------------------------------------------------------------------
    // Batch compute environment — on-demand only. Spot was supported
    // earlier but g5.xlarge spot capacity is intermittent (us-east-1
    // recent windows have seen `instance-terminated-no-capacity`
    // reclaims mid-job), and the spot savings don't justify the
    // operational pain for the workloads krunch targets (one-shot
    // compress/decompress jobs, not long-running services).
    // ---------------------------------------------------------------------------
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
    // Job queue
    // ---------------------------------------------------------------------------
    const jobQueue = new batch.CfnJobQueue(this, "JobQueue", {
      state: "ENABLED",
      priority: 10,
      computeEnvironmentOrder: [
        { order: 1, computeEnvironment: onDemandEnv.ref },
      ],
    });

    // Cap CloudWatch retention on the Batch job log group. AWS Batch
    // creates `/aws/batch/job` lazily on first task; without explicit
    // retention it accumulates forever. We use `LogRetention` (a
    // Lambda-backed custom resource that calls PutRetentionPolicy)
    // rather than `new LogGroup(...)` because the log group may
    // already exist from prior Batch activity in the account — and
    // `LogGroup` creation would fail in that case. `LogRetention`
    // is idempotent and works whether the log group exists yet or
    // not, and survives `cdk destroy` (it just stops managing
    // retention; the group itself is owned by Batch).
    new logs.LogRetention(this, "BatchJobLogRetention", {
      logGroupName: "/aws/batch/job",
      retention: logRetention,
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
        // 14336 (14 GB) fits g5.xlarge after ECS+OS overhead. Setting
        // 16384 on a 16 GB instance trips MISCONFIGURATION:JOB_RESOURCE_REQUIREMENT.
        { type: "MEMORY", value: "14336" },
        { type: "GPU",    value: "1" },
      ],
      environment: [
        { name: "RWKV_CUDA_ON", value: "1" },
        { name: "RWKV_JIT_ON",  value: "1" },
      ],
      mountPoints: [],
      volumes: [],
    };

    // GPU job definitions — same container shape for both compress and
    // decompress. KRUNCH_MODE / KRUNCH_INPUT_URL / KRUNCH_OUTPUT_URL /
    // KRUNCH_INPUT_LEN / KRUNCH_PART_INDEX / KRUNCH_PART_COUNT are
    // injected per-submission via containerOverrides by `krunch plan`.
    const compressJobDef = new batch.CfnJobDefinition(this, "CompressJobDef", {
      type: "container",
      containerProperties: containerProps,
      retryStrategy: { attempts: 2 },  // retry once on transient ECS / image-pull failure
      timeout: { attemptDurationSeconds: 3600 },
    });

    const decompressJobDef = new batch.CfnJobDefinition(this, "DecompressJobDef", {
      type: "container",
      containerProperties: containerProps,
      retryStrategy: { attempts: 2 },
      timeout: { attemptDurationSeconds: 3600 },
    });

    // Finalize job: CPU only — stitches partial blobs into the final
    // output. Same image, no GPU. Used by both compress and decompress
    // flows (KRUNCH_FINALIZE_OF=compress|decompress set at submit time).
    const finalizeJobDef = new batch.CfnJobDefinition(this, "FinalizeJobDef", {
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
    // Outputs — consumed by `krunch plan --target aws-batch` (--queue
    // and --job-definition flags) and by tests/integration/batch.sh.
    // ---------------------------------------------------------------------------
    new cdk.CfnOutput(this, "JobQueueArn", {
      value: jobQueue.ref,
      description: "Batch job queue — pass to krunch plan --queue",
    });

    new cdk.CfnOutput(this, "CompressJobDefOutput", {
      value: compressJobDef.ref,
      description: "GPU job definition for compress array tasks",
    });

    new cdk.CfnOutput(this, "DecompressJobDefOutput", {
      value: decompressJobDef.ref,
      description: "GPU job definition for decompress array tasks",
    });

    new cdk.CfnOutput(this, "FinalizeJobDefOutput", {
      value: finalizeJobDef.ref,
      description: "CPU job definition for the finalize stitcher (compress + decompress)",
    });

    new cdk.CfnOutput(this, "BucketName", {
      value: bucket.bucketName,
      description: "S3 bucket for compressed output and temp parts",
    });

    new cdk.CfnOutput(this, "PlanExample", {
      value: [
        "krunch plan --target aws-batch --mode compress",
        `  --source s3://${bucket.bucketName}/input/data.jsonl`,
        `  --dest   s3://${bucket.bucketName}/output/data.krunch`,
        "  --workers 4 --input-len $(stat -c%s data.jsonl)",
        "  --queue <JobQueueArn> --job-definition <CompressJobDefOutput>",
        "  > job.json",
      ].join(" \\\n"),
      description: "Example krunch plan invocation; submit job.main then job.finalize via aws batch submit-job",
    });
  }
}

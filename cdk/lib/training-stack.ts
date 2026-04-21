import * as cdk from "aws-cdk-lib";
import * as batch from "aws-cdk-lib/aws-batch";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as ecr_assets from "aws-cdk-lib/aws-ecr-assets";
import * as events from "aws-cdk-lib/aws-events";
import * as targets from "aws-cdk-lib/aws-events-targets";
import * as iam from "aws-cdk-lib/aws-iam";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as sqs from "aws-cdk-lib/aws-sqs";
import * as lambdaEventSources from "aws-cdk-lib/aws-lambda-event-sources";
import { Construct } from "constructs";
import { NodejsFunction } from "aws-cdk-lib/aws-lambda-nodejs";
import { Runtime, Architecture } from "aws-cdk-lib/aws-lambda";
import { EnvName, named } from "./shared/constants";

interface TrainingStackProps extends cdk.StackProps {
  envName: EnvName;
  bucket: s3.IBucket;
  datasetsTable: dynamodb.ITable;
  modelVersionsTable: dynamodb.ITable;
  trainingSubmitQueue: sqs.IQueue;
  compressionQueue: sqs.IQueue;
}

/**
 * AWS Batch compute environment + job queue + job definition for
 * running per-customer model training on EC2 spot instances.
 *
 * Also wires:
 *   - training-launcher Lambda (SQS → Batch.submitJob)
 *   - training-complete Lambda (EventBridge Batch state change → DDB)
 */
export class TrainingStack extends cdk.Stack {
  public readonly jobDefinition: batch.EcsJobDefinition;
  public readonly jobQueue: batch.JobQueue;

  constructor(scope: Construct, id: string, props: TrainingStackProps) {
    super(scope, id, props);

    // -----------------------------------------------------------------
    // VPC for Batch compute env (simple default; one AZ, no NAT).
    // Training jobs pull from S3 via gateway endpoint — no internet
    // egress needed for inference data.
    // -----------------------------------------------------------------
    const vpc = new ec2.Vpc(this, "BatchVpc", {
      // Spot g5.xlarge is capacity-constrained in us-east-1; give Batch
      // more AZs to shop across. Bumped to 4 after UnfulfillableCapacity
      // in the initial 2-AZ run.
      maxAzs: 4,
      natGateways: 0,
      subnetConfiguration: [
        {
          name: "public",
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,
        },
      ],
    });
    vpc.addGatewayEndpoint("S3Endpoint", {
      service: ec2.GatewayVpcEndpointAwsService.S3,
    });

    // -----------------------------------------------------------------
    // Container image — bundles the Python training scripts.
    // Built from docker/training/Dockerfile; assumes that Dockerfile
    // COPYs in the repo-level scripts/ directory.
    // -----------------------------------------------------------------
    const trainingImage = new ecr_assets.DockerImageAsset(this, "TrainingImage", {
      directory: "../", // repo root — Dockerfile references scripts/ + vendor/
      file: "cdk/docker/training/Dockerfile",
      platform: ecr_assets.Platform.LINUX_AMD64,
    });

    // -----------------------------------------------------------------
    // Batch compute environment — spot EC2 g5.xlarge (1× NVIDIA A10G
    // 24 GB VRAM, 4 vCPU, 16 GB RAM). Chosen because:
    //   · 200K-param training on GPU is ~50× faster than CPU
    //   · A10G has headroom for up to ~3-4B param models later
    //   · Spot pricing ~$0.25/hr vs on-demand ~$1/hr
    //
    // maxvCpus=8 allows 2 concurrent g5.xlarge instances, matching the
    // default AWS G/VT spot instance quota of 8 vCPUs. Bump the quota
    // before bumping maxvCpus here.
    //
    // Setting images to a GPU-optimized ECS AMI is handled automatically
    // by Batch when the compute env contains G-class instances.
    // -----------------------------------------------------------------
    const computeEnv = new batch.ManagedEc2EcsComputeEnvironment(
      this,
      "BatchCompute",
      {
        computeEnvironmentName: named(props.envName, "batch-compute"),
        vpc,
        vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
        minvCpus: 0,
        maxvCpus: 8,
        // Spot g5.xlarge was UnfulfillableCapacity across 4 AZs on
        // 2026-04-21 — the spike switched to ON_DEMAND to unblock. Flip
        // back to spot=true for production runs where we can tolerate
        // queueing for spot capacity.
        spot: false,
        instanceTypes: [
          ec2.InstanceType.of(ec2.InstanceClass.G5, ec2.InstanceSize.XLARGE),
          ec2.InstanceType.of(ec2.InstanceClass.G5, ec2.InstanceSize.XLARGE2),
          ec2.InstanceType.of(ec2.InstanceClass.G6, ec2.InstanceSize.XLARGE),
        ],
        useOptimalInstanceClasses: false,
        updateToLatestImageVersion: true,
      },
    );

    this.jobQueue = new batch.JobQueue(this, "TrainingJobQueue", {
      jobQueueName: named(props.envName, "training-queue"),
      priority: 1,
      computeEnvironments: [
        { computeEnvironment: computeEnv, order: 1 },
      ],
    });

    // -----------------------------------------------------------------
    // Job role — grants the training container access to S3 (read raw,
    // write model + metadata) and DynamoDB (update model_versions).
    // -----------------------------------------------------------------
    const jobRole = new iam.Role(this, "TrainingJobRole", {
      assumedBy: new iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
    });
    props.bucket.grantReadWrite(jobRole);
    props.modelVersionsTable.grantReadWriteData(jobRole);

    this.jobDefinition = new batch.EcsJobDefinition(this, "TrainingJobDef", {
      jobDefinitionName: named(props.envName, "training-jobdef"),
      container: new batch.EcsEc2ContainerDefinition(this, "TrainingContainer", {
        image: ecs.ContainerImage.fromDockerImageAsset(trainingImage),
        // g5.xlarge = 4 vCPU, 16 GB RAM, 1 A10G. Reserve enough for
        // data loading + torch overhead but leave some headroom.
        cpu: 4,
        memory: cdk.Size.gibibytes(14),
        gpu: 1,
        jobRole,
      }),
      retryAttempts: 1,
      timeout: cdk.Duration.hours(8),
    });

    // -----------------------------------------------------------------
    // Launcher Lambda — reads training-submit queue, submits Batch job.
    // -----------------------------------------------------------------
    const launcherFn = new NodejsFunction(this, "TrainingLauncherFn", {
      functionName: named(props.envName, "training-launcher-fn"),
      entry: "src/handlers/training-launcher.ts",
      handler: "handler",
      runtime: Runtime.NODEJS_20_X,
      architecture: Architecture.ARM_64,
      timeout: cdk.Duration.seconds(60),
      memorySize: 256,
      environment: {
        BATCH_JOB_DEFINITION: this.jobDefinition.jobDefinitionArn,
        BATCH_JOB_QUEUE: this.jobQueue.jobQueueArn,
        BUCKET_NAME: props.bucket.bucketName,
        DATASETS_TABLE_NAME: props.datasetsTable.tableName,
        NODE_OPTIONS: "--enable-source-maps",
      },
      bundling: { minify: false, sourceMap: true, format: undefined },
    });

    launcherFn.addEventSource(
      new lambdaEventSources.SqsEventSource(props.trainingSubmitQueue, {
        batchSize: 10,
        maxBatchingWindow: cdk.Duration.seconds(5),
      }),
    );
    launcherFn.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["batch:SubmitJob"],
        resources: [
          this.jobQueue.jobQueueArn,
          this.jobDefinition.jobDefinitionArn,
        ],
      }),
    );
    props.datasetsTable.grantReadWriteData(launcherFn);

    // -----------------------------------------------------------------
    // Completion Lambda — fires on Batch job state change.
    // -----------------------------------------------------------------
    const completionFn = new NodejsFunction(this, "TrainingCompleteFn", {
      functionName: named(props.envName, "training-complete-fn"),
      entry: "src/handlers/training-complete.ts",
      handler: "handler",
      runtime: Runtime.NODEJS_20_X,
      architecture: Architecture.ARM_64,
      timeout: cdk.Duration.minutes(5),
      memorySize: 512,
      environment: {
        BUCKET_NAME: props.bucket.bucketName,
        DATASETS_TABLE_NAME: props.datasetsTable.tableName,
        MODEL_VERSIONS_TABLE_NAME: props.modelVersionsTable.tableName,
        COMPRESSION_QUEUE_URL: props.compressionQueue.queueUrl,
        NODE_OPTIONS: "--enable-source-maps",
      },
      bundling: { minify: false, sourceMap: true, format: undefined },
    });
    props.bucket.grantRead(completionFn);
    props.datasetsTable.grantReadWriteData(completionFn);
    props.modelVersionsTable.grantReadWriteData(completionFn);
    props.compressionQueue.grantSendMessages(completionFn);

    new events.Rule(this, "BatchCompletionRule", {
      ruleName: named(props.envName, "batch-completion"),
      eventPattern: {
        source: ["aws.batch"],
        detailType: ["Batch Job State Change"],
        detail: {
          status: ["SUCCEEDED", "FAILED"],
          jobQueue: [this.jobQueue.jobQueueArn],
        },
      },
      targets: [new targets.LambdaFunction(completionFn)],
    });

    new cdk.CfnOutput(this, "JobQueueArn", {
      value: this.jobQueue.jobQueueArn,
    });
    new cdk.CfnOutput(this, "JobDefinitionArn", {
      value: this.jobDefinition.jobDefinitionArn,
    });
  }
}

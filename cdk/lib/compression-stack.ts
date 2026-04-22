import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as ecs_patterns from "aws-cdk-lib/aws-ecs-patterns";
import * as ecr_assets from "aws-cdk-lib/aws-ecr-assets";
import * as logs from "aws-cdk-lib/aws-logs";
import * as iam from "aws-cdk-lib/aws-iam";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as sqs from "aws-cdk-lib/aws-sqs";
import { Construct } from "constructs";
import { EnvName, named } from "./shared/constants";

interface CompressionStackProps extends cdk.StackProps {
  envName: EnvName;
  bucket: s3.IBucket;
  datasetsTable: dynamodb.ITable;
  modelVersionsTable: dynamodb.ITable;
  compressionQueue: sqs.IQueue;
}

/**
 * ECS Fargate task for the compression worker.
 *
 * The worker polls the compression SQS queue directly (long-poll). For
 * each message, it downloads the raw NDJSON from S3, fetches the
 * per-dataset model, compresses using the l3tc-rust binary, writes
 * the compressed blob back to S3 under compressed/, deletes the raw
 * object, and updates DynamoDB byte counters.
 *
 * MVP: run as a permanent single-task service with QueueProcessingFargateService,
 * which auto-scales between minTaskCount and maxTaskCount based on
 * queue depth. Set min=0, max=4 to pay ~nothing when idle.
 */
export class CompressionStack extends cdk.Stack {
  public readonly service: ecs_patterns.QueueProcessingFargateService;

  constructor(scope: Construct, id: string, props: CompressionStackProps) {
    super(scope, id, props);

    const vpc = new ec2.Vpc(this, "CompressionVpc", {
      maxAzs: 2,
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

    const cluster = new ecs.Cluster(this, "CompressionCluster", {
      clusterName: named(props.envName, "compression-cluster"),
      vpc,
    });

    // -----------------------------------------------------------------
    // Compression container image — wraps the l3tc-rust binary.
    // -----------------------------------------------------------------
    const image = new ecr_assets.DockerImageAsset(this, "CompressionImage", {
      directory: "../", // repo root
      file: "cdk/docker/compression/Dockerfile",
      platform: ecr_assets.Platform.LINUX_AMD64,
    });

    this.service = new ecs_patterns.QueueProcessingFargateService(
      this,
      "CompressionService",
      {
        serviceName: named(props.envName, "compression-service"),
        cluster,
        queue: props.compressionQueue as sqs.Queue,
        image: ecs.ContainerImage.fromDockerImageAsset(image),
        cpu: 2048, // 2 vCPU — zstd-22 is single-threaded but this gives the OS room and cuts compression walltime by ~30% in practice
        memoryLimitMiB: 4096,
        // min=1 on purpose: the default queue-depth scale-in metric
        // (ApproximateNumberOfMessagesVisible) drops to 0 the moment
        // a worker pulls a message (message becomes in-flight), which
        // triggered scale-in and killed compression mid-job during
        // Spike 1. $0.08/hr idle is cheap enough for pre-MVP; in
        // prod we'd use a MathExpression that sums visible + in-flight
        // and only scales to 0 when both are 0. See PRODUCTION_TODO
        // item for the proper fix.
        minScalingCapacity: 1,
        maxScalingCapacity: 4,
        // VPC has no NAT gateway, so tasks need public IPs to reach ECR / S3 over the internet.
        assignPublicIp: true,
        enableLogging: true,
        logDriver: ecs.LogDrivers.awsLogs({
          streamPrefix: named(props.envName, "compression"),
          logRetention: logs.RetentionDays.ONE_WEEK,
        }),
        environment: {
          BUCKET_NAME: props.bucket.bucketName,
          DATASETS_TABLE_NAME: props.datasetsTable.tableName,
          MODEL_VERSIONS_TABLE_NAME: props.modelVersionsTable.tableName,
        },
        scalingSteps: [
          { upper: 0, change: 0 },   // queue empty → no scale-in beyond the minimum (min=1 already)
          { lower: 1, change: +1 },
          { lower: 10, change: +2 },
        ],
      },
    );

    props.bucket.grantReadWrite(this.service.taskDefinition.taskRole);
    props.datasetsTable.grantReadWriteData(
      this.service.taskDefinition.taskRole,
    );
    props.modelVersionsTable.grantReadData(this.service.taskDefinition.taskRole);

    new cdk.CfnOutput(this, "CompressionServiceArn", {
      value: this.service.service.serviceArn,
    });
  }
}

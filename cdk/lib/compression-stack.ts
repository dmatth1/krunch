import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as ecs_patterns from "aws-cdk-lib/aws-ecs-patterns";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as ecr_assets from "aws-cdk-lib/aws-ecr-assets";
import * as cloudwatch from "aws-cdk-lib/aws-cloudwatch";
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
      // Container Insights publishes RunningTaskCount +
      // CpuUtilization + MemoryUtilization under
      // ECS/ContainerInsights. The dashboard below consumes the
      // task-count metric. ~$0.01/hr per task monitored, negligible
      // at current scale.
      containerInsightsV2: ecs.ContainerInsights.ENABLED,
    });

    // -----------------------------------------------------------------
    // Compression container image — wraps the l3tc-rust binary.
    //
    // Two modes:
    //   - `cdk deploy --context compressImageTag=<tag>` references a
    //     pre-built image pushed to the CDK assets ECR repo by
    //     CodeBuild (see buildspec.yml). Preferred for prod because
    //     CI is a clean, validated linux/amd64 build environment.
    //   - Without the context value, CDK rebuilds the Dockerfile
    //     locally (slow on M-series via buildx cross-compile, but
    //     useful for iterating on the Dockerfile without going
    //     through CodeBuild).
    // -----------------------------------------------------------------
    const compressImageTag = this.node.tryGetContext(
      "compressImageTag",
    ) as string | undefined;
    let image: ecs.ContainerImage;
    if (compressImageTag) {
      const repoName = `cdk-hnb659fds-container-assets-${cdk.Stack.of(this).account}-${cdk.Stack.of(this).region}`;
      const repo = ecr.Repository.fromRepositoryName(
        this,
        "CompressionRepo",
        repoName,
      );
      image = ecs.ContainerImage.fromEcrRepository(repo, compressImageTag);
    } else {
      const asset = new ecr_assets.DockerImageAsset(this, "CompressionImage", {
        directory: "../", // repo root
        file: "cdk/docker/compression/Dockerfile",
        platform: ecr_assets.Platform.LINUX_AMD64,
      });
      image = ecs.ContainerImage.fromDockerImageAsset(asset);
    }

    this.service = new ecs_patterns.QueueProcessingFargateService(
      this,
      "CompressionService",
      {
        serviceName: named(props.envName, "compression-service"),
        cluster,
        queue: props.compressionQueue as sqs.Queue,
        image,
        cpu: 2048, // 2 vCPU — zstd-22 is single-threaded but this gives the OS room and cuts compression walltime by ~30% in practice
        memoryLimitMiB: 4096,
        // Graviton ARM64 Fargate. The l3tc-rust neural codec uses
        // the `matrixmultiply` crate's NEON SIMD path; on x86 Fargate
        // we saw ~10 min wall-clock on a 5 MB / 200K-param hybrid run,
        // vs ~30 s on M-series (both NEON). Graviton recovers that
        // gap AND costs ~20% less than x86 at the same spec. The
        // ECR image must be built linux/arm64 — buildspec.yml passes
        // the flag and uses ARM_CONTAINER CodeBuild compute.
        runtimePlatform: {
          cpuArchitecture: ecs.CpuArchitecture.ARM64,
          operatingSystemFamily: ecs.OperatingSystemFamily.LINUX,
        },
        // min=0 is now safe: the worker calls the ECS task-protection
        // endpoint (PUT $ECS_AGENT_URI/task-protection/v1/state) with
        // ProtectionEnabled=true for the lifetime of each SQS message,
        // and clears it on completion. ECS refuses scale-in on
        // protected tasks regardless of what the auto-scaler decides,
        // so the old Visible-drops-to-0-mid-job bug can't kill work
        // in flight any more. The scaler goes back to the simple
        // "Visible > 0 → scale up; Visible = 0 → scale toward min"
        // pattern, but with the protection flag gating actual
        // termination. $0 idle, no custom metric.
        //
        // See AWS docs: Protect your Amazon ECS tasks from being
        // terminated by scale-in events.
        minScalingCapacity: 0,
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

    // Task scale-in protection (set/cleared by the worker around
    // each SQS message). The ECS agent proxies these calls but still
    // requires the task's own IAM role to carry the two permissions;
    // without them the agent returns HTTP 400 and the worker can't
    // protect in-flight work from auto-scaler-driven termination.
    this.service.taskDefinition.taskRole.addToPrincipalPolicy(
      new iam.PolicyStatement({
        actions: [
          "ecs:UpdateTaskProtection",
          "ecs:GetTaskProtection",
        ],
        // Scope to the compression cluster's tasks only. We use the
        // cluster's ARN substring since the task ARN isn't knowable
        // at synth time.
        resources: [
          `arn:aws:ecs:${cdk.Stack.of(this).region}:${cdk.Stack.of(this).account}:task/${cluster.clusterName}/*`,
        ],
      }),
    );

    new cdk.CfnOutput(this, "CompressionServiceArn", {
      value: this.service.service.serviceArn,
    });

    // -----------------------------------------------------------------
    // CloudWatch dashboard
    //
    // The worker emits EMF records to stdout per-run (see
    // emit_compression_metrics in compression_worker.py). Amazon
    // CloudWatch Logs auto-parses the embedded metric blocks into
    // Metric data under namespace `Krunch/Hybrid` with
    // dimensions [CustomerId, DatasetId, Env] (main record) and
    // [CustomerId, DatasetId, Env, Codec] (per-codec record). Both
    // hybrid and zstd-fallback runs populate this — fallback shows
    // up as Codec="zstd_fallback" with SavingsVsZstdPct=0.
    //
    // Widgets are aggregated (Average / Sum) across all customers;
    // drill-down per customer is done from the "Metrics" explorer
    // by filtering on the CustomerId dimension.
    // -----------------------------------------------------------------
    const NS = "Krunch/Hybrid";
    const periodMin = cdk.Duration.minutes(1);

    const mainDims: cloudwatch.DimensionsMap = {}; // aggregate across all
    const mkMetric = (name: string, statistic: string): cloudwatch.Metric =>
      new cloudwatch.Metric({
        namespace: NS,
        metricName: name,
        statistic,
        period: periodMin,
        dimensionsMap: mainDims,
      });

    const ratio = mkMetric("Ratio", cloudwatch.Stats.AVERAGE);
    const savings = mkMetric("SavingsVsZstdPct", cloudwatch.Stats.AVERAGE);
    const throughput = mkMetric("ThroughputMBps", cloudwatch.Stats.AVERAGE);
    const safetyNet = mkMetric(
      "SafetyNetSubstitutions",
      cloudwatch.Stats.SUM,
    );
    const bytesIn = mkMetric("BytesIn", cloudwatch.Stats.SUM);
    const bytesOut = mkMetric("BytesOut", cloudwatch.Stats.SUM);

    // Per-codec stacked bytes: one series per known codec. We hardcode
    // the tag set here to keep the dashboard schema stable when a
    // codec produces zero chunks for a period (otherwise the series
    // would disappear from the chart mid-session and the user would
    // assume the codec is "broken").
    const KNOWN_CODECS = [
      "neural",
      "bzip3",
      "zstd",
      "zstd_dict",
      "lz4",
      "passthrough",
      "brotli_dict",
      "clp",
      "zstd_fallback",
    ];
    const bytesByCodecSeries = KNOWN_CODECS.map(
      (c) =>
        new cloudwatch.Metric({
          namespace: NS,
          metricName: "BytesByCodec",
          statistic: cloudwatch.Stats.SUM,
          period: periodMin,
          dimensionsMap: { Codec: c },
          label: c,
        }),
    );
    const chunksByCodecSeries = KNOWN_CODECS.map(
      (c) =>
        new cloudwatch.Metric({
          namespace: NS,
          metricName: "ChunksByCodec",
          statistic: cloudwatch.Stats.SUM,
          period: periodMin,
          dimensionsMap: { Codec: c },
          label: c,
        }),
    );

    // Operational metrics: queue depth + running task count. Living
    // in the same dashboard means the oncall can see "0 tasks running
    // + 5 visible messages = scale-up hasn't fired yet" in one glance.
    const queueVisible = new cloudwatch.Metric({
      namespace: "AWS/SQS",
      metricName: "ApproximateNumberOfMessagesVisible",
      statistic: cloudwatch.Stats.AVERAGE,
      period: periodMin,
      dimensionsMap: { QueueName: (props.compressionQueue as sqs.Queue).queueName },
      label: "visible",
    });
    const queueInFlight = new cloudwatch.Metric({
      namespace: "AWS/SQS",
      metricName: "ApproximateNumberOfMessagesNotVisible",
      statistic: cloudwatch.Stats.AVERAGE,
      period: periodMin,
      dimensionsMap: { QueueName: (props.compressionQueue as sqs.Queue).queueName },
      label: "in-flight",
    });
    const runningTasks = new cloudwatch.Metric({
      namespace: "ECS/ContainerInsights",
      metricName: "RunningTaskCount",
      statistic: cloudwatch.Stats.AVERAGE,
      period: periodMin,
      dimensionsMap: {
        ClusterName: cluster.clusterName,
        ServiceName: this.service.service.serviceName,
      },
      label: "running tasks",
    });

    const dashboard = new cloudwatch.Dashboard(this, "CompressionDashboard", {
      dashboardName: named(props.envName, "compression"),
      defaultInterval: cdk.Duration.hours(3),
    });

    dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: "Compression ratio (lower = better)",
        left: [ratio],
        leftYAxis: { min: 0, max: 1 },
        width: 12,
        height: 6,
      }),
      new cloudwatch.GraphWidget({
        title: "Savings vs zstd-22 per-chunk shadow (%)",
        left: [savings],
        leftYAxis: { label: "percent" },
        width: 12,
        height: 6,
      }),
    );
    dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: "Compression throughput (MB/s, per-task)",
        left: [throughput],
        width: 12,
        height: 6,
      }),
      new cloudwatch.SingleValueWidget({
        title: "Safety-net substitutions (sum)",
        metrics: [safetyNet],
        width: 12,
        height: 6,
      }),
    );
    dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: "Bytes out by codec (stacked)",
        left: bytesByCodecSeries,
        stacked: true,
        width: 12,
        height: 6,
      }),
      new cloudwatch.GraphWidget({
        title: "Chunks by codec (stacked)",
        left: chunksByCodecSeries,
        stacked: true,
        width: 12,
        height: 6,
      }),
    );
    dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: "Total traffic (BytesIn vs BytesOut)",
        left: [bytesIn, bytesOut],
        width: 12,
        height: 6,
      }),
      new cloudwatch.GraphWidget({
        title: "Queue depth + running tasks",
        left: [queueVisible, queueInFlight],
        right: [runningTasks],
        width: 12,
        height: 6,
      }),
    );

    new cdk.CfnOutput(this, "DashboardUrl", {
      value: `https://${cdk.Stack.of(this).region}.console.aws.amazon.com/cloudwatch/home?region=${cdk.Stack.of(this).region}#dashboards:name=${dashboard.dashboardName}`,
    });
  }
}

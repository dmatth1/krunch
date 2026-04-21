import * as cdk from "aws-cdk-lib";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as events from "aws-cdk-lib/aws-events";
import * as targets from "aws-cdk-lib/aws-events-targets";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as sqs from "aws-cdk-lib/aws-sqs";
import { Construct } from "constructs";
import { NodejsFunction } from "aws-cdk-lib/aws-lambda-nodejs";
import { Runtime, Architecture } from "aws-cdk-lib/aws-lambda";
import { EnvName, named } from "./shared/constants";

interface IngestStackProps extends cdk.StackProps {
  envName: EnvName;
  bucket: s3.IBucket;
  datasetsTable: dynamodb.ITable;
  trainingSubmitQueue: sqs.IQueue;
  compressionQueue: sqs.IQueue;
}

/**
 * S3 ObjectCreated (delivered via EventBridge) → Ingest Lambda.
 *
 * We use EventBridge rather than direct S3 event notifications because
 * the notification-configuration approach creates a cross-stack
 * dependency cycle: the bucket (in StorageStack) would need to
 * reference the Lambda ARN (in this stack), while the Lambda already
 * references the bucket for IAM grants. EventBridge breaks the cycle
 * — the bucket emits events to the default event bus without
 * knowing about specific consumers.
 *
 * The bucket has `eventBridgeEnabled: true` set in StorageStack for this.
 *
 * The Lambda still owns the concurrency guard: a conditional DDB
 * update that flips status from awaiting_corpus → training atomically.
 * See docs/SERVICE_ARCHITECTURE.md for the state machine.
 */
export class IngestStack extends cdk.Stack {
  public readonly ingestFn: NodejsFunction;

  constructor(scope: Construct, id: string, props: IngestStackProps) {
    super(scope, id, props);

    this.ingestFn = new NodejsFunction(this, "IngestFn", {
      functionName: named(props.envName, "ingest-fn"),
      entry: "src/handlers/ingest.ts",
      handler: "handler",
      runtime: Runtime.NODEJS_20_X,
      architecture: Architecture.ARM_64,
      timeout: cdk.Duration.seconds(60),
      memorySize: 512,
      environment: {
        BUCKET_NAME: props.bucket.bucketName,
        DATASETS_TABLE_NAME: props.datasetsTable.tableName,
        TRAINING_SUBMIT_QUEUE_URL: props.trainingSubmitQueue.queueUrl,
        COMPRESSION_QUEUE_URL: props.compressionQueue.queueUrl,
        NODE_OPTIONS: "--enable-source-maps",
      },
      bundling: { minify: false, sourceMap: true, format: undefined },
    });

    props.bucket.grantRead(this.ingestFn);
    props.datasetsTable.grantReadWriteData(this.ingestFn);
    props.trainingSubmitQueue.grantSendMessages(this.ingestFn);
    props.compressionQueue.grantSendMessages(this.ingestFn);

    // EventBridge rule: match S3 "Object Created" events on our bucket
    // for .ndjson suffix keys. The S3 EventBridge integration emits
    // events to the default bus in this region.
    new events.Rule(this, "S3RawObjectRule", {
      ruleName: named(props.envName, "raw-object-created"),
      eventPattern: {
        source: ["aws.s3"],
        detailType: ["Object Created"],
        detail: {
          bucket: { name: [props.bucket.bucketName] },
          object: {
            key: [{ suffix: ".ndjson" }],
          },
        },
      },
      targets: [
        new targets.LambdaFunction(this.ingestFn, {
          retryAttempts: 3,
        }),
      ],
    });
  }
}

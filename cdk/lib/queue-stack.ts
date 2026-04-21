import * as cdk from "aws-cdk-lib";
import * as sqs from "aws-cdk-lib/aws-sqs";
import { Construct } from "constructs";
import { EnvName, named } from "./shared/constants";

interface QueueStackProps extends cdk.StackProps {
  envName: EnvName;
}

/**
 * All SQS queues used for inter-stage decoupling, each with a DLQ.
 *
 * - trainingSubmitQueue: ingest Lambda → training launcher Lambda
 * - compressionQueue:    completion/ingest Lambda → Fargate compression worker
 *
 * Training completion notifications come via EventBridge → Lambda
 * (no dedicated queue; handled in TrainingStack).
 */
export class QueueStack extends cdk.Stack {
  public readonly trainingSubmitQueue: sqs.Queue;
  public readonly trainingSubmitDlq: sqs.Queue;
  public readonly compressionQueue: sqs.Queue;
  public readonly compressionDlq: sqs.Queue;

  constructor(scope: Construct, id: string, props: QueueStackProps) {
    super(scope, id, props);

    this.trainingSubmitDlq = new sqs.Queue(this, "TrainingSubmitDlq", {
      queueName: named(props.envName, "training-submit-dlq"),
      retentionPeriod: cdk.Duration.days(14),
    });
    this.trainingSubmitQueue = new sqs.Queue(this, "TrainingSubmitQueue", {
      queueName: named(props.envName, "training-submit"),
      visibilityTimeout: cdk.Duration.minutes(1),
      retentionPeriod: cdk.Duration.days(4),
      deadLetterQueue: {
        queue: this.trainingSubmitDlq,
        maxReceiveCount: 5,
      },
    });

    this.compressionDlq = new sqs.Queue(this, "CompressionDlq", {
      queueName: named(props.envName, "compression-dlq"),
      retentionPeriod: cdk.Duration.days(14),
    });
    this.compressionQueue = new sqs.Queue(this, "CompressionQueue", {
      queueName: named(props.envName, "compression"),
      // Compression tasks can be slow; 30 min visibility covers GB-scale work.
      visibilityTimeout: cdk.Duration.minutes(30),
      retentionPeriod: cdk.Duration.days(4),
      deadLetterQueue: {
        queue: this.compressionDlq,
        maxReceiveCount: 5,
      },
    });

    new cdk.CfnOutput(this, "TrainingSubmitQueueUrl", {
      value: this.trainingSubmitQueue.queueUrl,
    });
    new cdk.CfnOutput(this, "CompressionQueueUrl", {
      value: this.compressionQueue.queueUrl,
    });
  }
}

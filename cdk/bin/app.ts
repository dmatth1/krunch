#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { StorageStack } from "../lib/storage-stack";
import { QueueStack } from "../lib/queue-stack";
import { ApiStack } from "../lib/api-stack";
import { IngestStack } from "../lib/ingest-stack";
import { TrainingStack } from "../lib/training-stack";
import { CompressionStack } from "../lib/compression-stack";
import {
  EnvName,
  commonTags,
  named,
} from "../lib/shared/constants";

const app = new cdk.App();

const envName: EnvName = (app.node.tryGetContext("env") ?? "dev") as EnvName;
if (envName !== "dev" && envName !== "prod") {
  throw new Error(`unknown env ${envName}; expected 'dev' or 'prod'`);
}

const awsEnv = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION ?? "us-east-1",
};

const stackId = (name: string) => named(envName, name);

// -----------------------------------------------------------------------
// Stack 1: Storage — S3 + DynamoDB. Long-lived, depended on by all others.
// -----------------------------------------------------------------------
const storage = new StorageStack(app, stackId("storage"), {
  env: awsEnv,
  envName,
});

// -----------------------------------------------------------------------
// Stack 2: Queues — SQS queues + DLQs.
// -----------------------------------------------------------------------
const queues = new QueueStack(app, stackId("queues"), {
  env: awsEnv,
  envName,
});

// -----------------------------------------------------------------------
// Stack 3: API Gateway + PUT/GET Lambdas
// -----------------------------------------------------------------------
const api = new ApiStack(app, stackId("api"), {
  env: awsEnv,
  envName,
  bucket: storage.bucket,
  datasetsTable: storage.datasetsTable,
});

// -----------------------------------------------------------------------
// Stack 4: Ingest (S3 event → Lambda with concurrency guard)
// -----------------------------------------------------------------------
const ingest = new IngestStack(app, stackId("ingest"), {
  env: awsEnv,
  envName,
  bucket: storage.bucket,
  datasetsTable: storage.datasetsTable,
  trainingSubmitQueue: queues.trainingSubmitQueue,
  compressionQueue: queues.compressionQueue,
});

// -----------------------------------------------------------------------
// Stack 5: Training (Batch compute + launcher/completion Lambdas)
// -----------------------------------------------------------------------
const training = new TrainingStack(app, stackId("training"), {
  env: awsEnv,
  envName,
  bucket: storage.bucket,
  datasetsTable: storage.datasetsTable,
  modelVersionsTable: storage.modelVersionsTable,
  trainingSubmitQueue: queues.trainingSubmitQueue,
  compressionQueue: queues.compressionQueue,
});

// -----------------------------------------------------------------------
// Stack 6: Compression (Fargate worker)
// -----------------------------------------------------------------------
const compression = new CompressionStack(app, stackId("compression"), {
  env: awsEnv,
  envName,
  bucket: storage.bucket,
  datasetsTable: storage.datasetsTable,
  modelVersionsTable: storage.modelVersionsTable,
  compressionQueue: queues.compressionQueue,
});

// Apply common tags to every resource under each stack.
for (const stack of [storage, queues, api, ingest, training, compression]) {
  for (const [k, v] of Object.entries(commonTags(envName))) {
    cdk.Tags.of(stack).add(k, v);
  }
}

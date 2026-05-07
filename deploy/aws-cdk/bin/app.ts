#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import { KrunchStack } from "../lib/krunch-stack";

const app = new cdk.App();

new KrunchStack(app, "KrunchStack", {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION ?? "us-east-1",
  },
  description: "Krunch v1 — distributed neural compression on AWS Batch",

  // Defaults: g5.xlarge spot, ghcr.io/dmatth1/krunch:latest, 4 workers
  // max, 30-day CloudWatch retention. See KrunchStackProps in
  // ../lib/krunch-stack.ts for all overrides (maxWorkers, instanceType,
  // image, imageId, s3BucketName, spot, logRetention).
});

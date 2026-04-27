#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import { KrunchStack } from "../lib/krunch-stack";

const app = new cdk.App();

new KrunchStack(app, "KrunchStack", {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION ?? "us-east-1",
  },
  description: "Krunch v1 — distributed neural compression server",

  // Defaults: g5.xlarge spot, port 8080, ghcr.io/dmatth1/krunch:v1
  // Override examples:
  //   instanceType: ec2.InstanceType.of(ec2.InstanceClass.G5, ec2.InstanceSize.X2LARGE),
  //   spot: false,
  //   sshAllowedCidr: "203.0.113.0/32",  // your IP for SSH access
});

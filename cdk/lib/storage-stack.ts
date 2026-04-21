import * as cdk from "aws-cdk-lib";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as s3 from "aws-cdk-lib/aws-s3";
import { Construct } from "constructs";
import { EnvName, named } from "./shared/constants";

interface StorageStackProps extends cdk.StackProps {
  envName: EnvName;
}

/**
 * Long-lived data resources: the single S3 bucket for raw + compressed +
 * models, and two DynamoDB tables for dataset metadata + model registry.
 *
 * Retain policy: DESTROY in dev (so `cdk destroy` cleans up), RETAIN in
 * prod (never delete customer data accidentally).
 */
export class StorageStack extends cdk.Stack {
  public readonly bucket: s3.Bucket;
  public readonly datasetsTable: dynamodb.Table;
  public readonly modelVersionsTable: dynamodb.Table;

  constructor(scope: Construct, id: string, props: StorageStackProps) {
    super(scope, id, props);

    const isProd = props.envName === "prod";
    const removalPolicy = isProd
      ? cdk.RemovalPolicy.RETAIN
      : cdk.RemovalPolicy.DESTROY;
    const autoDeleteObjects = !isProd;

    // -----------------------------------------------------------------
    // S3 bucket — single bucket, customer/dataset prefixed.
    // Lifecycle cleans up orphaned raw/ objects after 7 days (safety net
    // if the compression stage fails to delete them).
    // -----------------------------------------------------------------
    this.bucket = new s3.Bucket(this, "ArchiveBucket", {
      bucketName: named(props.envName, "archive"),
      removalPolicy,
      autoDeleteObjects,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      versioned: false,
      eventBridgeEnabled: true,
      lifecycleRules: [
        {
          id: "gc-abandoned-raw",
          prefix: "", // applies to any raw/ object across all customers
          tagFilters: { lifecycle: "raw" },
          expiration: cdk.Duration.days(7),
        },
      ],
    });

    // -----------------------------------------------------------------
    // datasets table
    //   pk = CUST#{cid}, sk = DS#{dsid}
    //   attributes: status, raw_bytes_held, compressed_bytes,
    //   current_model_version, training_job_id, timestamps
    // -----------------------------------------------------------------
    this.datasetsTable = new dynamodb.Table(this, "DatasetsTable", {
      tableName: named(props.envName, "datasets"),
      partitionKey: { name: "pk", type: dynamodb.AttributeType.STRING },
      sortKey: { name: "sk", type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy,
      pointInTimeRecovery: isProd,
    });

    // -----------------------------------------------------------------
    // model_versions table
    //   pk = CUST#{cid}#DS#{dsid}, sk = version (number)
    // -----------------------------------------------------------------
    this.modelVersionsTable = new dynamodb.Table(this, "ModelVersionsTable", {
      tableName: named(props.envName, "model-versions"),
      partitionKey: { name: "pk", type: dynamodb.AttributeType.STRING },
      sortKey: { name: "sk", type: dynamodb.AttributeType.NUMBER },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy,
      pointInTimeRecovery: isProd,
    });

    new cdk.CfnOutput(this, "BucketName", {
      value: this.bucket.bucketName,
      exportName: named(props.envName, "bucket-name"),
    });
    new cdk.CfnOutput(this, "DatasetsTableName", {
      value: this.datasetsTable.tableName,
      exportName: named(props.envName, "datasets-table"),
    });
    new cdk.CfnOutput(this, "ModelVersionsTableName", {
      value: this.modelVersionsTable.tableName,
      exportName: named(props.envName, "model-versions-table"),
    });
  }
}

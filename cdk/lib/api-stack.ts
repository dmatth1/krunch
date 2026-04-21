import * as cdk from "aws-cdk-lib";
import * as apigw from "aws-cdk-lib/aws-apigateway";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import { Construct } from "constructs";
import { EnvName, named } from "./shared/constants";
import { NodejsFunction } from "aws-cdk-lib/aws-lambda-nodejs";
import { Runtime, Architecture } from "aws-cdk-lib/aws-lambda";

interface ApiStackProps extends cdk.StackProps {
  envName: EnvName;
  bucket: s3.IBucket;
  datasetsTable: dynamodb.ITable;
}

/**
 * REST API (API Gateway) + PUT/GET Lambda handlers.
 *
 * API is private in MVP: requires an API key, accessible via the
 * AWS console or from within a VPC. Auth layer gets bigger once we
 * have customers.
 */
export class ApiStack extends cdk.Stack {
  public readonly putFn: NodejsFunction;
  public readonly getFn: NodejsFunction;
  public readonly api: apigw.RestApi;

  constructor(scope: Construct, id: string, props: ApiStackProps) {
    super(scope, id, props);

    const baseEnv = {
      BUCKET_NAME: props.bucket.bucketName,
      DATASETS_TABLE_NAME: props.datasetsTable.tableName,
      NODE_OPTIONS: "--enable-source-maps",
    };

    // -----------------------------------------------------------------
    // PUT Lambda — handles small-payload inline PUTs and returns pre-
    // signed URLs for large payloads.
    // -----------------------------------------------------------------
    this.putFn = new NodejsFunction(this, "PutFn", {
      functionName: named(props.envName, "put-fn"),
      entry: "src/handlers/put.ts",
      handler: "handler",
      runtime: Runtime.NODEJS_20_X,
      architecture: Architecture.ARM_64,
      timeout: cdk.Duration.seconds(30),
      memorySize: 512,
      environment: baseEnv,
      bundling: { minify: false, sourceMap: true, format: undefined },
    });

    // -----------------------------------------------------------------
    // GET Lambda — MVP returns metadata only; range retrieval TBD.
    // -----------------------------------------------------------------
    this.getFn = new NodejsFunction(this, "GetFn", {
      functionName: named(props.envName, "get-fn"),
      entry: "src/handlers/get.ts",
      handler: "handler",
      runtime: Runtime.NODEJS_20_X,
      architecture: Architecture.ARM_64,
      timeout: cdk.Duration.seconds(30),
      memorySize: 256,
      environment: baseEnv,
      bundling: { minify: false, sourceMap: true, format: undefined },
    });

    props.bucket.grantReadWrite(this.putFn);
    props.datasetsTable.grantReadWriteData(this.putFn);
    props.datasetsTable.grantReadData(this.getFn);

    // -----------------------------------------------------------------
    // API Gateway: private REST API with API-key auth.
    // /v1/customers/{cid}/datasets/{dsid}/events
    // -----------------------------------------------------------------
    this.api = new apigw.RestApi(this, "ArchiveApi", {
      restApiName: named(props.envName, "api"),
      deployOptions: {
        stageName: props.envName,
        throttlingBurstLimit: 100,
        throttlingRateLimit: 50,
      },
      defaultMethodOptions: { apiKeyRequired: true },
    });

    const v1 = this.api.root.addResource("v1");
    const customers = v1.addResource("customers");
    const customer = customers.addResource("{cid}");
    const datasets = customer.addResource("datasets");
    const dataset = datasets.addResource("{dsid}");
    const events = dataset.addResource("events");

    const putIntegration = new apigw.LambdaIntegration(this.putFn, {
      proxy: true,
    });
    const getIntegration = new apigw.LambdaIntegration(this.getFn, {
      proxy: true,
    });

    events.addMethod("PUT", putIntegration);
    events.addMethod("GET", getIntegration);
    dataset.addMethod("GET", getIntegration); // dataset metadata

    // Single API key + usage plan for MVP. Real multi-tenant comes later.
    const apiKey = this.api.addApiKey("DevApiKey", {
      apiKeyName: named(props.envName, "api-key"),
    });
    const usagePlan = this.api.addUsagePlan("UsagePlan", {
      name: named(props.envName, "default-plan"),
      throttle: { rateLimit: 50, burstLimit: 100 },
    });
    usagePlan.addApiKey(apiKey);
    usagePlan.addApiStage({
      stage: this.api.deploymentStage,
    });

    new cdk.CfnOutput(this, "ApiEndpoint", {
      value: this.api.url,
      exportName: named(props.envName, "api-endpoint"),
    });
    new cdk.CfnOutput(this, "ApiKeyId", {
      value: apiKey.keyId,
      exportName: named(props.envName, "api-key-id"),
    });
  }
}

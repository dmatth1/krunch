import type {
  APIGatewayProxyEventV2,
  APIGatewayProxyStructuredResultV2,
} from "aws-lambda";
import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { randomUUID } from "node:crypto";
import { createDatasetIfNotExists, addRawBytes } from "../shared/ddb";

const s3 = new S3Client({});

const BUCKET = process.env.BUCKET_NAME!;
const SMALL_PUT_MAX_BYTES = 10 * 1024 * 1024; // 10 MB

/**
 * PUT /v1/customers/{cid}/datasets/{dsid}/events
 *
 * Small payloads (<10 MB) are forwarded directly to S3. Larger
 * payloads get a 307 redirect to a pre-signed S3 URL so the client
 * uploads directly without blowing through API Gateway's payload cap.
 *
 * Either way, the resulting S3 ObjectCreated event kicks the ingest
 * Lambda (in ingest-stack.ts), which is where the "should we train?"
 * decision happens. This Lambda does NOT enqueue training itself.
 */
export const handler = async (
  event: APIGatewayProxyEventV2,
): Promise<APIGatewayProxyStructuredResultV2> => {
  const cid = event.pathParameters?.cid;
  const dsid = event.pathParameters?.dsid;
  if (!cid || !dsid) {
    return json(400, { error: "missing path parameters" });
  }

  const contentLength = parseInt(event.headers["content-length"] ?? "0", 10);
  const batchId = randomUUID();
  const rawKey = `${cid}/${dsid}/raw/${batchId}.ndjson`;

  await createDatasetIfNotExists({ cid, dsid });

  // Large-payload path: return a pre-signed URL. Client re-uploads
  // directly to S3. Ingest Lambda picks up the resulting event.
  if (contentLength > SMALL_PUT_MAX_BYTES || contentLength === 0) {
    const url = await getSignedUrl(
      s3,
      new PutObjectCommand({
        Bucket: BUCKET,
        Key: rawKey,
        ContentType: "application/x-ndjson",
        Tagging: "lifecycle=raw",
      }),
      { expiresIn: 3600 },
    );
    return {
      statusCode: 307,
      headers: {
        "Location": url,
        "X-Batch-Id": batchId,
      },
      body: JSON.stringify({
        batch_id: batchId,
        upload_url: url,
        expires_in_seconds: 3600,
      }),
    };
  }

  // Small-payload path: accept inline, write to S3 directly.
  const bodyBuf = event.isBase64Encoded
    ? Buffer.from(event.body ?? "", "base64")
    : Buffer.from(event.body ?? "", "utf-8");

  await s3.send(
    new PutObjectCommand({
      Bucket: BUCKET,
      Key: rawKey,
      Body: bodyBuf,
      ContentType: "application/x-ndjson",
      Tagging: "lifecycle=raw",
    }),
  );

  const lineCount = bodyBuf
    .toString("utf-8")
    .split("\n")
    .filter((line) => line.trim().length > 0).length;
  await addRawBytes({ cid, dsid }, bodyBuf.byteLength, lineCount);

  return json(202, {
    batch_id: batchId,
    bytes_accepted: bodyBuf.byteLength,
    events_accepted: lineCount,
  });
};

function json(
  statusCode: number,
  body: unknown,
): APIGatewayProxyStructuredResultV2 {
  return {
    statusCode,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

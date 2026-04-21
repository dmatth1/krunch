import { S3Client, GetObjectCommand, ListObjectsV2Command } from "@aws-sdk/client-s3";
import { SQSClient, SendMessageBatchCommand } from "@aws-sdk/client-sqs";
import type { EventBridgeEvent } from "aws-lambda";
import {
  recordTrainingComplete,
  recordTrainingFailed,
} from "../shared/ddb";
import type { CompressionMessage } from "../shared/types";

const s3 = new S3Client({});
const sqs = new SQSClient({});

const BUCKET = process.env.BUCKET_NAME!;
const COMPRESSION_QUEUE_URL = process.env.COMPRESSION_QUEUE_URL!;

/**
 * Shape of the AWS Batch job-state-change event. Detail fields used:
 *   status: SUCCEEDED | FAILED | RUNNING | ...
 *   container.environment: our passed env vars (CUSTOMER_ID, DATASET_ID)
 *   jobId, jobName
 */
interface BatchJobStateChange {
  jobId: string;
  jobName: string;
  status: string;
  container?: {
    environment?: Array<{ name: string; value: string }>;
  };
}

/**
 * EventBridge handler: reacts to AWS Batch job state changes.
 *
 * On SUCCEEDED:
 * 1. Read training metadata JSON from S3 (emitted by the job at
 *    {cid}/{dsid}/models/v{N}.metadata.json)
 * 2. Update DDB: status = ready (or fallback_zstd), record model version
 * 3. List all pending raw/ objects for this dataset and enqueue them
 *    for compression. This is the "completion sweep" that handles PUTs
 *    that arrived while training was running.
 *
 * On FAILED:
 * Reset DDB status back to awaiting_corpus so the next PUT can retry.
 * The raw data is untouched.
 */
export const handler = async (
  event: EventBridgeEvent<"Batch Job State Change", BatchJobStateChange>,
): Promise<void> => {
  const status = event.detail.status;
  if (status !== "SUCCEEDED" && status !== "FAILED") {
    // Intermediate states (RUNNING, STARTING, etc.) — ignore.
    return;
  }

  const envMap = new Map<string, string>(
    (event.detail.container?.environment ?? []).map(
      (e): [string, string] => [e.name, e.value],
    ),
  );
  const cid = envMap.get("CUSTOMER_ID");
  const dsid = envMap.get("DATASET_ID");
  if (!cid || !dsid) {
    console.warn(
      `batch event for job ${event.detail.jobId} missing CUSTOMER_ID/DATASET_ID env; ignoring`,
    );
    return;
  }

  if (status === "FAILED") {
    await recordTrainingFailed({ cid, dsid });
    console.log(`training FAILED for ${cid}/${dsid}; status reset`);
    return;
  }

  // SUCCEEDED — read the training metadata emitted by the job.
  const modelMetadata = await readLatestModelMetadata(cid, dsid);
  if (!modelMetadata) {
    console.error(
      `training SUCCEEDED for ${cid}/${dsid} but no metadata file found; leaving in awaiting_corpus`,
    );
    await recordTrainingFailed({ cid, dsid });
    return;
  }

  await recordTrainingComplete(
    { cid, dsid },
    modelMetadata.version,
    modelMetadata.corpus_bytes,
    modelMetadata.codec,
  );

  await enqueueAllPendingRawObjects(cid, dsid, modelMetadata.version);
  console.log(
    `training complete for ${cid}/${dsid}; model v${modelMetadata.version} (codec=${modelMetadata.codec})`,
  );
};

interface ModelMetadata {
  version: number;
  corpus_bytes: number;
  codec: "l3tc" | "zstd_fallback";
  held_out_ratio: number;
  zstd_baseline_ratio: number;
}

async function readLatestModelMetadata(
  cid: string,
  dsid: string,
): Promise<ModelMetadata | null> {
  // List model metadata files; find the highest version.
  const prefix = `${cid}/${dsid}/models/`;
  const res = await s3.send(
    new ListObjectsV2Command({
      Bucket: BUCKET,
      Prefix: prefix,
    }),
  );
  const metadataKeys = (res.Contents ?? [])
    .map((o) => o.Key!)
    .filter((k) => k.endsWith(".metadata.json"));
  if (metadataKeys.length === 0) return null;
  metadataKeys.sort(); // v1, v2, ... lex-sorts correctly up to v9; fix at v10+

  const latestKey = metadataKeys[metadataKeys.length - 1];
  const obj = await s3.send(
    new GetObjectCommand({ Bucket: BUCKET, Key: latestKey }),
  );
  const text = await obj.Body!.transformToString();
  return JSON.parse(text) as ModelMetadata;
}

/**
 * Enqueue compression for every raw/ object under this dataset. Runs
 * after training completes to sweep up everything that accumulated.
 */
async function enqueueAllPendingRawObjects(
  cid: string,
  dsid: string,
  modelVersion: number,
): Promise<void> {
  const rawPrefix = `${cid}/${dsid}/raw/`;
  let continuationToken: string | undefined;
  const batch: CompressionMessage[] = [];

  do {
    const res = await s3.send(
      new ListObjectsV2Command({
        Bucket: BUCKET,
        Prefix: rawPrefix,
        ContinuationToken: continuationToken,
      }),
    );
    for (const obj of res.Contents ?? []) {
      if (!obj.Key) continue;
      batch.push({
        cid,
        dsid,
        s3_raw_key: obj.Key,
        model_version: modelVersion,
      });
    }
    continuationToken = res.IsTruncated ? res.NextContinuationToken : undefined;
  } while (continuationToken);

  // SQS SendMessageBatch allows up to 10 per call.
  for (let i = 0; i < batch.length; i += 10) {
    const chunk = batch.slice(i, i + 10);
    await sqs.send(
      new SendMessageBatchCommand({
        QueueUrl: COMPRESSION_QUEUE_URL,
        Entries: chunk.map((m, idx) => ({
          Id: `${i + idx}`,
          MessageBody: JSON.stringify(m),
        })),
      }),
    );
  }
  console.log(
    `enqueued ${batch.length} raw objects for compression under ${cid}/${dsid}`,
  );
}

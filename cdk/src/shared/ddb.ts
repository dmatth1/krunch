import {
  DynamoDBClient,
  ConditionalCheckFailedException,
} from "@aws-sdk/client-dynamodb";
import {
  DynamoDBDocumentClient,
  GetCommand,
  UpdateCommand,
  PutCommand,
} from "@aws-sdk/lib-dynamodb";
import {
  DatasetKey,
  DatasetRow,
  DatasetStatus,
  datasetPk,
  datasetSk,
} from "./types";

const ddb = DynamoDBDocumentClient.from(new DynamoDBClient({}));

const DATASETS_TABLE = process.env.DATASETS_TABLE_NAME!;

export { ConditionalCheckFailedException };

export async function getDataset(
  key: DatasetKey,
): Promise<DatasetRow | undefined> {
  const res = await ddb.send(
    new GetCommand({
      TableName: DATASETS_TABLE,
      Key: { pk: datasetPk(key.cid), sk: datasetSk(key.dsid) },
    }),
  );
  return res.Item as DatasetRow | undefined;
}

/**
 * Create a dataset row if it doesn't exist. Idempotent — if it exists,
 * returns the existing row without modification.
 *
 * Used by PUT Lambda on first write from a customer for a new dataset.
 */
export async function createDatasetIfNotExists(
  key: DatasetKey,
): Promise<void> {
  const now = new Date().toISOString();
  try {
    await ddb.send(
      new PutCommand({
        TableName: DATASETS_TABLE,
        Item: {
          pk: datasetPk(key.cid),
          sk: datasetSk(key.dsid),
          created_at: now,
          status: DatasetStatus.AwaitingCorpus,
          current_model_version: null,
          raw_bytes_held: 0,
          compressed_bytes: 0,
          total_events_ingested: 0,
        },
        ConditionExpression: "attribute_not_exists(pk)",
      }),
    );
  } catch (e) {
    if (e instanceof ConditionalCheckFailedException) return; // already exists
    throw e;
  }
}

/**
 * Increment raw_bytes_held and total_events_ingested.
 *
 * Does NOT change status. Called on every PUT to track bytes in the
 * raw holding area before compression.
 */
export async function addRawBytes(
  key: DatasetKey,
  bytes: number,
  events: number,
): Promise<void> {
  await ddb.send(
    new UpdateCommand({
      TableName: DATASETS_TABLE,
      Key: { pk: datasetPk(key.cid), sk: datasetSk(key.dsid) },
      UpdateExpression:
        "ADD raw_bytes_held :b, total_events_ingested :e",
      ExpressionAttributeValues: { ":b": bytes, ":e": events },
    }),
  );
}

/**
 * THE concurrency guard. Atomically transitions status from
 * AwaitingCorpus → Training. Throws ConditionalCheckFailedException
 * if the transition is not permitted (another Lambda got there first,
 * or status is something other than AwaitingCorpus).
 *
 * Caller catches the exception and treats it as "someone else is
 * handling this dataset's training; skip submission."
 */
export async function claimTrainingSlot(key: DatasetKey): Promise<void> {
  const now = new Date().toISOString();
  await ddb.send(
    new UpdateCommand({
      TableName: DATASETS_TABLE,
      Key: { pk: datasetPk(key.cid), sk: datasetSk(key.dsid) },
      UpdateExpression: "SET #s = :training, last_training_started_at = :now",
      ConditionExpression: "#s = :awaiting",
      ExpressionAttributeNames: { "#s": "status" },
      ExpressionAttributeValues: {
        ":training": DatasetStatus.Training,
        ":awaiting": DatasetStatus.AwaitingCorpus,
        ":now": now,
      },
    }),
  );
}

/**
 * Record the Batch job id after submission. Best-effort; failure here
 * just means we don't have the job id handy for a later kill.
 */
export async function recordTrainingJobId(
  key: DatasetKey,
  jobId: string,
): Promise<void> {
  await ddb.send(
    new UpdateCommand({
      TableName: DATASETS_TABLE,
      Key: { pk: datasetPk(key.cid), sk: datasetSk(key.dsid) },
      UpdateExpression: "SET training_job_id = :j",
      ExpressionAttributeValues: { ":j": jobId },
    }),
  );
}

/**
 * Called by the training-complete handler when a Batch job succeeds.
 * Transitions status to Ready and records the new model version.
 */
export async function recordTrainingComplete(
  key: DatasetKey,
  modelVersion: number,
  trainingCorpusBytes: number,
  codec: "l3tc" | "zstd_fallback",
): Promise<void> {
  const now = new Date().toISOString();
  const nextStatus =
    codec === "l3tc" ? DatasetStatus.Ready : DatasetStatus.FallbackZstd;
  await ddb.send(
    new UpdateCommand({
      TableName: DATASETS_TABLE,
      Key: { pk: datasetPk(key.cid), sk: datasetSk(key.dsid) },
      UpdateExpression:
        "SET #s = :s, current_model_version = :v, last_training_completed_at = :now, training_corpus_bytes = :b REMOVE training_job_id",
      ExpressionAttributeNames: { "#s": "status" },
      ExpressionAttributeValues: {
        ":s": nextStatus,
        ":v": modelVersion,
        ":now": now,
        ":b": trainingCorpusBytes,
      },
    }),
  );
}

/**
 * Called on training failure — reset status to AwaitingCorpus so the
 * next PUT can try again. Keep the raw bytes (customer data is intact).
 */
export async function recordTrainingFailed(key: DatasetKey): Promise<void> {
  await ddb.send(
    new UpdateCommand({
      TableName: DATASETS_TABLE,
      Key: { pk: datasetPk(key.cid), sk: datasetSk(key.dsid) },
      UpdateExpression: "SET #s = :a REMOVE training_job_id",
      ConditionExpression: "#s = :t",
      ExpressionAttributeNames: { "#s": "status" },
      ExpressionAttributeValues: {
        ":a": DatasetStatus.AwaitingCorpus,
        ":t": DatasetStatus.Training,
      },
    }),
  );
}

import { SQSClient, SendMessageCommand } from "@aws-sdk/client-sqs";
import type { EventBridgeEvent } from "aws-lambda";
import {
  ConditionalCheckFailedException,
  addRawBytes,
  claimTrainingSlot,
  createDatasetIfNotExists,
  getDataset,
} from "../shared/ddb";
import {
  DatasetStatus,
  type CompressionMessage,
  type TrainingSubmitMessage,
} from "../shared/types";

/** Shape of the S3-to-EventBridge "Object Created" event detail. */
interface S3ObjectCreatedDetail {
  version: string;
  bucket: { name: string };
  object: {
    key: string;
    size: number;
    etag?: string;
    "version-id"?: string;
    sequencer?: string;
  };
  "request-id"?: string;
  requester?: string;
  "source-ip-address"?: string;
  reason?: string;
}

const sqs = new SQSClient({});

const TRAINING_SUBMIT_QUEUE_URL = process.env.TRAINING_SUBMIT_QUEUE_URL!;
const COMPRESSION_QUEUE_URL = process.env.COMPRESSION_QUEUE_URL!;
const INITIAL_TRAINING_MIN_BYTES = 256 * 1024 * 1024; // 256 MB

/**
 * S3 ObjectCreated handler. Triggered when a raw NDJSON file lands
 * under {cid}/{dsid}/raw/.
 *
 * Decision tree:
 * 1. Decode cid + dsid from the S3 key
 * 2. Increment raw_bytes_held in DDB
 * 3. Read dataset status:
 *    - AwaitingCorpus, and raw_bytes_held >= threshold
 *        → CLAIM training via atomic DDB conditional update
 *        → send msg to training-submit SQS
 *        (losing the race here silently skips, which is correct —
 *         whoever won the race will handle this raw object later)
 *    - Training: skip. When training completes, the completion
 *                handler will enqueue all pending raw/ objects for
 *                compression.
 *    - Ready: send msg to compression SQS with current model version
 *    - Retraining/FallbackZstd: same as Ready for compression
 */
export const handler = async (
  event: EventBridgeEvent<"Object Created", S3ObjectCreatedDetail>,
): Promise<void> => {
  const key = decodeURIComponent(
    event.detail.object.key.replace(/\+/g, " "),
  );
  const size = event.detail.object.size ?? 0;

  const parsed = parseRawKey(key);
  if (!parsed) {
    console.warn(`ignoring non-raw key: ${key}`);
    return;
  }

  const { cid, dsid } = parsed;
  await createDatasetIfNotExists({ cid, dsid });
  await addRawBytes({ cid, dsid }, size, 0);

  const ds = await getDataset({ cid, dsid });
  if (!ds) {
    console.error(`dataset vanished after create: ${cid}/${dsid}`);
    return;
  }

  switch (ds.status) {
    case DatasetStatus.AwaitingCorpus:
      await maybeKickoffTraining({ cid, dsid }, ds.raw_bytes_held, key);
      break;
    case DatasetStatus.Training:
      // Someone else is training; they'll pick this up on completion sweep.
      console.log(`dataset ${cid}/${dsid} is training; skipping enqueue`);
      break;
    case DatasetStatus.Ready:
    case DatasetStatus.Retraining:
    case DatasetStatus.FallbackZstd:
      if (ds.current_model_version == null) {
        console.error(
          `dataset ${cid}/${dsid} is ${ds.status} but has no current_model_version; skipping`,
        );
        break;
      }
      await enqueueCompression({
        cid,
        dsid,
        s3_raw_key: key,
        model_version: ds.current_model_version,
      });
      break;
    default:
      console.warn(`unknown dataset status: ${ds.status}`);
  }
};

function parseRawKey(
  key: string,
): { cid: string; dsid: string } | null {
  // Expected: {cid}/{dsid}/raw/{uuid}.ndjson
  const parts = key.split("/");
  if (parts.length !== 4 || parts[2] !== "raw") return null;
  return { cid: parts[0], dsid: parts[1] };
}

/**
 * Try to claim the training slot for this dataset and submit a training
 * job. The conditional DDB update is the concurrency guard — only one
 * Lambda invocation can win the race to flip status=training.
 *
 * If raw_bytes_held is still below the minimum corpus size, skip —
 * later PUTs will re-check when more data has accumulated.
 */
async function maybeKickoffTraining(
  key: { cid: string; dsid: string },
  rawBytesHeld: number,
  triggeringRawKey: string,
): Promise<void> {
  if (rawBytesHeld < INITIAL_TRAINING_MIN_BYTES) {
    console.log(
      `dataset ${key.cid}/${key.dsid} has ${rawBytesHeld}B < ${INITIAL_TRAINING_MIN_BYTES}B; waiting for more`,
    );
    return;
  }

  try {
    await claimTrainingSlot(key);
  } catch (e) {
    if (e instanceof ConditionalCheckFailedException) {
      // Another Lambda won the race. They'll submit the job and handle
      // all queued raw/ objects on completion. Nothing for us to do.
      console.log(
        `lost training-slot race for ${key.cid}/${key.dsid}; skipping submission`,
      );
      return;
    }
    throw e;
  }

  // We own the slot. Submit to training queue.
  const msg: TrainingSubmitMessage = {
    cid: key.cid,
    dsid: key.dsid,
    trigger: "initial",
    raw_key: triggeringRawKey,
  };
  await sqs.send(
    new SendMessageCommand({
      QueueUrl: TRAINING_SUBMIT_QUEUE_URL,
      MessageBody: JSON.stringify(msg),
    }),
  );
  console.log(
    `submitted training for ${key.cid}/${key.dsid} on ${rawBytesHeld}B corpus`,
  );
}

async function enqueueCompression(msg: CompressionMessage): Promise<void> {
  await sqs.send(
    new SendMessageCommand({
      QueueUrl: COMPRESSION_QUEUE_URL,
      MessageBody: JSON.stringify(msg),
    }),
  );
}

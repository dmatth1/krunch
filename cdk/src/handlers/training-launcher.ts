import { BatchClient, SubmitJobCommand } from "@aws-sdk/client-batch";
import type { SQSEvent } from "aws-lambda";
import { recordTrainingJobId } from "../shared/ddb";
import type { TrainingSubmitMessage } from "../shared/types";

const batch = new BatchClient({});

const JOB_DEFINITION = process.env.BATCH_JOB_DEFINITION!;
const JOB_QUEUE = process.env.BATCH_JOB_QUEUE!;
const BUCKET_NAME = process.env.BUCKET_NAME!;

/**
 * SQS handler: consumes messages from training-submit queue and submits
 * AWS Batch training jobs.
 *
 * The DDB status=training transition was already done by the ingest
 * Lambda; this Lambda's job is just to submit to Batch and record the
 * job id. If submission fails, the message returns to the queue via
 * SQS retry, ultimately landing in the DLQ after maxReceiveCount.
 */
export const handler = async (event: SQSEvent): Promise<void> => {
  for (const record of event.Records) {
    const msg: TrainingSubmitMessage = JSON.parse(record.body);
    const jobName = sanitizeJobName(
      `${msg.cid}-${msg.dsid}-${Date.now()}`,
    );
    const res = await batch.send(
      new SubmitJobCommand({
        jobName,
        jobDefinition: JOB_DEFINITION,
        jobQueue: JOB_QUEUE,
        parameters: {
          CUSTOMER_ID: msg.cid,
          DATASET_ID: msg.dsid,
          TRIGGER: msg.trigger,
        },
        containerOverrides: {
          environment: [
            { name: "CUSTOMER_ID", value: msg.cid },
            { name: "DATASET_ID", value: msg.dsid },
            { name: "TRIGGER", value: msg.trigger },
            { name: "BUCKET_NAME", value: BUCKET_NAME },
            {
              name: "S3_RAW_PREFIX",
              value: `${msg.cid}/${msg.dsid}/raw/`,
            },
            {
              name: "S3_MODEL_PREFIX",
              value: `${msg.cid}/${msg.dsid}/models/`,
            },
          ],
        },
        // Retries inside Batch are separate from SQS; each failed
        // attempt is recorded as a separate job.
        retryStrategy: { attempts: 2 },
      }),
    );

    if (res.jobId) {
      await recordTrainingJobId({ cid: msg.cid, dsid: msg.dsid }, res.jobId);
    }
    console.log(
      `submitted batch job for ${msg.cid}/${msg.dsid}: jobId=${res.jobId}`,
    );
  }
};

/**
 * AWS Batch job names must be 1-128 chars, letters/numbers/hyphens/underscores.
 */
function sanitizeJobName(name: string): string {
  return name.replace(/[^a-zA-Z0-9_-]/g, "-").slice(0, 128);
}

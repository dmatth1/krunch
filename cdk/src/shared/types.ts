/** Types shared by Lambda handlers. Stay dependency-free. */

export interface DatasetKey {
  cid: string;
  dsid: string;
}

export const DatasetStatus = {
  AwaitingCorpus: "awaiting_corpus",
  Training: "training",
  Ready: "ready",
  Retraining: "retraining",
  FallbackZstd: "fallback_zstd",
} as const;

export type DatasetStatusType =
  (typeof DatasetStatus)[keyof typeof DatasetStatus];

export interface DatasetRow {
  pk: string; // CUST#{cid}
  sk: string; // DS#{dsid}
  created_at: string;
  status: DatasetStatusType;
  current_model_version: number | null;
  raw_bytes_held: number;
  compressed_bytes: number;
  total_events_ingested: number;
  last_training_started_at?: string;
  last_training_completed_at?: string;
  training_job_id?: string;
  training_corpus_bytes?: number;
}

export interface TrainingSubmitMessage {
  cid: string;
  dsid: string;
  trigger: "initial" | "retrain";
  /** S3 key of the raw object that triggered this; passed through so
   *  the launcher can include it in the job's input manifest. */
  raw_key?: string;
}

export interface CompressionMessage {
  cid: string;
  dsid: string;
  s3_raw_key: string;
  model_version: number;
}

export const datasetPk = (cid: string) => `CUST#${cid}`;
export const datasetSk = (dsid: string) => `DS#${dsid}`;
export const modelVersionsPk = (cid: string, dsid: string) =>
  `CUST#${cid}#DS#${dsid}`;

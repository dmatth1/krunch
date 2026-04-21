export const PROJECT_TAG = "learned-archive";

/** Environments currently in use. MVP only ships `dev`. */
export type EnvName = "dev" | "prod";

/** Naming helpers — every resource gets a stable, env-scoped name. */
export const named = (env: EnvName, name: string) => `archive-${env}-${name}`;

/** Tags applied to every CDK-managed resource. */
export const commonTags = (env: EnvName) => ({
  Project: PROJECT_TAG,
  Environment: env,
  ManagedBy: "cdk",
});

/** Dataset state machine. See docs/SERVICE_ARCHITECTURE.md. */
export const DatasetStatus = {
  AwaitingCorpus: "awaiting_corpus",
  Training: "training",
  Ready: "ready",
  Retraining: "retraining",
  FallbackZstd: "fallback_zstd",
} as const;

export type DatasetStatusType =
  (typeof DatasetStatus)[keyof typeof DatasetStatus];

/** Small-payload threshold — PUTs above this require pre-signed URL flow. */
export const SMALL_PUT_MAX_BYTES = 10 * 1024 * 1024; // 10 MB

/** Minimum raw bytes to accumulate before kicking off initial training. */
export const INITIAL_TRAINING_MIN_BYTES = 256 * 1024 * 1024; // 256 MB

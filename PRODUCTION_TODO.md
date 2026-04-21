# Production readiness — open items

Captured 2026-04-21 during Spike 1. The service skeleton is working
end-to-end (all 6 stacks deployed, PUT → ingest → training → metadata
emit confirmed), but the list below stands between "works for the
pilot dataset" and "ready for real customers". Numbered in priority
order within each bucket.

## Must-have before any real customer

### 1. GPU capacity strategy
- Spike 1 runs on **on-demand g5.xlarge** because spot g5/g6 across 4
  AZs was `UnfulfillableCapacity` on 2026-04-21. `cdk/lib/training-stack.ts`
  has `spot: false` with a comment.
- Fix: add a fallback path — job queue attempts spot first, falls back
  to on-demand if spot is unavailable for >N minutes. AWS Batch supports
  two compute envs on one queue with ordered priority. Re-enable spot
  when we reintroduce the fallback.

### 2. Observability
- Today, pulling pipeline state takes 5 separate CLI calls (DDB row,
  S3 listing, Batch status, CloudWatch logs, CFN status). Not viable.
- Fix:
  - CloudWatch dashboard: ingest-Lambda invocations, Batch job
    successes/failures, queue depth, Fargate scaling events.
  - SNS alarm on any Batch job FAILED.
  - Per-dataset timeline in DDB (`events` list) so the status GET can
    show "awaiting_corpus → training (v1 started 15:30) → training
    FAILED (15:45) → retry started".

### 3. Error surfacing to the API user
- On Batch FAILED, `training-complete.ts` silently resets DDB to
  `awaiting_corpus`. The customer sees status=training, then sees it
  mysteriously drop back. No explanation.
- Fix:
  - Add `training_failed_reason` + `training_failed_at` fields to DDB.
  - GET endpoint exposes them. "Training failed: tokenizer rejected
    the input format" is infinitely better than silent rollback.
  - Separate status `training_failed` from `awaiting_corpus` so the
    customer knows an action is needed.

### 4. Retry policy
- JobDefinition has `retryAttempts: 1`. An OOM from a transient memory
  spike fails the whole attempt.
- Fix:
  - `retryAttempts: 3` with exponential backoff.
  - Distinguish transient (spot interruption, OOM, network) from
    terminal (bad corpus, code bug). Only retry transient. Terminal
    → `training_failed` permanently, require customer action.

### 5. Authentication / tenant isolation
- `/v1/customers/{cid}/datasets/{dsid}/...` accepts any valid API key
  for any `cid`. A customer with their own API key can read/write any
  other customer's data.
- Fix:
  - Per-customer API keys mapped to their `cid`. Handlers validate
    `key.customer_id === path.cid`.
  - Move key metadata out of the API Gateway and into DDB; rotate on
    schedule.
  - IAM condition on S3 read/write scoped to `{cid}/` prefix per key.

### 6. Decompression API
- `GET /v1/customers/{cid}/datasets/{dsid}/events` Lambda exists but
  returns placeholder JSON; it doesn't actually decode anything.
- Fix: once compression moves off zstd-fallback, implement
  streaming decode: list `compressed/` objects, stream each through
  the codec, concatenate as NDJSON response. Pagination via
  continuation tokens.

## Should-have (blocks growth past a handful of customers)

### 7. L3TC runtime integration (gated on Spike 1 pass)
- Fargate compression worker currently runs **zstd_fallback only**.
  The model is trained, the ratio is measured, but we don't store
  model-compressed bytes.
- Fix (if Spike 1 passes): build a Rust-based compression image from
  `l3tc-rust/`; wire into the worker when `codec=l3tc`; add
  decode path for the GET endpoint.

### 8. Per-dataset tokenizer domain
- Per-dataset tokenizer training: ✓ already in place.
- `--domain` flag: **purely a label** in
  `scripts/train_specialist_tokenizer.py` (tested 2026-04-21) —
  hyperparameters are identical for all domain values. Spike 1 hardcodes
  `DOMAIN=logs`. Not a correctness issue, just a labelling issue.
- Fix: let customer pass `domain` at dataset-create time for operator
  tagging and future-proofing (if domain ever starts affecting
  hyperparameters). Plumb from API → DDB → launcher → Batch env.

### 9. vendor/L3TC version pin
- `vendor/L3TC/` is a plain snapshot copy of
  `github.com/alipay/L3TC-...`. No git submodule, no SHA recorded,
  and we've patched it locally (`rwkv_v4_train.py` deepspeed fallback
  on 2026-04-21). Re-clone would wipe the patch.
- Fix (pick one):
  1. **Submodule + patches dir.** Pin upstream SHA; store diffs in
     `patches/` applied at build time. Upstream-friendly, least work.
  2. **Fork into `github.com/dmatth1/L3TC-fork`**, bake in patches,
     pin by SHA. Clearest ownership.
  3. **Rewrite the parts we actually use** (~300-500 lines for
     `build_rwkv_v4`, the model class, and the config shim) — drop
     vendor dep entirely. Best long-term.
- Recommend (2) before first customer, (3) after Spike 2.

### 10. CI/CD for Docker image builds

**Partially done 2026-04-21.** CodeBuild project
`krunch-image-build` is live. First build confirmed: 141 s BUILD
phase (vs. 30+ min laptop push). Iteration loop + commands are in
[`cdk/README.md`](cdk/README.md).

Still open:
- CDK's `DockerImageAsset` still does a local build when someone runs
  `cdk deploy`. Swap for `ContainerImage.fromEcrRepository(repo, sha)`
  so the laptop only touches CFN, never builds images.
- Integrate CodeBuild with CDK so the Batch JobDefinition's
  `container.image` auto-updates to the CodeBuild-produced tag on
  each successful build, instead of the current manual
  `register-job-definition` dance.
- ECR pull-through cache for Docker Hub so the PyTorch base layer
  stays inside our account (eliminates the Docker-Hub-to-CodeBuild
  download on cold builds).
- GitHub webhook trigger so a push to `main` fires CodeBuild
  automatically instead of needing `aws codebuild start-build`.

## Faster iteration (operational, not customer-facing)

### Test the training container locally before pushing
- Every dep/import bug in Spike 1 (`deepspeed` CUDA toolkit
  requirement, wkv_op.cpp relative path, missing `tqdm`, domain
  enum rejection) would have surfaced in 20 seconds of a local
  `docker run --entrypoint python <image> -c "…"` on the dev box.
  Instead each bug cost a 10-15 min cdk-deploy + Batch-submit cycle.
- Fix: before `cdk deploy`, run a shake-down command that imports
  the training entrypoint modules. Bake this into a
  `scripts/prebuild_check.sh` and run it in the CodeBuild pipeline.

### SSH / exec into the running Batch instance
- The on-demand g5.xlarge stays alive between jobs (Batch scale-down
  takes ~10 min). Pulling the image locally and `docker exec`-ing in
  is faster than submitting a fresh Batch job for each deploy.

### Skip Batch entirely during shake-down
- For pre-prod iteration, run the entrypoint on the bare g5 instance
  with the env vars directly, not through Batch. Saves the
  SUBMITTED → RUNNABLE → STARTING → RUNNING cycle per retry.

## Tracking

When Spike 1 lands (pass or fail), re-prioritize this list based on
outcome:
- Pass → items 1-5 are the path to first customer; 7/9/10 close after.
- Fail → revisit thesis first; none of this matters if the model
  doesn't beat zstd.

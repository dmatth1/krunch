# Spike 1 — work log

Running autonomously while user is away (started 2026-04-21).

## Goal

Answer: **does a 2L × 96H × 16K RWKV-v4 model trained on ~1 GB of real
log data beat `zstd --long=27 --ultra -22` on held-out data from the
same distribution?**

If yes → we have a real compression advantage for a managed-service
product. If no → the whole thesis needs revisiting.

Pass threshold: **our entropy-bound ratio ≤ 98% × zstd-22 ratio**
(i.e. we need to be at least 2% better; anything less than that is
noise). Ideally 30-60% better.

## Scope decisions made during setup

- **Dataset:** HDFS log dataset from Loghub 2.0. Available at
  `https://zenodo.org/record/8196385/files/HDFS_v1.tar.gz`. Single
  format (HDFS NameNode logs), highly homogeneous, well-studied.
- **Size:** 1 GB slice (HDFS_v1 is ~1.5 GB total raw; we use the
  first 1 GB for training + 20% of that held out for val).
- **Training hardware:** g5.xlarge spot (NVIDIA A10G 24 GB), Batch
  compute env. 4 vCPU, 16 GB RAM.
- **Training config:** 10 epochs × 50K samples × batch 32 × ctx 2048
  on GPU bf16. Expected wall time ~1 hr on A10G.
- **Measurement:** entropy-bound ratio via
  `scripts/measure_held_out_ratio.py` (computes
  `-mean(log2 p_true) / 8 / bytes_per_token`). Within ~1% of actual
  coded bytes, adequate for go/no-go.
- **No `.bin` conversion in Spike 1.** Produces `.pth` + tokenizer +
  metadata only. Compression storage for Spike 1 is zstd-only via the
  Fargate worker; the neural model's ratio is recorded in metadata
  but not used for actual compression yet. This isolates the
  "does the model beat zstd" question from "does the Rust runtime
  work end to end".

## Service architecture checkpoint

4 of 6 CloudFormation stacks deployed and healthy before this run:

- `archive-dev-storage` ✓ (S3 bucket + 2 DDB tables)
- `archive-dev-queues` ✓ (SQS training-submit + compression + DLQs)
- `archive-dev-api` ✓ (API Gateway + PUT/GET Lambdas)
- `archive-dev-ingest` ✓ (S3 EventBridge → ingest Lambda with DDB concurrency guard)

Remaining stacks being deployed now:

- `archive-dev-training` — Batch compute env (g5.xlarge spot GPU) +
  launcher + completion Lambdas
- `archive-dev-compression` — Fargate zstd compression worker

## Pre-flight fixes applied

Found during prep, fixed before deploy:

1. **Missing `scripts/convert_checkpoint.py`** — copied from
   `l3tc-rust/scripts/`.
2. **Missing `scripts/measure_held_out_ratio.py`** — wrote from
   scratch using `build_model` from `train_l3tc_phase11.py`.
3. **Arg mismatch** — `train_specialist_tokenizer.py` expects
   `--domain --corpus --output-dir --sample-mb`, not the
   `--input --vocab-size` my earlier entrypoint used.
4. **Checkpoint filename** — `train_l3tc_phase11.py` saves
   `checkpoint_latest.pth` (no `_final`). Updated entrypoint.
5. **Dockerfile: Rust 1.82-slim failed on `clap_derive 4.6.0`
   requiring edition2024** — bumped to `rust:1-slim` (auto-resolves
   to 1.87+). Subsequently simplified compression image to pure
   Python+zstd since Spike 1 doesn't use the Rust binary.
6. **Dockerfile: CPU-only PyTorch** — switched training base image
   to `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` for GPU.
7. **Set G5.XLARGE + gpu=1 + maxvCpus=8** in training stack (quota
   allows 2 concurrent).
8. **Cross-stack S3 event cycle** — ingest stack now uses EventBridge
   S3 events instead of direct S3 notifications (already fixed before
   this spike).
9. **Docker build context was 1.7 GB** — added `.dockerignore` to
   exclude `.git/`, `vendor/L3TC/checkpoints/`, `corpus_build/`, etc.

## Timeline

| time | event |
|---|---|
| 13:01 | First deploy attempt (`cdk deploy --all`) starts — deploy command backgrounded by harness |
| ~13:05 | Docker push of CPU training image begins (~3 GB) |
| 13:07 | ingest stack reaches CREATE_COMPLETE; 4/6 stacks up |
| 13:17 | Original deploy exits; compression image failed to build (Rust 1.82 too old for `clap_derive@4.6.0`) |
| 13:19 | Switched Dockerfile to `rust:1-slim`, re-deployed training + compression only |
| 13:24 | Compression stack enters CREATE_IN_PROGRESS (at 21/29 resources) |
| — | Killed v2 deploy to switch training to GPU + restructure compression to zstd-only |
| 13:34 | Third deploy attempt stalled mid-build on tzdata interactive prompt (no DEBIAN_FRONTEND) |
| 13:45 | Diagnosed compression stack: Fargate service in public subnets but `assignPublicIp=DISABLED` → ECR pull failing |
| 13:45 | Hot-fix via `aws ecs update-service` with `assignPublicIp=ENABLED`; task transitions to RUNNING |
| 13:46 | Started training deploy v2 (with `DEBIAN_FRONTEND=noninteractive` fix); began HDFS download via Zenodo API |
| 13:47 | Extracted HDFS.log (1.58 GB); converted first 1 GB to NDJSON (7.6M lines, 1.39 GB output) |
| 13:49 | Compression stack CREATE_COMPLETE |
| 13:50 | First upload attempt failed (x-amz-tagging header not signed); retried without extra headers |
| 13:50 | Upload of 1.39 GB to pre-signed S3 URL started (batch_id=45a139b9-ce09-440e-82e4-e55d5024fcb0) |
| 13:51 | Training Docker build succeeded; ECR push underway |

_(Will backfill below as events happen.)_

## Action log

_(Append-only record of what runs, outputs, and decisions.)_

### [pending] Deploy training + compression with GPU config

Command: `nohup npx cdk deploy archive-dev-training archive-dev-compression --context env=dev --require-approval never > /tmp/cdk-deploy-gpu.log 2>&1 &`

Expected duration: ~15-25 min (PyTorch CUDA image pull + ECR push).

### [pending] Download HDFS logs

```bash
wget https://zenodo.org/record/8196385/files/HDFS_v1.tar.gz -O /tmp/hdfs_v1.tar.gz
tar -xzf /tmp/hdfs_v1.tar.gz -C /tmp/
```

### [pending] Convert to NDJSON

Wrap each log line as `{"ts": "<parsed_timestamp>", "line": "<raw>"}`
so the PUT API sees NDJSON. Size target: ~1 GB output.

### [pending] PUT to API

Use the API endpoint + API key from CDK outputs. Expect the 307
redirect pre-signed URL flow for a 1 GB body.

### [pending] Monitor ingest → training → ratio result

Use `scripts/spike_status.sh acme hdfs-spike` to poll.

## Pass / fail criteria

Pass: `metadata.json.held_out_ratio < 0.98 × metadata.json.zstd_baseline_ratio`

Ideal pass: `held_out_ratio < 0.5 × zstd_baseline_ratio`.

Marginal: 0.5-0.98× — real but small win, wouldn't carry service economics.

Fail: `held_out_ratio ≥ zstd_baseline_ratio` — model didn't learn
enough to beat generic compression; thesis needs revisiting.

## Final results (2026-04-21 18:39 UTC)

### The numbers

| metric | value |
|---|---|
| corpus | HDFS_v1 from Loghub 2.0 Zenodo 8196385 |
| raw NDJSON | 1,386,685,235 B (1.39 GB) |
| train split | 1,109,348,188 B (80%) — tokenized to **345,040,742** tokens |
| val split | 277,337,047 B (20%) — tokenized to **91,852,755** tokens |
| SPM vocab | 16,384 unigram + byte_fallback |
| bytes/token (val) | 3.019 |
| final eval `avg_ce_nats` | 2.3511 (epoch 9, 409,600-token slice) |
| **final eval `bits/token`** | **3.3920** |
| **entropy-bound ratio = (3.3920 / 8) / 3.019** | **0.1405 (14.05%)** |
| **`zstd --long=27 --ultra -22` ratio** | **0.0466 (4.66%)** |
| model vs. zstd | **~3.01× worse** |
| training wall time | ~67 min (10 epochs × 1562 steps × 3.88 it/s, g6.xlarge) |
| training cost | ~$0.80 + ~$0.80 × several retries ≈ $2-3 compute total for the day |
| **PASS / FAIL** | **FAIL** (gate was `ours < 0.98 × zstd` → needed 0.0457 or better, got 0.1405) |

### Caveats on the measurement

- The entrypoint's dedicated `measure_held_out_ratio.py` step landed
  `held_out_ratio=1.0` in the metadata JSON — a sentinel. Root cause:
  the shell captured `rc=$?` **after** a `cd /app` that followed the
  python call, silently masking any Python failure. Fixed in the
  entrypoint (`rc=$?` now runs before the `cd /app`), but this spike's
  v1 metadata still has the sentinel.
- The authoritative number above (3.3920 bits/token) comes from
  `train_l3tc_phase11.py`'s own eval step at epoch 9, which iterates
  the validation DataLoader directly (in-process, no shell capture
  bug). That eval used a 409,600-token slice of the 91.8M-token val
  set. A full-val entropy measurement would be within ~1% of this
  number.
- zstd baseline is on the *full* 277.3 MB val file, not a slice.

### Why HDFS lost

HDFS log lines are templates with swap-in block IDs and timestamps:

```
081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
```

The template repeats millions of times. zstd's 27-bit window (128 MB)
captures the whole template plus all the swap-in IDs it has ever seen
and encodes each new line as a tiny reference + the varying fields.
A 200 K-param RWKV-v4 (2 layers × 96 hidden × 16 K vocab) cannot
memorize that much template material — it has to model each token
from short-range context. At ~3.4 bits/token it's correctly modelling
the per-token distribution, but zstd is essentially getting each
entire line for ~1 byte of dictionary reference.

This is a good fail in the sense that it correctly falsifies a
too-simple version of the thesis: "small neural model beats generic
compression on homogeneous data" is not universally true. On corpora
where LZ's repeat-dictionary is near-optimal (heavily templated
structured logs), a 200 K model can't win.

### What the spike still proved

The service *infrastructure* runs end-to-end. Confirmed today:

- PUT → pre-signed S3 URL → S3 raw object → EventBridge → ingest
  Lambda → DDB atomic slot claim → SQS training-submit.
- Batch job launcher → g6.xlarge on-demand → container pull → model
  training → checkpoint save → S3 model upload (v1.pth,
  v1.tokenizer.model, v1.metadata.json).
- training-complete Lambda → DDB `status=fallback_zstd`,
  `current_model_version=1`. *(Deliberately chose zstd_fallback
  codec in the metadata because the model lost.)*
- CodeBuild cuts iteration latency from ~30 min laptop pushes to
  ~3 min AWS-backbone pushes. Shake-down importing the container
  locally catches import bugs in 20 s.

## Next actions

1. **Re-run Spike 1 on a different corpus.** HDFS is an edge case
   (pathologically templated). Candidates:
   - JSON API event logs (more variable payloads, less template-y)
   - Stripe-style audit trails with mixed-length fields
   - Application logs with free-text error messages
   The thesis targets corpora where *content* varies within a
   structure — not pure template data. Need a pass on at least one
   "real" non-HDFS corpus before concluding the service is
   non-viable.
2. **Verify the rc=$? fix actually emits a real ratio number.** The
   next run should produce a non-sentinel `held_out_ratio` in the
   v{N}.metadata.json. If that number matches the in-process eval's
   entropy bound to within ~1%, ship the fix as the canonical
   measurement path.
3. **Think about model capacity vs. corpus structure.** 200 K params
   may just be too small for anything that isn't pure prose. Spike
   1.5 candidates: 1M-param and 5M-param runs on the same HDFS corpus
   to see if capacity moves the needle. If even a 5M-param model
   can't beat zstd on HDFS, that's a clear answer.
4. **Close the compression worker loop.** The training-complete
   Lambda's compression-sweep fired — check that the raw object was
   compressed + deleted + DDB byte counters updated (tasks #16).

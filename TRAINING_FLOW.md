# Training flow — end-to-end walk through one job

What happens inside a single AWS Batch training job from container
boot to the model landing in S3. Verified against
`cdk/docker/training/train_entrypoint.sh`,
`scripts/train_l3tc_phase11.py`,
`scripts/measure_held_out_ratio.py`,
`cdk/src/handlers/training-complete.ts`,
`cdk/docker/compression/compression_worker.py` as of 2026-04-21.

## Job inputs

Set by the training-launcher Lambda via Batch container overrides:

| env var | value |
|---|---|
| `CUSTOMER_ID` | e.g. `acme` |
| `DATASET_ID` | e.g. `hdfs-spike` |
| `BUCKET_NAME` | `archive-dev-archive` |
| `S3_RAW_PREFIX` | `{cid}/{dsid}/raw/` |
| `S3_MODEL_PREFIX` | `{cid}/{dsid}/models/` |
| `TRIGGER` | `initial` or `retrain` |
| `DOMAIN` | tokenizer preset label; defaults to `logs` (the label is currently cosmetic — see PRODUCTION_TODO item 8) |

## 1. Stage-in (~10 s)

- `aws s3 sync s3://${BUCKET_NAME}/${S3_RAW_PREFIX}` → `/tmp/raw/`
  pulls every `.ndjson` under the dataset's raw prefix.
- `cat /tmp/raw/*.ndjson > /tmp/model/corpus.txt` concatenates them.
- 80 / 20 split by byte offset → `train.txt` + `val.txt`.
- Listing existing `s3://.../models/v*.pth` picks the next free
  version number (`v1` on first run).

## 2. SPM tokenizer (~2-3 min)

Runs `scripts/train_specialist_tokenizer.py --domain $DOMAIN
--corpus train.txt --output-dir /tmp/model --sample-mb 200`.

- Subsamples up to 200 MB of `train.txt` (speed).
- Trains a **16 K-vocab SentencePiece unigram** tokenizer with the
  lossless-round-trip settings:
  - `byte_fallback=True`, `character_coverage=1.0` — every UTF-8 byte
    must be representable, even unseen ones, via `<0xNN>` pieces.
  - `normalization_rule_name="identity"` — NFKC disabled so `[INFO]`,
    quotes, non-ASCII survive encode/decode untouched.
  - `split_by_unicode_script=False`, `split_digits=False` — let the
    trainer merge domain-specific patterns (IPs, timestamps, hex).
  - `\n` and `\t` registered as protected `user_defined_symbols` so
    they appear literally in the vocab, not collapsed.
- Outputs: `/tmp/model/spm.model`, `spm.vocab`, `spm.bt_report.json`
  (bytes-per-token on the sample).
- The `--domain` flag is currently just a label for the output
  directory and report JSON; it does not change SPM hyperparameters.

## 3. RWKV-v4 training (~45-60 min on A10G)

Runs `scripts/train_l3tc_phase11.py` with **cwd = `/app/vendor/L3TC`**
so the vendor's relative JIT-extension paths
(`models/RWKV_V4/cuda/wkv_op.cpp`) resolve.

### Model

`RWKV_TC_HIRA` from `vendor/L3TC/models/RWKV_V4/rwkv_tc_hira_train.py`.
This is L3TC's RWKV-v4 + HiRA low-rank residual architecture. We
import only the model class — everything else (dataset, loop,
optimizer, schedule, eval, checkpointing) is our own code in
`train_l3tc_phase11.py`.

Hyperparameters (defaults; overridable via CLI):

| param | value | notes |
|---|---|---|
| `num_hidden_layers` | 2 | L3TC-200K |
| `hidden_size` | 96 | L3TC-200K |
| `intermediate_size` | 96 | L3TC-200K |
| `rwkv_rank` | 4 | HiRA low-rank dim |
| `vocab_size` | 16384 | matches SPM |
| `ctx_len` | 2048 | fixed context |
| total params | ~200 K | |

Bootstrap note: on the way to importing `rwkv_tc_hira_train`, we import
`rwkv_v4_train` transitively; both do `from deepspeed.ops.adam import
FusedAdam` at module top. Real deepspeed wants nvcc + CUDA_HOME at
*import* time. Our trainer never calls `FusedAdam` — it builds its
own `torch.optim.AdamW` optimizer — so the Dockerfile writes a stub
deepspeed/pkuseg module into site-packages (mirrors phase11's
`phase11_write_stubs`), and the vendor's own `try: FusedAdam(...) except:
torch.optim.Adam(...)` handles the runtime fallback.

### Data pipeline

- Tokenize `train.txt` + `val.txt` with `spm.model`, cached to `.npy`
  binary files.
- `L3TCTokenDataset` memory-maps the cache. Each `__getitem__` picks
  a random offset into the token stream and returns a 2048-token
  `(input, target, input_types, output_types)` quad, shifted by one
  position for next-token prediction. Types are 0 (pad) / 1 (real).
- `DataLoader` with batch 32, 2 workers, pinned memory.

### Optimizer + schedule

- `torch.optim.AdamW(lr=1e-4, betas=(0.9, 0.99), eps=1e-8,
  weight_decay=0.01)` — **not** FusedAdam.
- LR schedule: linear warmup for 500 steps, then cosine annealing
  down to `lr_min=1e-6` over the remaining steps. Replaces L3TC's
  original broken double-stepping StepLR.
- `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)`
  before each optimizer step.

### Training loop

- 10 epochs × 50 000 samples per epoch = 500 000 gradient steps
  (default; `--epochs` + `--epoch-length` override).
- Precision: bf16 mixed precision (`autocast(bfloat16)`).
- `torch.compile` disabled for this run (`--no-compile`) — the WKV
  CUDA kernel doesn't play well with graph capture.
- Loss = cross-entropy on next-token prediction + L3TC's `L2Wrap`
  activation-norm regularizer (pushes max-logit toward 0 for
  stability; back-prop adds a small gradient in that direction).
- Per-step log format:
  `step=X epoch=Y lr=Z loss=L bits/token=B/T`. Evaluated every
  1 000 steps on a small held-out slice.

### Checkpointing

- `save_checkpoint` writes `{output_dir}/checkpoint_latest.pth` with
  `model_state_dict`, `optimizer.state_dict()`, `epoch`, `step`,
  `best_val_loss`, and the args namespace.
- Overwritten every epoch and any time val loss improves.
- **File name is `checkpoint_latest.pth`** (not `_final`) —
  important: the entrypoint reads that exact path after training
  exits.

## 4. Ratio measurement (~2-3 min)

Runs `scripts/measure_held_out_ratio.py` with **cwd = `/app/vendor/L3TC`**
(same JIT path constraint).

- Loads `checkpoint_latest.pth` into `build_model(device,
  num_layers=2, vocab_size=16384)`, moves to GPU, `eval()`.
- Reads raw bytes of `val.txt` → `original_bytes`.
- Tokenizes via `spm.model` → token stream.
- Segments the stream into non-overlapping windows of 2048 tokens
  (drops the final partial window; bias is negligible on a 278 MB
  val set).
- For each segment:
  - Forward pass, output `(1, seq, vocab)` logits.
  - `log_softmax` along the vocab axis.
  - `gather` the log-prob of the true next token at each of the
    2047 positions.
  - Convert nats → bits: `-log_prob / ln(2)`, sum.
- Accumulate `total_neg_log2_bits` and `total_positions`.
- `bits_per_token = total_neg_log2 / total_positions`.
- `bytes_per_token = original_bytes / total_tokens`.
- **`held_out_ratio = (bits_per_token / 8) / bytes_per_token`** —
  the Shannon entropy bound. A real arithmetic coder running on the
  same model produces compressed output within ~1% of this number
  for sequences > a few KB.
- Prints the ratio on stdout (captured into the entrypoint's
  `$held_out_ratio` shell var), diagnostics on stderr.

## 5. zstd baseline + decision (~30 s)

- `zstd --long=27 --ultra -22 --stdout --quiet val.txt | wc -c`
  → `compressed_bytes` of the exact same val split.
- `zstd_baseline_ratio = compressed_bytes / original_bytes`.
- Spike 1 gate:
  `would_have_beaten_zstd = (held_out_ratio > 0) &&
                            (held_out_ratio < 0.98 × zstd_baseline_ratio)`.
- Spike 1 always sets `codec = zstd_fallback` (the actual stored
  bytes are zstd no matter what). The model's ratio is recorded in
  metadata but doesn't yet drive stored compression; that wiring is
  Spike 2.

## 6. S3 upload + metadata emit

- `aws s3 cp /tmp/model/train_out/checkpoint_latest.pth
  s3://${BUCKET}/${S3_MODEL_PREFIX}v${VERSION}.pth`
- `aws s3 cp /tmp/model/spm.model
  s3://.../v${VERSION}.tokenizer.model`
- `aws s3 cp /tmp/model/metadata.json
  s3://.../v${VERSION}.metadata.json`

metadata.json shape:
```json
{
  "version": 1,
  "corpus_bytes": 1386685235,
  "codec": "zstd_fallback",
  "would_have_beaten_zstd": true,
  "held_out_ratio": 0.0xxx,
  "zstd_baseline_ratio": 0.0xxx,
  "trigger": "initial",
  "trained_at": "2026-04-21T...Z",
  "spike": "spike_1"
}
```

## Model at inference time

Training is Python; inference (compression) is Rust. `l3tc-rust/` is
the production inference runtime — a Rust port of the same
L3TC-200K RWKV-v4 architecture, with its own arithmetic coder,
checkpoint loader, and codec path. It reads a `.bin` checkpoint
(converted from the Python `.pth` via
`l3tc-rust/scripts/convert_checkpoint.py`) and runs compress /
decompress end-to-end.

**Spike 1 skips the Rust side deliberately** — the training job does
not produce a `.bin`, and the Fargate compression worker runs
`zstd --long=27 --ultra -22` regardless of what the model predicts.
The model ratio recorded in `metadata.json` is the Python-side
entropy bound; a real Rust arithmetic coder on the same model would
land within ~1% of that number.

Spike 2 (contingent on Spike 1 passing) is where the Rust runtime
gets wired in: training job emits `.pth` + `.bin` + tokenizer, the
Fargate worker image adds the Rust binary, and the worker picks
codec based on `metadata.codec == "l3tc"`. The GET endpoint gains a
streaming decode path through the same binary.

## 7. Post-training (handled outside the job)

- Batch exits 0 → EventBridge fires `Batch Job State Change`.
- `training-complete` Lambda:
  - Lists `s3://.../models/*.metadata.json`, picks the highest version.
  - Updates DDB: `status=ready`, `current_model_version=1`,
    `codec=zstd_fallback`.
  - Lists every `s3://.../raw/*.ndjson` still present and enqueues
    one compression SQS message per object (the "completion sweep" —
    covers raw data that arrived while training was running).
- Compression Fargate service (minTask=0, maxTask=4, scales on queue
  depth):
  - Pulls a message, downloads the raw NDJSON, runs
    `zstd --long=27 --ultra -22`, uploads to
    `s3://.../compressed/YYYY/MM/DD/HH/{uuid}.bin`,
    deletes the raw object, updates DDB byte counters
    (`compressed_bytes += N, raw_bytes_held -= N`).

## On failure

- `training-complete` Lambda on `status=FAILED` calls
  `recordTrainingFailed` which resets DDB `status=awaiting_corpus`.
  Raw data is untouched.
- Batch JobDefinition has `retryAttempts=1` today. See
  PRODUCTION_TODO item 4 for the retry-policy upgrade.

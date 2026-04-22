"""Compression worker for Krunch.

Polls the compression SQS queue. For each message:
  1. Download raw NDJSON from S3 → /tmp
  2. Read model metadata (codec field)
  3. Compress according to codec:
     - "zstd_fallback": whole-file zstd-22 --long=27 (no model needed)
     - "hybrid": invoke `l3tc hybrid-compress` with the trained
       zstd dict (and model+tokenizer once the .bin conversion lands)
  4. Upload compressed blob to S3 under compressed/YYYY/MM/DD/HH/
  5. Delete raw S3 object
  6. Update DynamoDB byte counters
  7. Emit CloudWatch EMF metrics (per-codec breakdown, savings vs
     zstd shadow, throughput) on hybrid runs so we can tune
     dispatcher behavior per customer without re-reading S3.

Deployed as ECS Fargate service via QueueProcessingFargateService.
Scales 0 → N based on queue depth.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import boto3  # type: ignore
from botocore.exceptions import ClientError  # type: ignore


BUCKET = os.environ["BUCKET_NAME"]
DATASETS_TABLE = os.environ["DATASETS_TABLE_NAME"]
MODEL_VERSIONS_TABLE = os.environ["MODEL_VERSIONS_TABLE_NAME"]
ENV = os.environ.get("ENV", "dev")

s3 = boto3.client("s3")
sqs = boto3.client("sqs")
ddb = boto3.client("dynamodb")
cw = boto3.client("cloudwatch")

WORKDIR = Path("/tmp/work")
WORKDIR.mkdir(parents=True, exist_ok=True)

# Path to the Rust l3tc binary. The Dockerfile's multi-stage build
# installs it here (follow-up). If missing at container start, the
# hybrid path gracefully falls back to zstd so Spike-1-era datasets
# still get compressed; a WARN is logged for operations.
L3TC_BIN = os.environ.get("L3TC_BIN", "/usr/local/bin/l3tc")


def download_model_metadata(cid: str, dsid: str, version: int) -> dict:
    """Read v{N}.metadata.json from S3 to determine codec choice."""
    key = f"{cid}/{dsid}/models/v{version}.metadata.json"
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    return json.loads(resp["Body"].read())


def compress_zstd(raw_path: Path, out_path: Path) -> None:
    """Compress raw NDJSON with zstd-22 --long=27 (maximum ratio)."""
    subprocess.run(
        ["zstd", "--long=27", "--ultra", "-22", "-q", "-f",
         "-o", str(out_path), str(raw_path)],
        check=True,
    )


def download_optional(cid: str, dsid: str, version: int, suffix: str, dest: Path) -> bool:
    """Download an optional model artifact (tokenizer / dict / bin) to
    `dest`. Returns True on success, False if the object is missing
    (404) — other S3 errors re-raise."""
    key = f"{cid}/{dsid}/models/v{version}.{suffix}"
    try:
        s3.download_file(BUCKET, key, str(dest))
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey"):
            return False
        raise


def compress_hybrid(
    raw_path: Path,
    out_path: Path,
    cid: str,
    dsid: str,
    version: int,
    work: Path,
    chunk_size_bytes: int,
) -> dict:
    """Invoke `l3tc hybrid-compress` with whichever optional assets
    (zstd dict, RWKV model + tokenizer) the training job uploaded.

    Returns the parsed stats.json on success. Raises on any failure
    (caller is responsible for the fallback path).

    The l3tc binary must be on PATH at `L3TC_BIN`. If it's not
    present the caller should catch the resulting FileNotFoundError
    and fall back to `compress_zstd`.
    """
    dict_path = work / "zstd_dict.bin"
    stats_path = work / "stats.json"
    tokenizer_path = work / "tokenizer.model"
    model_bin_path = work / "model.bin"

    has_dict = download_optional(cid, dsid, version, "zstd_dict", dict_path)
    # The Rust .bin model isn't produced yet (requires
    # convert_checkpoint.py to be wired into the training job). When
    # that lands, pair it with the tokenizer here; until then we run
    # hybrid without the neural codec in the menu.
    has_bin = download_optional(cid, dsid, version, "bin", model_bin_path)
    has_tok = False
    if has_bin:
        has_tok = download_optional(
            cid, dsid, version, "tokenizer.model", tokenizer_path
        )

    cmd = [
        L3TC_BIN,
        "hybrid-compress",
        str(raw_path),
        "-o",
        str(out_path),
        "--stats",
        str(stats_path),
        "--chunk-size",
        str(chunk_size_bytes),
    ]
    if has_dict:
        cmd += ["--zstd-dict", str(dict_path)]
    if has_bin and has_tok:
        cmd += ["--model", str(model_bin_path), "--tokenizer", str(tokenizer_path)]

    print(f"[worker] hybrid-compress cmd: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    with open(stats_path) as f:
        return json.load(f)


def emit_compression_metrics(
    cid: str,
    dsid: str,
    version: int,
    codec_used: str,
    raw_bytes: int,
    compressed_bytes: int,
    elapsed_seconds: float,
    hybrid_stats: dict | None,
) -> None:
    """Write CloudWatch EMF records capturing every compression run —
    hybrid OR zstd-fallback. EMF is cheaper than PutMetricData for
    high-cardinality dimensions and lets us query per-customer /
    per-codec breakdowns without a separate metrics pipeline.

    Schema:
      namespace = Krunch/Hybrid (kept for continuity; covers
                  all compression runs, not just successful hybrid)
      main record:
        dimensions = [CustomerId, DatasetId, Env]
        metrics    = Ratio, SavingsVsZstdPct, ThroughputMBps,
                     SafetyNetSubstitutions, ChunksTotal,
                     BytesIn, BytesOut
      per-codec record (one per codec tag that appeared):
        dimensions = [CustomerId, DatasetId, Env, Codec]
        metrics    = BytesByCodec, ChunksByCodec

    Fallback-path runs still land here so the dashboard shows total
    traffic per customer even when the hybrid path disables itself.
    In that case Codec=zstd_fallback, SavingsVsZstdPct=0, and the
    hybrid-specific fields (SafetyNetSubstitutions) report 0.
    """
    if hybrid_stats is not None:
        # Hybrid path populated every field; use it verbatim.
        per_codec_bytes = hybrid_stats.get("per_codec_bytes", {}) or {}
        per_codec_chunks = hybrid_stats.get("per_codec_chunks", {}) or {}
        stats = hybrid_stats
    else:
        # Fallback path: synthesize a stats dict that looks like a
        # single-codec hybrid run so downstream widgets render the
        # same way for both paths.
        ratio = compressed_bytes / max(1, raw_bytes)
        tput = (raw_bytes / 1_048_576.0) / max(elapsed_seconds, 1e-6)
        stats = {
            "ratio": ratio,
            "zstd_shadow_ratio": ratio,  # we ARE zstd in this path
            "savings_vs_zstd_pct": 0.0,
            "throughput_mb_per_sec": tput,
            "safety_net_substitutions": 0,
            "chunks_total": 1,
            "bytes_in": raw_bytes,
            "bytes_out": compressed_bytes,
            "encode_seconds": elapsed_seconds,
        }
        per_codec_bytes = {codec_used: compressed_bytes}
        per_codec_chunks = {codec_used: 1}

    # Metric definitions the dashboard will look for. We emit every
    # metric unconditionally so the absence of a codec still shows as
    # zero (not missing), which makes the "savings by codec" chart
    # legible when neural is disabled on some customers.
    metric_defs = [
        {"Name": "Ratio", "Unit": "None"},
        {"Name": "SavingsVsZstdPct", "Unit": "Percent"},
        {"Name": "ThroughputMBps", "Unit": "None"},
        {"Name": "SafetyNetSubstitutions", "Unit": "Count"},
        {"Name": "ChunksTotal", "Unit": "Count"},
        {"Name": "BytesIn", "Unit": "Bytes"},
        {"Name": "BytesOut", "Unit": "Bytes"},
    ]
    values: dict = {
        "Ratio": stats.get("ratio", 0.0),
        "SavingsVsZstdPct": stats.get("savings_vs_zstd_pct", 0.0),
        "ThroughputMBps": stats.get("throughput_mb_per_sec", 0.0),
        "SafetyNetSubstitutions": stats.get("safety_net_substitutions", 0),
        "ChunksTotal": stats.get("chunks_total", 0),
        "BytesIn": stats.get("bytes_in", 0),
        "BytesOut": stats.get("bytes_out", 0),
    }

    # One per-codec "BytesByCodec" metric with Codec as a dimension
    # so CloudWatch can group by codec cheaply. EMF allows nested
    # dimension sets; we use two sets — one keyed just by customer/
    # dataset (for total trends) and one that additionally keys by
    # codec (for the stacked-area view).
    per_codec_records = []
    for codec_name in set(per_codec_bytes) | set(per_codec_chunks):
        per_codec_records.append(
            {
                "codec": codec_name,
                "BytesByCodec": per_codec_bytes.get(codec_name, 0),
                "ChunksByCodec": per_codec_chunks.get(codec_name, 0),
            }
        )

    emf = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": "Krunch/Hybrid",
                    "Dimensions": [["CustomerId", "DatasetId", "Env"]],
                    "Metrics": metric_defs,
                }
            ],
        },
        "CustomerId": cid,
        "DatasetId": dsid,
        "Env": ENV,
        "ModelVersion": version,
        **values,
    }
    # Printing EMF to stdout is how the CloudWatch agent / awslogs
    # driver picks it up for free — no PutMetricData call needed.
    print(json.dumps(emf))

    # Emit per-codec records under a separate EMF payload so the
    # dimension set can include Codec. Keeping them as distinct EMF
    # entries is simpler than trying to declare two dimension sets
    # in one payload (which the EMF spec allows but is brittle in
    # Lambda log group views).
    for rec in per_codec_records:
        emf_codec = {
            "_aws": {
                "Timestamp": int(time.time() * 1000),
                "CloudWatchMetrics": [
                    {
                        "Namespace": "Krunch/Hybrid",
                        "Dimensions": [["CustomerId", "DatasetId", "Env", "Codec"]],
                        "Metrics": [
                            {"Name": "BytesByCodec", "Unit": "Bytes"},
                            {"Name": "ChunksByCodec", "Unit": "Count"},
                        ],
                    }
                ],
            },
            "CustomerId": cid,
            "DatasetId": dsid,
            "Env": ENV,
            "Codec": rec["codec"],
            "BytesByCodec": rec["BytesByCodec"],
            "ChunksByCodec": rec["ChunksByCodec"],
        }
        print(json.dumps(emf_codec))


def process_message(body: dict) -> None:
    cid = body["cid"]
    dsid = body["dsid"]
    raw_key = body["s3_raw_key"]
    model_version = body["model_version"]

    work = WORKDIR / uuid4().hex
    work.mkdir(parents=True)
    raw_local = work / "raw.ndjson"
    compressed_local = work / "compressed.bin"

    print(f"[worker] processing {cid}/{dsid} raw={raw_key} model=v{model_version}")

    try:
        metadata = download_model_metadata(cid, dsid, model_version)
    except Exception as e:
        print(f"[worker] WARN: could not read model metadata ({e}); assuming zstd_fallback")
        metadata = {"codec": "zstd_fallback"}
    codec = metadata.get("codec", "zstd_fallback")
    # 1 MB default matches RUST_DISPATCHER_BENCH.md's sweet spot
    # (+7.9% on prose, +20% on logs vs whole-file zstd-22). Datasets
    # with unusual distributions can override by writing a different
    # chunk_size_bytes into their metadata.
    chunk_size_bytes = int(metadata.get("chunk_size_bytes", 1048576))

    s3.download_file(BUCKET, raw_key, str(raw_local))
    raw_bytes = raw_local.stat().st_size

    hybrid_stats: dict | None = None
    codec_used = codec  # actual codec the output ended up using
    compress_started_at = time.time()
    if codec in ("hybrid", "l3tc"):
        if codec == "l3tc":
            print("[worker] codec=l3tc is legacy; routing to hybrid")
        # Tier 1 hybrid path. Binary may be absent in older compression
        # images — fall back to zstd with a loud warning so we never
        # silently skip compression.
        if not shutil.which(L3TC_BIN) and not Path(L3TC_BIN).exists():
            print(
                f"[worker] WARN: codec={codec} but {L3TC_BIN} not present; "
                f"falling back to zstd. Rebuild the compression image with "
                f"the Rust multi-stage build to enable hybrid."
            )
            compress_zstd(raw_local, compressed_local)
            codec_used = "zstd_fallback"
        else:
            try:
                hybrid_stats = compress_hybrid(
                    raw_local,
                    compressed_local,
                    cid,
                    dsid,
                    model_version,
                    work,
                    chunk_size_bytes,
                )
                codec_used = "hybrid"
            except subprocess.CalledProcessError as e:
                print(
                    f"[worker] hybrid-compress failed (rc={e.returncode}); "
                    f"falling back to zstd"
                )
                compress_zstd(raw_local, compressed_local)
                codec_used = "zstd_fallback"
    else:
        # zstd_fallback — the Spike 1 / pre-hybrid path.
        compress_zstd(raw_local, compressed_local)
        codec_used = "zstd_fallback"

    compressed_bytes = compressed_local.stat().st_size
    elapsed_seconds = time.time() - compress_started_at

    # Always emit metrics — hybrid and zstd-fallback runs both land in
    # the same namespace keyed by Codec dimension, so the dashboard
    # can show per-customer traffic and the codec mix even when the
    # hybrid path isn't wired yet or fell back.
    try:
        emit_compression_metrics(
            cid,
            dsid,
            model_version,
            codec_used,
            raw_bytes,
            compressed_bytes,
            elapsed_seconds,
            hybrid_stats,
        )
    except Exception as e:
        print(f"[worker] WARN: EMF emission failed: {e}")

    # Time-bucket the compressed blob by upload time.
    now = datetime.now(timezone.utc)
    time_prefix = now.strftime("%Y/%m/%d/%H")
    blob_uuid = uuid4().hex
    compressed_key = f"{cid}/{dsid}/compressed/{time_prefix}/{blob_uuid}.bin"

    s3.upload_file(str(compressed_local), BUCKET, compressed_key)

    # Delete raw object once compressed is safely in S3.
    s3.delete_object(Bucket=BUCKET, Key=raw_key)

    # Update byte counters in DynamoDB.
    ddb.update_item(
        TableName=DATASETS_TABLE,
        Key={"pk": {"S": f"CUST#{cid}"}, "sk": {"S": f"DS#{dsid}"}},
        UpdateExpression="ADD compressed_bytes :c, raw_bytes_held :r",
        ExpressionAttributeValues={
            ":c": {"N": str(compressed_bytes)},
            ":r": {"N": str(-raw_bytes)},
        },
    )

    ratio = compressed_bytes / max(1, raw_bytes)
    print(
        f"[worker] done: raw={raw_bytes}B compressed={compressed_bytes}B "
        f"ratio={ratio:.4f} codec={codec} → {compressed_key}"
    )

    for p in work.iterdir():
        p.unlink(missing_ok=True)
    work.rmdir()


def set_task_protection(enabled: bool, expires_minutes: int = 60) -> None:
    """Tell the ECS agent whether this task is currently processing.

    When enabled, ECS refuses scale-in requests against this task
    regardless of what the auto-scaler says. That lets us run the
    compression service at min=0 / max=N without the SQS scale-in-
    based-on-Visible race killing tasks mid-job (see
    cdk/lib/compression-stack.ts comment for the background).

    Works via the agent endpoint at $ECS_AGENT_URI; Fargate platform
    1.4.0+ injects the env var automatically. Outside Fargate (local
    testing) the env var is absent and this is a no-op.

    Failures are logged and swallowed: if protection can't be set we'd
    rather finish the message than drop it, and a scale-in during
    processing is recoverable via the SQS redrive-policy.
    """
    agent_uri = os.environ.get("ECS_AGENT_URI")
    if not agent_uri:
        return
    body: dict = {"ProtectionEnabled": enabled}
    if enabled:
        body["ExpiresInMinutes"] = expires_minutes
    url = f"{agent_uri}/task-protection/v1/state"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
            print(f"[worker] task-protection={enabled}: {payload}")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"[worker] WARN: task-protection={enabled} request failed: {e}")


def get_queue_url() -> str:
    """Find the compression queue URL via several env-var conventions.

    CDK's QueueProcessingFargateService injects QUEUE_NAME (not
    QUEUE_URL), so the primary path is name -> url via SQS API.
    """
    for url_var in ("QUEUE_URL", "COMPRESSION_QUEUE_URL", "SQS_QUEUE_URL"):
        if os.environ.get(url_var):
            return os.environ[url_var]
    q_name = os.environ.get("QUEUE_NAME") \
             or f"archive-{os.environ.get('ENV', 'dev')}-compression"
    resp = sqs.get_queue_url(QueueName=q_name)
    return resp["QueueUrl"]


def main() -> None:
    queue_url = get_queue_url()
    print(f"[worker] starting, polling {queue_url}")
    while True:
        resp = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
            VisibilityTimeout=1800,
        )
        msgs = resp.get("Messages", [])
        if not msgs:
            continue

        for msg in msgs:
            # Mark the task protected for the lifetime of this
            # message. ECS honors the flag by refusing scale-in on
            # this task even when the auto-scaler wants to drop
            # desired=0 because Visible=0 in the queue. 60 min is a
            # dead-man's-switch: if we crash between setting
            # protection and clearing it, the flag expires naturally.
            set_task_protection(True, expires_minutes=60)
            try:
                body = json.loads(msg["Body"])
                process_message(body)
                sqs.delete_message(
                    QueueUrl=queue_url, ReceiptHandle=msg["ReceiptHandle"],
                )
            except Exception as e:
                print(f"[worker] failed processing message: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                time.sleep(5)
            finally:
                # Clear protection so the auto-scaler can scale to 0
                # once the queue drains.
                set_task_protection(False)


if __name__ == "__main__":
    main()

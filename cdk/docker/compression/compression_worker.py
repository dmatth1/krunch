"""Compression worker for Krunch (Spike 1 scope).

Polls the compression SQS queue. For each message:
  1. Download raw NDJSON from S3 → /tmp
  2. Read model metadata (codec field)
  3. For Spike 1: always compress with zstd-22 (codec=zstd_fallback).
     In a later spike we'll load the Rust l3tc binary for codec=l3tc.
  4. Upload compressed blob to S3 under compressed/YYYY/MM/DD/HH/
  5. Delete raw S3 object
  6. Update DynamoDB byte counters

Deployed as ECS Fargate service via QueueProcessingFargateService.
Scales 0 → N based on queue depth.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import boto3  # type: ignore


BUCKET = os.environ["BUCKET_NAME"]
DATASETS_TABLE = os.environ["DATASETS_TABLE_NAME"]
MODEL_VERSIONS_TABLE = os.environ["MODEL_VERSIONS_TABLE_NAME"]

s3 = boto3.client("s3")
sqs = boto3.client("sqs")
ddb = boto3.client("dynamodb")

WORKDIR = Path("/tmp/work")
WORKDIR.mkdir(parents=True, exist_ok=True)


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

    s3.download_file(BUCKET, raw_key, str(raw_local))
    raw_bytes = raw_local.stat().st_size

    if codec == "l3tc":
        # Spike 2+: use the Rust l3tc binary with the .bin model.
        # Spike 1 emits codec=zstd_fallback always so we never hit this.
        print(f"[worker] WARN: codec=l3tc requested but not implemented in spike 1; falling back to zstd")
        compress_zstd(raw_local, compressed_local)
    else:
        # zstd_fallback — the Spike 1 path.
        compress_zstd(raw_local, compressed_local)

    compressed_bytes = compressed_local.stat().st_size

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


if __name__ == "__main__":
    main()

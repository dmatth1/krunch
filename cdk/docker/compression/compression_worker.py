"""Compression worker for the learned-archive service.

Polls the compression SQS queue. For each message:
  1. Download raw NDJSON from S3 → /tmp
  2. Download model + tokenizer from S3 → /tmp
  3. Run l3tc compress → produces .bin output
  4. Upload compressed blob to S3 under compressed/YYYY/MM/DD/HH/
  5. Delete raw S3 object
  6. Update DynamoDB byte counters

Deployed as an ECS Fargate service via QueueProcessingFargateService.
The container scales 0 → N based on queue depth.
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
# QUEUE_NAME is set by QueueProcessingFargateService; we don't need it
# directly because Fargate is handling the long-poll for us via the
# AWS SDK default — but we wire this in case we want to do our own
# polling loop.

s3 = boto3.client("s3")
sqs = boto3.client("sqs")
ddb = boto3.client("dynamodb")

WORKDIR = Path("/tmp/work")
WORKDIR.mkdir(parents=True, exist_ok=True)


def process_message(body: dict) -> None:
    cid = body["cid"]
    dsid = body["dsid"]
    raw_key = body["s3_raw_key"]
    model_version = body["model_version"]

    work = WORKDIR / uuid4().hex
    work.mkdir(parents=True)
    raw_local = work / "raw.ndjson"
    model_local = work / "model.bin"
    tok_local = work / "tokenizer.model"
    compressed_local = work / "compressed.bin"

    print(f"[worker] processing {cid}/{dsid} raw={raw_key} model=v{model_version}")

    model_key = f"{cid}/{dsid}/models/v{model_version}.bin"
    tok_key = f"{cid}/{dsid}/models/v{model_version}.tokenizer.model"

    s3.download_file(BUCKET, raw_key, str(raw_local))
    s3.download_file(BUCKET, model_key, str(model_local))
    s3.download_file(BUCKET, tok_key, str(tok_local))

    raw_bytes = raw_local.stat().st_size

    # Run the Rust l3tc compressor.
    subprocess.run(
        [
            "l3tc",
            "compress",
            str(raw_local),
            "--model",
            str(model_local),
            "--tokenizer",
            str(tok_local),
            "-o",
            str(compressed_local),
        ],
        check=True,
        capture_output=True,
    )

    compressed_bytes = compressed_local.stat().st_size

    # Time-bucket the compressed blob by UPLOAD time. Later we'll
    # improve this by reading timestamps from inside the NDJSON so
    # range queries are accurate.
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
        UpdateExpression=(
            "ADD compressed_bytes :c, raw_bytes_held :r"
        ),
        ExpressionAttributeValues={
            ":c": {"N": str(compressed_bytes)},
            ":r": {"N": str(-raw_bytes)},
        },
    )

    print(
        f"[worker] done: raw={raw_bytes}B compressed={compressed_bytes}B "
        f"ratio={compressed_bytes / max(1, raw_bytes):.4f} → {compressed_key}"
    )

    # Clean up scratch.
    for p in work.iterdir():
        p.unlink(missing_ok=True)
    work.rmdir()


def main() -> None:
    queue_url = os.environ.get("QUEUE_URL")
    if not queue_url:
        # QueueProcessingFargateService injects the SQS queue URL in
        # the task role environment; the exact env var name isn't
        # guaranteed. Fall back to reading via EC2 metadata or set
        # manually.
        queue_url = os.environ["COMPRESSION_QUEUE_URL"]

    print(f"[worker] starting, polling {queue_url}")
    while True:
        resp = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
            VisibilityTimeout=1800,  # 30 min, matches queue config
        )
        msgs = resp.get("Messages", [])
        if not msgs:
            continue

        for msg in msgs:
            try:
                body = json.loads(msg["Body"])
                process_message(body)
                sqs.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=msg["ReceiptHandle"],
                )
            except Exception as e:
                print(f"[worker] failed processing message: {e}", file=sys.stderr)
                # Leave message in queue; will retry until maxReceiveCount,
                # then hit DLQ.
                time.sleep(5)


if __name__ == "__main__":
    main()

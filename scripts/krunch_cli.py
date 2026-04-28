#!/usr/bin/env python3
"""
krunch CLI — submit distributed compression jobs to AWS Batch.

Commands:
  krunch submit   --source <url> --dest <url> [--workers N] [--stack KrunchStack]
  krunch status   --job-id <id>
  krunch assemble --parts-prefix <url> --n-parts N --dest <url> --original-len N

Requires: boto3, aws credentials with BatchSubmitJob + S3 permissions.
"""

import argparse
import sys
import time
import json
import os


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

def cmd_submit(args):
    import boto3
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from server import url_io

    # Resolve job queue + job definitions from stack outputs if not provided
    job_queue, compress_jd, assemble_jd = _resolve_resources(
        args.stack, args.job_queue, args.compress_job_def, args.assemble_job_def
    )

    print(f"Source:      {args.source}")
    print(f"Dest:        {args.dest}")
    print(f"Workers:     {args.workers}")
    print(f"Job queue:   {job_queue}")

    # Get file size
    print("Getting file size...", end=" ", flush=True)
    total_size = url_io.size(args.source)
    print(f"{total_size:,} bytes ({total_size / 1024**3:.2f} GB)")

    # Parts prefix: dest + ".parts/" + job-<timestamp>/
    job_tag = f"job-{int(time.time())}"
    parts_prefix = f"{args.dest}.parts/{job_tag}"

    batch = boto3.client("batch")

    # ---------------------------------------------------------------------------
    # Submit compress array job
    # ---------------------------------------------------------------------------
    print(f"\nSubmitting compress array job ({args.workers} tasks)...")
    compress_resp = batch.submit_job(
        jobName=f"krunch-compress-{job_tag}",
        jobQueue=job_queue,
        jobDefinition=compress_jd,
        arrayProperties={"size": args.workers},
        containerOverrides={
            "environment": [
                {"name": "KRUNCH_JOB_TYPE",     "value": "compress"},
                {"name": "KRUNCH_SOURCE",        "value": args.source},
                {"name": "KRUNCH_TOTAL_SIZE",    "value": str(total_size)},
                {"name": "KRUNCH_TOTAL_TASKS",   "value": str(args.workers)},
                {"name": "KRUNCH_PARTS_PREFIX",  "value": parts_prefix},
            ]
        },
    )
    compress_job_id = compress_resp["jobId"]
    print(f"Compress job ID: {compress_job_id}")

    # ---------------------------------------------------------------------------
    # Poll compress job
    # ---------------------------------------------------------------------------
    print("Waiting for compress tasks to complete...")
    t0 = time.time()
    _poll_job(batch, compress_job_id, total_tasks=args.workers)
    compress_elapsed = time.time() - t0
    throughput = total_size / 1024 / compress_elapsed
    print(f"Compress done in {compress_elapsed:.0f}s ({throughput:.0f} KB/s aggregate)")

    # ---------------------------------------------------------------------------
    # Submit assemble job
    # ---------------------------------------------------------------------------
    print("\nSubmitting assemble job...")
    assemble_resp = batch.submit_job(
        jobName=f"krunch-assemble-{job_tag}",
        jobQueue=job_queue,
        jobDefinition=assemble_jd,
        dependsOn=[{"jobId": compress_job_id, "type": "SEQUENTIAL"}],
        containerOverrides={
            "environment": [
                {"name": "KRUNCH_JOB_TYPE",      "value": "assemble"},
                {"name": "KRUNCH_PARTS_PREFIX",  "value": parts_prefix},
                {"name": "KRUNCH_N_PARTS",       "value": str(args.workers)},
                {"name": "KRUNCH_DEST",          "value": args.dest},
                {"name": "KRUNCH_ORIGINAL_LEN",  "value": str(total_size)},
            ]
        },
    )
    assemble_job_id = assemble_resp["jobId"]
    print(f"Assemble job ID: {assemble_job_id}")

    _poll_job(batch, assemble_job_id, total_tasks=1)
    total_elapsed = time.time() - t0

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    try:
        dest_size = url_io.size(args.dest)
        ratio = dest_size / total_size
        print(f"\n{'='*50}")
        print(f"Done in {total_elapsed:.0f}s")
        print(f"  Original:   {total_size:,} bytes")
        print(f"  Compressed: {dest_size:,} bytes")
        print(f"  Ratio:      {ratio:.4f} ({ratio*100:.1f}% of original)")
        print(f"  Output:     {args.dest}")
    except Exception:
        print(f"\nDone in {total_elapsed:.0f}s → {args.dest}")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def cmd_status(args):
    import boto3
    batch = boto3.client("batch")
    resp = batch.describe_jobs(jobs=[args.job_id])
    if not resp["jobs"]:
        print(f"Job {args.job_id} not found")
        sys.exit(1)
    job = resp["jobs"][0]
    print(json.dumps({
        "jobId": job["jobId"],
        "jobName": job["jobName"],
        "status": job["status"],
        "statusReason": job.get("statusReason", ""),
        "createdAt": job.get("createdAt"),
        "startedAt": job.get("startedAt"),
        "stoppedAt": job.get("stoppedAt"),
    }, indent=2, default=str))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _poll_job(batch, job_id: str, total_tasks: int,
              poll_interval: int = 15):
    terminal = {"SUCCEEDED", "FAILED"}
    last_status = None
    while True:
        resp = batch.describe_jobs(jobs=[job_id])
        job = resp["jobs"][0]
        status = job["status"]

        if total_tasks > 1:
            arr = job.get("arrayProperties", {}).get("statusSummary", {})
            pending = arr.get("PENDING", 0) + arr.get("RUNNABLE", 0) + arr.get("STARTING", 0)
            running = arr.get("RUNNING", 0)
            succeeded = arr.get("SUCCEEDED", 0)
            failed = arr.get("FAILED", 0)
            line = (f"  {status}: pending={pending} running={running} "
                    f"succeeded={succeeded} failed={failed}")
        else:
            line = f"  {status}"
            if "statusReason" in job:
                line += f" — {job['statusReason']}"

        if line != last_status:
            print(line)
            last_status = line

        if status in terminal:
            if status == "FAILED":
                print(f"Job {job_id} FAILED", file=sys.stderr)
                sys.exit(1)
            return

        time.sleep(poll_interval)


def _resolve_resources(stack_name, job_queue, compress_jd, assemble_jd):
    """Look up resource names from CloudFormation stack outputs if not provided."""
    if job_queue and compress_jd and assemble_jd:
        return job_queue, compress_jd, assemble_jd

    import boto3
    cf = boto3.client("cloudformation")
    resp = cf.describe_stacks(StackName=stack_name)
    outputs = {o["OutputKey"]: o["OutputValue"]
               for o in resp["Stacks"][0].get("Outputs", [])}

    return (
        job_queue   or outputs.get("JobQueueArn")   or _fail("--job-queue or --stack required"),
        compress_jd or outputs.get("CompressJobDef") or _fail("--compress-job-def or --stack required"),
        assemble_jd or outputs.get("AssembleJobDef") or _fail("--assemble-job-def or --stack required"),
    )


def _fail(msg):
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(prog="krunch",
                                     description="Krunch distributed compression CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # submit
    p_submit = sub.add_parser("submit", help="Submit a compression job")
    p_submit.add_argument("--source", required=True,
                          help="Source URL (s3://, http://, file://)")
    p_submit.add_argument("--dest", required=True,
                          help="Destination URL for compressed output")
    p_submit.add_argument("--workers", type=int, default=1,
                          help="Number of parallel compress tasks (default: 1)")
    p_submit.add_argument("--spot", action=argparse.BooleanOptionalAction,
                          default=True,
                          help="Use spot instances (default: --spot)")
    p_submit.add_argument("--stack", default="KrunchStack",
                          help="CloudFormation stack name (default: KrunchStack)")
    p_submit.add_argument("--job-queue",
                          help="Batch job queue ARN or name (overrides --stack lookup)")
    p_submit.add_argument("--compress-job-def",
                          help="Batch job definition for compress tasks")
    p_submit.add_argument("--assemble-job-def",
                          help="Batch job definition for assemble task")

    # status
    p_status = sub.add_parser("status", help="Check job status")
    p_status.add_argument("--job-id", required=True)

    args = parser.parse_args()
    if args.command == "submit":
        cmd_submit(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()

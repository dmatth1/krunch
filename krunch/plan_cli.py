"""In-image entry point for `krunch plan`.

The host-side `scripts/krunch` wrapper docker-runs this so the user
doesn't have to install the Python package locally — they only need
docker + the wrapper script. Templates live alongside this module
inside the image, so host and image never go out of sync.

Usage (inside the image):
    python -m krunch.plan_cli --target aws-batch ...
"""

from __future__ import annotations

import argparse
import sys
import time

from . import plan as krunch_plan


def main(argv=None):
    p = argparse.ArgumentParser(prog="krunch.plan_cli")
    p.add_argument("--target", required=True,
                   choices=sorted(krunch_plan.TARGETS.keys()))
    p.add_argument("--mode", default="compress",
                   choices=["compress", "decompress"])
    p.add_argument("--source", "--input", dest="input_url", required=True)
    p.add_argument("--dest", "--output", dest="output_url", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--image", required=True)
    p.add_argument("--queue", default="krunch-queue")
    p.add_argument("--job-definition", default="krunch-job")
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--memory-mb", type=int, default=16384)
    p.add_argument("--timeout-s", type=int, default=3600)
    p.add_argument("--input-len", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    input_len = args.input_len
    if input_len == 0:
        try:
            from . import url_io
            input_len = url_io.size(args.input_url)
        except Exception as e:  # noqa: BLE001
            print(f"warn: couldn't stat {args.input_url}: {e}; "
                  f"pass --input-len explicitly", file=sys.stderr)

    run_id = args.run_id or f"{int(time.time())}"
    ctx = {
        "target":            args.target,
        "mode":              args.mode,
        "input_url":         args.input_url,
        "output_url":        args.output_url,
        "input_len":         input_len,
        "n_workers":         args.workers,
        "n_workers_minus_1": max(0, args.workers - 1),
        "image":             args.image,
        "queue":             args.queue,
        "job_definition":    args.job_definition,
        "cpus":              args.cpus,
        "gpus":              args.gpus,
        "memory_mb":         args.memory_mb,
        "timeout_s":         args.timeout_s,
        "slurm_time":        "01:00:00",
        "run_id":            run_id,
    }
    try:
        rendered = krunch_plan.render(args.target, ctx)
        krunch_plan.validate(args.target, rendered)
    except ValueError as e:
        print(f"plan render failed: {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        # Validated; CI consumes the exit code only.
        return 0
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())

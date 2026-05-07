# Contributing

Two areas where contributions are most useful right now.

## 1. Batch-system support

AWS Batch is the only orchestrator shipped today. We'd like to support
the rest. Each row below means: writing a Jinja template under
`src/krunch/plan/templates/`, registering it in `src/krunch/plan/__init__.py`,
and validating it end-to-end against the real orchestrator.

| Target | Status |
|---|---|
| AWS Batch | ✅ shipped, end-to-end validated |
| Kubernetes | ❌ wanted |
| Modal | ❌ wanted |
| Ray | ❌ wanted |
| Slurm | ❌ wanted |
| GCP Batch | ❌ wanted |
| Local (single host) | ❌ wanted |

The worker contract is documented in `README.md` ("Adding a new batch
target") — `src/krunch/job.py` reads a small set of env vars
(`KRUNCH_INPUT_URL`, `KRUNCH_PART_INDEX`, `KRUNCH_PART_COUNT`, …) that
any framework can inject.

## 2. Compress / decompress speed and ratio

Faster kernels and better compression ratios are always welcome. The
WKV / det-matmul / W8A8 paths are the hot spots; ratio is bounded by
the model, so improvements come from a better predictor or better
context handling. Microbench against the production kernel on the
production hardware before claiming an X× projection.

---

Open a PR; CI runs unit tests + CDK type-check. License is Apache-2.0.

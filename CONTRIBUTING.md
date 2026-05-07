# Contributing

Two areas are the most useful right now.

## 1. Batch-target validation

`krunch plan` emits artifacts for seven batch targets. Only **AWS Batch**
has been run end-to-end. The rest render + schema-validate in CI but
haven't been launched against a real orchestrator. If you run krunch
on one of these and hit a bug in the rendered artifact, file an issue
or PR with the fix and a small repro.

| Target | Template renders | Schema validates | End-to-end run |
|---|---|---|---|
| AWS Batch | ✅ | ✅ | ✅ |
| Kubernetes | ✅ | ✅ | ❌ |
| Modal | ✅ | ✅ | ❌ |
| Ray | ✅ | ✅ | ❌ |
| Slurm | ✅ | ✅ | ❌ |
| GCP Batch | ✅ | ✅ | ❌ |
| Local (single host) | ✅ | ✅ | ❌ |

## 2. Compress / decompress speed and ratio

Faster kernels and better compression ratios are always welcome. The
WKV / det-matmul / W8A8 paths are the hot spots; ratio is bounded by
the model, so improvements come from a better predictor or better
context handling. Microbench against the production kernel on the
production hardware before claiming an X× projection.

Open a PR; CI runs unit tests + CDK type-check. License is Apache-2.0.

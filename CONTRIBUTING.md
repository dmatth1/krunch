# Contributing to Krunch

Thanks for taking a look. Krunch is pre-1.0; we keep the surface small
on purpose, so the highest-value contributions right now are real-world
validation of paths we haven't been able to run end-to-end ourselves.

## How krunch is shipped

Distribution is one Docker image (`ghcr.io/dmatth1/krunch:latest`) plus
a thin shell wrapper installed via `install.sh`. The codec library and
kernels live in this repo; the model weights and CUDA build are baked
into the image at publish time.

To make a change:

1. Edit the Python under `krunch/`, the C++/CUDA under `krunch_ac/`,
   the wrapper at `scripts/krunch`, or the plan templates at
   `krunch/plan/templates/`.
2. Run the unit tests: `pip install -r requirements-dev.txt && PYTHONPATH=. pytest tests/unit/ -v`.
3. Open a PR. CI runs unit tests + CDK type-check + a `krunch plan`
   smoke render automatically.
4. For kernel changes: also run the relevant script under `tests/gpu/`
   inside the published image (see `tests/gpu/README.md`).
5. For end-to-end validation: `tests/integration/quick.sh` (free, fast)
   and, if you have AWS access, `tests/integration/batch.sh`.

## Batch-target validation matrix

`krunch plan` currently emits artifacts for seven batch targets. Only
**AWS Batch** has been run end-to-end — the rest are
template-rendered + schema-validated in CI but never actually
launched against the real orchestrator. **High-value contributions
welcome here:** if you run krunch on Modal/Ray/k8s/Slurm/GCP Batch
and hit a bug in the rendered artifact, file an issue or PR with the
fix and a small repro.

| Target | Template renders | Schema validates | End-to-end run | Notes |
|---|---|---|---|---|
| AWS Batch | ✅ | ✅ | ✅ | 100 MB / 4 workers, byte-exact roundtrip on g5.xlarge |
| Kubernetes | ✅ | ✅ | ❌ | needs validation against a real cluster (GKE/EKS/local kind) |
| Modal | ✅ | ✅ | ❌ | needs `modal run` validation with real GPUs |
| Ray | ✅ | ✅ | ❌ | needs validation against a real Ray cluster |
| Slurm | ✅ | ✅ | ❌ | needs validation on an HPC scheduler |
| GCP Batch | ✅ | ✅ | ❌ | needs validation in a GCP project |
| Local (single host) | ✅ | ✅ | ❌ | drop-in shell script for non-cluster testing |

If you're adding a new batch system entirely, the worker contract is
documented in `README.md` ("Adding a new batch target") — `krunch/job.py`
reads a small set of env vars (`KRUNCH_INPUT_URL`, `KRUNCH_PART_INDEX`,
`KRUNCH_PART_COUNT`, …) that any framework can inject. Adding a new
target means writing a new Jinja template under
`krunch/plan/templates/` and a corresponding `TARGETS` entry in
`krunch/plan/__init__.py`.

## What kinds of contributions we're looking for

- **Real-orchestrator bug reports + fixes** for the six unvalidated
  targets above. Even "I tried `krunch plan --target modal` and the
  rendered artifact failed because of X" is useful.
- **Ratio benchmarks** on corpora we don't have numbers for yet
  (enwik8/9, nginx logs, code corpora, your domain-specific data).
  See `README.md` "Ratio comparisons" — table is mostly `_tbd_`.
- **Throughput on hardware we haven't measured.** README chart has
  data only for 1× and 4× A10G; the dotted lines are linear
  extrapolation. If you run on g6e (L40S), p5 (H100), or something
  else, share the numbers (`tests/integration/gpu.sh` reports them).
- **Bug fixes + tests for the codec itself.** `tests/unit/` is where
  the pure-Python invariants live (blob format, byte ranges, UTF-8
  boundaries, plan rendering). Add a regression test for any bug
  you fix.
- **CUDA kernel work** on the W8A8 / det-matmul / WKV paths. The
  performance gates and what's been tried live in the Tier 3
  optimization log; the 5 missed-projection lessons are summarized
  in `CLAUDE.md`. tl;dr — microbench against the production kernel
  on the production hardware before claiming an X× projection.

## What we won't merge

- New dependencies in the codec hot path without strong justification
  — we keep the runtime image lean (3.5 GB is already a lot).
- Per-tenant fine-tuning hooks. That's v2; in v1 the model is fixed
  RWKV-4-Pile-169M.
- Always-on hosted server modes. v1 is one-shot CLI; v2 will introduce
  a hosted offering separately.

## Reporting bugs

GitHub issues. Include the krunch version (`krunch --version` once we
ship one — for now, the image SHA from `docker inspect`), the GPU
type, the input size, and the exact command line. For correctness
bugs (roundtrip mismatch), the input bytes if you can share them, or
a deterministic synthetic that reproduces.

## License

By contributing, you agree your contribution is licensed under
Apache-2.0 (the project license). No CLA.

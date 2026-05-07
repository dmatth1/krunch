# Roadmap

What's next, in rough order. This is best-effort and not a commitment.

## v1.0 (next release)

The blocker for tagging v1.0 is closing out a few rough edges and
filling in benchmark gaps so the README's claims can stand without
caveats. None of these change the API.

- [ ] **Tier-4 sign-off.** AWS Batch end-to-end works (100 MB / 4
  workers byte-exact); the matching numbers are in the README chart.
  Need a final pass of the multi-worker UTF-8 byte-range fix on a
  larger corpus + a documented re-run procedure.
- [ ] **Ratio table fill-in.** README "Ratio comparisons" has measured
  numbers only for WildChat-English. Want enwik8, enwik9, an nginx
  log corpus, and The Stack (Python) before v1.0. Each row is one
  bench run on a g5.xlarge.
- [ ] **ts_zip head-to-head.** ts_zip is the closest peer codec; we
  haven't benched it on our corpus locally yet. Want a fair comparison
  on identical hardware before claiming "beats traditional codecs by
  20-40%" in the lead.
- [ ] **Cold-start measurement.** README footnote says cold-start "may
  increase runtimes" — replace with a measured number once we have one
  (current estimate is ~13 min on a fresh AWS Batch CE).
- [ ] **`krunch --version`.** Wrapper currently has no `--version`
  flag; needs to surface both the wrapper version and the image SHA so
  bug reports are actionable.

## v1.x (post-launch incremental)

- [ ] **Real-orchestrator validation** for the 6 unvalidated `krunch
  plan` targets (k8s, Modal, Ray, Slurm, GCP Batch, local). See
  `CONTRIBUTING.md` validation matrix — community contributions
  expected here.
- [ ] **Decompress workers stream their range** instead of downloading
  the whole compressed blob (`krunch/job.py` currently downloads the
  full blob then seeks). Fine for v1 small-N workloads; matters at
  v2 scale.
- [ ] **Custom AMI with pre-pulled image** to cut the ~13 min AWS
  Batch cold-start. Quality-of-life, not blocking.
- [ ] **Generic-orchestrator real-work timing.** `KRUNCH_WORK_TIME`
  log lines work everywhere; the parser in `tests/integration/batch.sh`
  is AWS-Batch-specific. Add a `KRUNCH_WORK_TIME_S3_URL` env var so
  workers POST timing JSON to S3 and any orchestrator can produce the
  same scaling-validation report.
- [ ] **Bigger-model option.** `RWKV-4-Pile-1B5` would lift ratio to
  ~0.07–0.09 on chat (matches ts_zip) at 3–5× slower forward. Behind
  a `KRUNCH_MODEL` env var + a separate published image.

## v2 (separate planning, not started)

v2 is the hosted offering and per-tenant adapters. **Not v1 scope** —
listed here only so the v1 design choices are legible.

- Hosted API (long-running service, batch-or-stream).
- LoRA adapters fine-tuned per customer corpus, swappable via
  `adapter_id` in the blob header.
- Multi-region / HA. v1 is explicitly single-region.
- Per-tenant streaming for inputs > GPU VRAM.

## What's *not* on the roadmap

- General-purpose binary compression. Krunch is text-only; if your
  data isn't text, use zstd/bzip3.
- Always-on local daemon mode. Compression is a one-shot transform;
  a warm local service makes no sense at this layer.
- Custom AC coder rewrite. constriction is good enough; the bits
  per token are model-bound, not coder-bound.

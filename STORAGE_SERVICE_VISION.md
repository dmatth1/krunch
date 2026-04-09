# Storage service vision (exploratory)

**Status:** exploratory back-burner. Not the current focus. Written
up during Phase 4b to capture the thinking while it's fresh so a
future session can pick it up without re-deriving everything.

**Scope note (added post Phase 4):** This document describes a
*different product* that could be built on top of the runtime —
a managed storage service whose unit economics depend on GPU
batch inference at ≥100 MB/s per GPU. The l3tc-prod runtime
itself is *not* on a roadmap toward that target. After Phase 4
we have strong evidence (Bellard's NNCP at 3.25 KB/s, CMIX at
1.66 KB/s, and our own Phase 4c/4e ceiling at ~150 KB/s) that
single-stream CPU neural compression at this ratio band is
**physically capped well below the MB/s range**, so any
"runtime + service" pivot would be a separate engineering
project, not a continuation of the existing phase plan. Treat
this doc as reference material for a possible future product,
not as a load-bearing roadmap item for the open source CLI.

## The idea in one paragraph

A cloud storage service that uses neural compression on the
backend. Customer sends data to us; we use it (plus additional
samples from them upfront) to train a small custom model on their
distribution; we compress with that model and write the compressed
bytes to a cold backing tier (S3 Glacier Deep Archive or similar);
when the customer needs data back we decompress on our side and
serve it. Retrieval is slow (hours), storage is cheap. We charge
per GB-month stored, at a number meaningfully below what the
customer would pay to put their own (possibly zstd-compressed)
data in GDA directly.

## Why the service model dodges the five product barriers

In [CLAUDE.md](CLAUDE.md) and earlier discussion we listed five
barriers that keep neural text compression stuck as research. The
service architecture turns every one of them from a deployment
problem into an ops problem:

1. **Model distribution:** the model lives on our servers
   forever. Customer never sees it, downloads it, or worries about
   version compatibility. The only "model distribution" problem
   left is internal — a model registry keyed by
   `(customer_id, model_version_id)` and a per-file header that
   records which version was used for compression. Database
   problem, not deployment problem.
2. **Cross-platform float determinism:** everything runs on
   hardware we provision. We pin CPU family, compile flags, libm,
   CUDA version, kernel. One of the hardest product problems
   becomes Docker hygiene. Multi-region replication just means
   keeping the inference image identical across regions.
3. **OOD behavior:** actually *improved* by the service model.
   Because we have the customer's data upfront, we can train a
   model per customer per data type. Their logs get one model,
   their emails get another. Ratios on homogeneous corpora are
   substantially better than on general English — this is the
   lever that turns "marginal ratio win" into "real cost savings".
4. **Speed / throughput:** compression is batch background work,
   nobody waits. Retrieval latency is the user-facing metric, and
   "hours" is an acceptable SLA for an explicit cold-archive tier
   (see Glacier Deep Archive). On compression we can use GPUs /
   NPUs without the "needs an NPU to be fast" shipping problem,
   because customers never touch our hardware.
5. **Forward compatibility:** every file header records which
   model version was used; we keep every version alive in a model
   store forever; decompression looks up the right version and
   spins up an inference worker with that checkpoint. Annoying but
   tractable. It's ops, not binary-compat.

All five barriers collapse to operational problems we control. The
hard questions shift from "can we ship this" to "does the
economics actually work".

## What AWS Glacier Deep Archive actually does

Worth being precise about this because the whole pitch hinges on
it.

- **S3 (every tier, including Glacier DA) is object-for-object
  storage.** You PUT X bytes, you GET X bytes, byte-identical. No
  server-side transform. AWS documentation is consistent across
  tiers: "compress before upload if you want compression".
- **There is no public documentation or third-party reverse
  engineering** suggesting AWS does invisible compression at the
  storage layer that saves bytes on your behalf. If they did,
  their billing would reflect it, and it doesn't.
- **Glacier Deep Archive's cost advantage is hardware, not
  software.** It's believed to be tape libraries (LTO-class) plus
  cold HDD, with a hot cache for retrieval staging. Tape is
  ~5-10× cheaper per raw TB than nearline HDD; HDDs are
  ~5-10× cheaper than SSDs. Stack those and $0.001/GB/month
  makes sense from hardware economics alone.
- **Erasure coding for durability** adds ~30-40% physical
  storage overhead. Physics, not compression.
- **LTO hardware compression** exists (enabled by default on LTO
  drives, usually gets ~2-2.5× on mixed data) but AWS does not
  pass those savings through to customer pricing — you still pay
  based on your object size as submitted.

**The takeaway:** AWS's cost structure does not include
sophisticated compression. They compete on hardware scale. That
leaves real headroom for a service that *does* compress
aggressively.

## The real competition: `zstd -19 + Glacier DA`

Not AWS's (non-existent) internal compression. The real
competition is the sophisticated customer who already runs
classical compression before upload:

| approach | 1 TB input | backend | monthly cost |
|---|---:|---:|---:|
| Raw upload → GDA | 1 TB | $1.00/mo | $1.00 |
| `zstd -19` → GDA | ~250 GB | $0.25/mo | **$0.25** |
| Your service (custom 200K model → GDA) | ~150 GB | $0.15/mo | $0.15 **+ compute cost** |

The naive-customer baseline ($1.00/TB/month) is a weak market;
any compression destroys it. The sophisticated-customer baseline
($0.25/TB/month) is the real competitor, and it's 4× cheaper than
naive.

## The binding constraint: compute cost

This was the counterintuitive finding. Against `zstd -19 + GDA`,
the compression compute itself is the dominant cost line, not the
storage savings.

Working the math for 1 PB of log-like homogeneous text:

**At current l3tc-rust CPU throughput (119 KB/s per core):**
- 10¹⁵ bytes / 122 KB/s ≈ 8.2 × 10⁹ core-seconds ≈ **260 core-years**
- At $0.01/core-hour spot: **$22,776 one-time to compress 1 PB**
- Amortized over retention:
  - 5-year: $380/month
  - 10-year: $190/month
  - 30-year: $63/month

**All-in monthly cost to serve a 1 PB customer (5-year retention):**

| line item | CPU path |
|---|---:|
| Backend storage (150 TB × GDA) | $150 |
| Compression amortized | $380 |
| Retrieval compute amortized (~1%/yr) | $25 |
| Platform overhead | $100 |
| **Total** | **~$655** |

vs customer's self-compressed baseline: **$220/month**. **3× more
expensive.** At 30-year retention it's still $338 vs $220.

**The compute cost at our current CPU speed does not pencil.** Not
even close. The ratio advantage is real but smaller than the
compute overhead.

## What flips the sign

Compute cost scales linearly with throughput. The service becomes
viable at roughly 10-100 MB/s effective per compression worker:

**At 10 MB/s per GPU (cautious — batched PyTorch forward pass):**
- 1 PB / 10 MB/s = 3.2 GPU-years
- At $0.50/GPU-hour spot (A10G / L4): **~$14,000 one-time**
- Amortized over 5 years: **$234/month**
- Total monthly: 150 + 234 + 100 ≈ **$484/month**
- Still ~2× the zstd baseline. Not there yet.

**At 100 MB/s per GPU (aggressive — hand-tuned batched kernels,
real work):**
- 1 PB / 100 MB/s = 0.32 GPU-years
- At $0.50/GPU-hour: **~$1,400 one-time**
- Amortized over 5 years: **$23/month**
- Total monthly: 150 + 23 + 100 ≈ **$273/month**
- At rough parity with zstd-self, then the ratio delta is
  pure margin / customer savings.

**The single most important engineering investment** to turn this
from research into a business is GPU batch inference at ≥100 MB/s
per GPU. Without it, the math doesn't work against a sophisticated
customer. With it, the math works and the ratio story becomes
actual customer savings.

## Target customer profile

A real customer profile for the service looks like:

- **Petabyte-scale** text-or-text-like data (tens of TB minimum;
  sweet spot at low PB)
- **Long retention** (5-30 years) so compression compute
  amortizes to near-zero
- **Rare retrieval** (<5% of data per year) so retrieval compute
  stays a line item not a cost center
- **Hours-acceptable retrieval SLA** (anything with a
  "compliance search" or "legal discovery" workflow pattern)
- **Homogeneous data** — the customer's data within a bucket is
  one format/schema/domain, so a custom-trained 200K model can
  fit it tightly. Enormous impact on the ratio side.

Best candidate verticals:
- **Regulatory / compliance archives** (finance, healthcare, law)
  — 7-30 year retention, rarely read, text-heavy, high willingness
  to pay for any cost reduction
- **Scientific / genomic / sequencing data**
- **Long-term log archival** for SaaS platforms (separate from
  the "hot log archive with search" pitch, which is a different
  business; here we're just the cheap backing storage)
- **Enterprise document archives** (email, contracts, tickets)
  that sit in cold storage for auditability

Worst-fit customers: anyone with heterogeneous mixed-format data,
anyone who already has clever domain-specific codecs (CRAM for
genomics, specialized columnar formats for time series), anyone
with interactive retrieval requirements.

## What would need to be true

For this to be a real business, all of these have to hold:

1. **Custom-trained 200K model beats `zstd -19` by ≥30% ratio on
   real customer-like data.** Untested. The ratio numbers we've
   collected are on enwik8/enwik6 (Wikipedia). Real logs are a
   mess of JSON, tracebacks, timestamps, hashes, UUIDs.
   Incompressible high-entropy tokens (hashes, UUIDs) don't
   benefit from the LM. This is the most important open
   empirical question.
2. **GPU batch inference runs at ≥100 MB/s** per reasonable GPU
   ($0.50/hr class). Achievable in principle but real work;
   probably 1-2 months of focused engineering.
3. **Target customers tolerate hours-latency retrieval** for the
   cost savings. GDA proves some do. Segment size is real.
4. **Per-customer model training amortizes**. ~$50-500 per model
   per training run is fine; $10K+ is not. Depends on corpus
   size and training config.
5. **Compute cost scales sub-linearly at batch size.** A GPU
   processing 256 streams in parallel should be much cheaper per
   GB than a GPU processing one stream at a time. This is
   standard batched-inference stuff but has to be measured.
6. **Customer-specific data handling is compliant** day one
   (SOC2, GDPR, HIPAA as table stakes for the target verticals).

## What to validate before building

Three experiments, in order, each a gate for the next:

1. **Ratio on real homogeneous text.** Take 10-100 GB of real
   log data (from a friendly SRE friend or a public corpus of
   logs — Loghub, HDFS logs, OpenStack logs on Zenodo). Train a
   fresh L3TC-200K on 80% of it. Measure the actual coded ratio
   on the other 20% vs `zstd -19` on the same 20%. Target: ≥30%
   win. **1-2 days with existing codebase + a Python training
   loop.**

2. **GPU batch inference throughput.** Port the RWKV-v4 forward
   pass to a batched PyTorch implementation and measure MB/s on
   an A10G or L4. Doesn't need to be correct to the last bit —
   we just want a throughput number. **1-2 weeks.**

3. **Unit economics model with those two numbers.** Plug
   measured ratio and measured GPU throughput into the cost
   model above. Determine the break-even retention period and
   the customer price point that clears your margin. If the
   numbers don't hold, stop; if they do, start thinking about
   platform and GTM. **1 day.**

Do not skip straight to platform work. All three are cheap
compared to the cost of building a platform before knowing
whether the technology actually supports the economics.

## Why this is not the current focus

- **Phase 4 is nearly done on ratio.** We're at 0.1699 on enwik6
  actual coded bytes, 0.61 pp from the entropy bound — 86% of
  the achievable gap closed. More ratio work is diminishing
  returns.
- **Phase 4b remaining polish** (classical fallback, enwik8
  confirmation, findings doc) is cheap and worth shipping.
- **Phase 5+ is retraining on broader / custom corpora**, which
  is ALSO what the service vision needs. The two paths merge.
- **GPU batch inference is the pivotal engineering item** for
  the service vision but also for any "push past Python on
  ratio" work in Phase 5+. Worth doing for its own sake
  eventually.

So the service vision is parked as "if we decide to pivot to a
product later, here's the blueprint". Everything we build for
Phase 4-5 also advances the service case. No wasted work either
direction.

## Open questions

- Does any cloud backend go below Glacier Deep Archive on raw
  $/GB/month? (Probably not. GDA is the floor.)
- What's AWS's likely reaction if this works? (They can bolt
  neural compression onto Glacier as a feature whenever they
  want. Moat has to be customer-specific models and ingest
  integration, not the underlying tech.)
- What's the right pricing model — per-GB-month stored, or
  per-GB-input with a retention commitment?
- S3-protocol-compatible gateway or a custom client SDK? First
  is much more valuable but harder to build.
- Where does onboarding data live during model training? How
  long do we hold it? Privacy implications.
- Can we offer a trial-onboarding that takes 1% of customer
  data, trains a model, reports the measured ratio, and quotes a
  price? That's the cleanest sales pitch but requires real
  onboarding infrastructure.

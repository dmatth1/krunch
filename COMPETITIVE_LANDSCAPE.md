# Competitive landscape — who else sells "cheaper compressed storage"

Research conducted 2026-04-21 via live web search of each
vendor's public pricing and marketing pages. Every specific claim
below (ratio, price, feature) is cited to the vendor's own page or
to a dated press release. Anything I'm inferring or extrapolating
is flagged with *(inferred)*.

## Headline finding

There are **two live commercial competitors** whose positioning
meaningfully overlaps with Krunch's "AI-powered
cheaper-than-zstd storage" thesis:

1. **[Granica Crunch](https://granica.ai/crunch/)** — Parquet /
   lakehouse compression. Claims ML-powered compression optimizer.
2. **[YScope CLP Cloud](https://yscope.com/clp-cloud)** — managed
   log archive based on the open-source CLP algorithm. Classical
   template-extraction, not ML, but positions on "smart
   compression."

Everything else in the market either (a) doesn't claim to do better
than generic compression, (b) bundles compression inside a larger
observability/archive product where customers buy for other reasons,
or (c) is a research project without a managed offering.

## Direct competitors: AI-powered managed compression

### Granica Crunch — the one to watch

**Source: [granica.ai/crunch](https://granica.ai/crunch/),
[docs.granica.ai](https://docs.granica.ai/granica-crunch-overview/)**

- **Positioning**: "Cloud cost optimization purpose-built for
  lakehouse data." Targets Parquet / Iceberg / Delta files on S3 and
  GCS.
- **Tech claim**: "ML-powered algorithms which run transparently in
  the background learning from and adapting to your data to build
  compression recipes." That's their explicit claim — we should
  take it at face value but note we haven't audited what "ML" means
  here (could be learned parameter search over classical codecs vs
  an actual learned coder; their docs don't distinguish).
- **Numbers they publish**: 15-60% reduction over the customer's
  existing compression, 43% average on TPC-DS benchmark (Snappy
  Parquet as input). On 10 PB Parquet → ~$1.2M/year gross S3
  savings claimed. On 25 PB, ~$11M over 3 years.
- **Narrow scope**: Parquet columnar data only. They don't compete
  on general JSON streams, log files, document archives, or
  free-text data — that's Krunch's lane.
- **Customer profile**: enterprise lakehouse operators
  (Snowflake/Databricks customers, big-data analytics teams).
  Different buyer from SaaS log retention / compliance archive.

**Verdict**: closest competitor by positioning. Focused on
**structured analytics data**; we cover **everything else**
(text-heavy archives, logs, documents, compliance records). The
markets don't directly overlap but both can claim "AI compression
cheaper than S3+zstd." Different verticals, different buyers.

### YScope CLP Cloud — the log-archive specialist

**Source: [yscope.com/clp-cloud](https://yscope.com/clp-cloud),
[github.com/y-scope/clp](https://github.com/y-scope/clp)**

- **Positioning**: "Compressed logs. Searchable. Forever." Managed
  service wrapping their open-source CLP tech.
- **Tech**: CLP extracts log templates + variables into a columnar
  format, compresses with zstd per column. **Not ML** —
  template-structure-aware classical. Reported ~2× better than
  zstd on real logs, searchable without decompression.
- **Pricing**: per compressed-GB stored + per-second compute. Exact
  rates not on the pricing page; enterprise contact.
- **Scope**: log data specifically. Variable-content archives
  (documents, JSON with free text, medical records) are out of
  scope for CLP's template extraction.

**Verdict**: direct overlap on **log archive** use cases. We plan
to dispatch *to* CLP for templated-log chunks per our hybrid codec
design, which makes YScope's tech an input to our product rather
than a pure competitor. But for customers deciding "managed log
archive with smart compression," YScope is the incumbent choice
today.

## Adjacent competitors: compressed log/observability stores

All use generic classical compression (zstd/gzip/snappy) and sell
on query + UX + integrations rather than compression. Customers buy
for search + dashboards, not for $/GB-at-rest. Krunch is *not*
trying to be this category.

| vendor | claimed compression | unit price | customers buy for |
|---|---|---|---|
| **[Datadog Flex Logs](https://docs.datadoghq.com/logs/log_configuration/flex_logs/)** | zstd, columnar; no ML claim | [$0.05 / million events-month](https://www.datadoghq.com/pricing/) (3/6/15-mo retention tiers) | query + dashboards |
| **[Datadog Flex Frozen](https://www.datadoghq.com/about/latest-news/press-releases/datadog-expands-log-management-offering-with-new-long-term-retention-search-and-data-residency-capabilities/)** | (same stack, longer tier) | 7-year retention, fully searchable, priced lower than Flex | compliance / audit retention |
| **Elastic Frozen Tier** | Lucene codecs, zstd since 8.12+ | searchable snapshots on S3 | enterprise SIEM / search |
| **Grafana Loki / Grafana Cloud** | gzip/snappy/zstd configurable | index-free architecture | generous free tier + OSS |
| **[Axiom](https://axiom.co/pricing)** | "customers typically enjoy a 95% reduction in storage footprint" (their page; vs raw, not vs zstd) | $0.12/GB ingest → $0.09/GB at high volume; free tier: 1000 GB/mo | $/GB for small-midmarket |
| **Humio / CrowdStrike Falcon LogScale** | proprietary + classical | bundled with EDR | security ops, bundling |
| **Splunk SmartStore / DDSS** | gzip/zstd in bucket format | enterprise SIEM market | existing Splunk spend |
| **[Hydrolix](https://hydrolix.io/)** | "up to 50× compression"; "observed 52:1" in their benchmarks | private pricing; pitches "4× more data, 4× lower cost" vs Splunk | TCO vs Splunk at scale |
| **Sumo Logic Archive** | gzip to Glacier | retention tier + compliance | Splunk competitor |

None of these vendors explicitly claim *AI-powered* compression.
They claim "compression + smart architecture." Hydrolix's 50×
number is the loudest ratio claim in the category but applies to
their specific columnar-telemetry workload, not general data.

## AWS-native baselines the customer might use instead

The gravity point. If a customer can get "good enough" from AWS
primitives, Krunch has to beat that, not just beat zstd on a
laptop.

Prices verified via [AWS S3 pricing](https://aws.amazon.com/s3/pricing/)
and [cloudzero.com/blog/s3-pricing](https://www.cloudzero.com/blog/s3-pricing/)
2026 guide.

| option | price | compression story |
|---|---|---|
| S3 Standard + client-side zstd | $0.023/GB/month | customer's own zstd, typically 10-15× ratio on text |
| S3 Intelligent-Tiering | auto $0.023 → $0.0125 → $0.004 by access pattern | **no compression — customer's job** |
| S3 Glacier Instant Retrieval | $0.004/GB/month | no compression |
| **S3 Glacier Deep Archive** | **$0.00099/GB/month** ($1/TB/month), 180-day min, $0.025 per 1K bulk retrieves | no compression, 12–48 h retrieval |
| CloudWatch Logs → S3 export | $0.50/GB ingest + $0.03/GB stored | internal compression ~5-10× |
| Athena on Parquet + zstd | S3 storage + query | Parquet columnar + zstd |

**The thing that kills pure-cold-storage compression pitches: S3
Deep Archive at $1/TB/month is already cheaper than any warm
storage no matter how well we compress.** Our pitch has to target
*warm, queryable, text-heavy* data where Deep Archive's retrieval
latency (hours) and 180-day minimum are disqualifying.

## Enterprise-archive incumbents (compliance verticals)

Missing from most compression-competitor analyses but genuinely the
dollars in our target verticals. Legal, financial, and healthcare
archive incumbents:

- **[Iron Mountain Digital](https://www.ironmountain.com/services/iron-cloud-data-management/cloud-storage-and-migration)** — legal / financial archive, pay-as-you-go cloud archive, compliance certifications.
- **[Smarsh](https://www.capterra.com/p/130954/Email-Archiving/)** — regulated industry communications archive (finance, healthcare); Capterra lists starting price "$5/month" but true enterprise pricing is quote-only.
- **[Global Relay](https://www.capterra.com/p/85993/Global-Relay-Archive/)** — finance communications archive, FINRA-compliant; quote-only.
- **Proofpoint Archive** / **Mimecast Archive** — email archive with eDiscovery / compliance.
- **Veritas Enterprise Vault** — on-prem / hybrid archive.

**Pricing reality**: none of these publish per-GB list prices on
their public sites — all require a quote. Capterra and third-party
review sites note that export / migration-out fees are often billed
per GB and are the visible pain point in these contracts
*(inferred from vendor-review summaries, not from vendor price
sheets)*. The retail-list anchor of "$5–50/GB/year" in earlier
drafts is a commonly-cited industry estimate but **I have not been
able to verify a specific 2026 rate from any of these vendors; real
pricing is private**.

**What this means for Krunch**: the compliance-archive
incumbents are almost certainly charging 10–100× S3+zstd DIY, but
we can't quantify the delta until we get actual quotes. Regardless
of the exact multiplier, on any "$N/GB-stored-per-year" contract a
30% compression improvement is $0.3 × N / GB-year of real money per
customer — on a Deep-Archive $0.01/GB/year workload the same
improvement is $0.003 and irrelevant.

**Action item**: get sample quotes from Iron Mountain, Smarsh, and
Global Relay before claiming specific $-savings numbers in the
sales pitch.

## Research-stage / not-yet-productized

Not competitors today but worth watching:

- **ts_zip / NNCP** (Fabrice Bellard). Research compressors, ratio
  records, not commercialized. ~100×–1000× slower than zstd on CPU,
  which is why no commercial offering has emerged.
- **DeepMind "Language Modeling Is Compression"** (ICLR 2024). Paper
  validating the thesis; no product.
- **[LMCompress](https://www.nature.com/articles/s42256-025-01033-7)**
  (2025, Nature MI). "Shatters all previous lossless compression
  records on four media types: text, images, video and audio."
  Academic; not productized as a service.
- **[IBM ZipNN](https://research.ibm.com/blog/Zip-NN-AI-compression)**
  (IEEE Cloud 2025). Compresses LLM model weights specifically
  (~33% savings, up to 150% faster transfers). **Different problem
  — compressing the model, not compressing customer data at rest**.
  Not a Krunch competitor.
- **[Google TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)**
  (ICLR 2026). KV-cache quantization for LLM inference (6× memory
  reduction). Also different problem; not a Krunch competitor.
- **DeepZip** (Stanford 2019). Research; no commercial follow-on.
- **Dropbox Lepton** (2016-2019). Internal JPEG recompression.
  ~25% image-storage savings at Dropbox. **Kept internal** — the
  value was in Dropbox's storage margin, not in selling the tech.
  Most relevant precedent to Krunch actually being deployed at
  scale, and the lesson is that learned compression is real but has
  historically been captured as internal margin, not sold as a
  separate product.

### Adjacent YC companies (not direct competitors)

Verified via YC Launch pages:

- **[Terark (YC W17)](https://www.ycombinator.com/companies/terark)** —
  embedded storage-engine compression (10:1 ratios, search on
  compressed data). Ships as a database engine for MySQL/MongoDB/
  RocksDB, not a managed storage service. Alibaba Cloud partner.
  Closest YC-backed compression company but not in our lane.
- **[Compresr (YC)](https://www.ycombinator.com/companies/compresr)** —
  LLM context-window compression (100× context compression).
  Compresses prompts, not files at rest.
- **[The Token Company (YC, 2025)](https://www.ycombinator.com/companies/the-token-company)** —
  compression middleware for LLM outputs. Prompt-layer, not
  storage-layer.
- **[Archil (YC)](https://www.ycombinator.com/companies/archil)** —
  "S3-as-a-filesystem", 30× faster than raw S3, 90% cheaper than
  EBS. Different product surface; doesn't claim compression.

**Note on YC 2024/2025 search results**: web search surfaced no
YC-backed company in W24–W25 positioned on "AI-powered
cheaper-than-zstd object storage." The token/prompt compression
companies are using the word "compression" in a different sense.

## Open-source self-host alternatives

- **CLP (Uber, y-scope/clp)**: log-specific, Apache-2.0, strong
  reference for our CLP codec path.
- **Hydrolix**: closed source.
- **ClickHouse** + specialized column codecs: OSS columnar DB that
  observability stacks (SigNoz, OpenObserve) build on. Strong
  compression on structured data.
- **Quickwit** (acquired by Datadog 2024): OSS log search on
  object storage; classical compression.
- **Parquet + zstd + Athena** DIY reference architecture.

## Where AI-powered compression is or isn't a moat

Honest take after the landscape:

### What's not a moat
- **"We use AI"** as marketing — Granica, IBM, Pure Storage all
  already do this. Ubiquitous claim by 2027.
- **Pure $/GB competition with Deep Archive** — impossible; they
  price below any compression can recover.
- **Compression ratio as the only differentiator** — Hydrolix has
  shown customers will switch for cheaper storage, but they sell on
  *TCO vs. Splunk*, not on ratio; customers need more than a
  number.

### What could be
- **Throughput parity with zstd on warm reads** via per-customer
  distilled small models. If we match S3+zstd retrieval latency
  within 2-3× on a GPU-backed decoder, we unlock warm-tier
  use-cases Deep Archive can't serve.
- **Verifiable ratio guarantee** — "we guarantee ≥25% smaller than
  zstd-22 on your data or you pay zstd-22-equivalent rates."
  Converts compression from marketing into a contract. Nobody else
  in this space is doing it because nobody else can predict ratio
  for new data; if we can (via our detector + dispatcher's
  per-chunk measurement), we can underwrite the promise.
- **Compliance-bundled archive** (HIPAA, SOC 2, legal hold,
  chain-of-custody) targeting the Iron-Mountain-class incumbents
  where pricing is 10–100× S3+zstd and compression savings become
  real money.
- **Searchable compressed archive** — via LM embeddings that allow
  approximate semantic search over compressed representations.
  Beats CLP on free text; beats Elastic on cost by not maintaining
  inverted indexes.
- **Specialization in text-heavy verticals** — Granica owns
  Parquet/lakehouse. YScope owns logs-with-templates. The unclaimed
  wedge is **text-heavy non-structured-analytics data**: documents,
  free-text logs, medical records, legal correspondence,
  compliance records, chat archives. That's where Krunch's
  neural-primary dispatcher has the biggest advantage and no direct
  competitor today.

### The right framing

Don't claim "we beat zstd" on its own — that's true but not a
business. Claim instead: **"on your specific data, we train a
compression model and we'll commit in writing to a ratio better
than zstd-22 with an SLA; if we miss, you get zstd-22 pricing. You
trade a managed service for verifiable savings."**

That reframes the competition:
- vs Granica: different data shape (text vs Parquet), different
  vertical.
- vs YScope: broader content type (not log-only).
- vs Deep Archive: different tier (warm + retrievable, not cold
  archive).
- vs Iron Mountain / Smarsh: same tier (compliance archive), but
  with a measurable savings story and probably better pricing.
- vs Axiom / Hydrolix / Datadog: adjacent but not competing —
  they're bundled observability platforms, we're pure storage with
  a compression contract.

## What we should verify (open action items)

1. **Real Granica pricing** — the blog numbers ($1.2M savings on 10
   PB) are aggregate, not per-GB. Get a quote, understand their
   pricing surface.
2. **Granica's "ML" implementation** — is it a learned coder or a
   learned parameter search? Talk to their sales eng, read docs
   carefully.
3. **YScope CLP Cloud actual pricing tiers** — not on their public
   pricing page; get a quote.
4. **Iron Mountain / Smarsh / Global Relay / Proofpoint current
   list prices** — the compliance-archive incumbent pricing that
   actually defines our revenue ceiling. Request sample quotes.
   *(Confirmed via public review sites that none publish list
   pricing; quote-only is the norm in the category.)*
5. **Datadog Flex Frozen unit economics** — the press release is
   dated 2025, and pricing at that tier isn't on the public
   calculator. Ask via sales to see where they land for 7-year
   compliance retention.
6. **Hydrolix real TCO** — their "4× cost reduction" is vs Splunk,
   not vs S3+zstd. Would want to see the denominator.

## Sources

Primary competitor pages (verified 2026-04-21):

- [Granica Crunch product page](https://granica.ai/crunch/)
- [Granica Crunch docs overview](https://docs.granica.ai/granica-crunch-overview/)
- [Granica savings analysis docs](https://docs.granica.ai/savings-analysis-crunch-s3-gcs/)
- [YScope CLP Cloud](https://yscope.com/clp-cloud)
- [YScope pricing page](https://www.yscope.com/pricing/)
- [y-scope/clp on GitHub](https://github.com/y-scope/clp)
- [Uber CLP blog post](https://www.uber.com/blog/reducing-logging-cost-by-two-orders-of-magnitude-using-clp/)

Observability / log-archive vendor pricing pages (verified
2026-04-21):

- [Datadog Flex Logs docs](https://docs.datadoghq.com/logs/log_configuration/flex_logs/)
- [Datadog pricing](https://www.datadoghq.com/pricing/)
- [Datadog Flex Frozen press release (2025)](https://www.datadoghq.com/about/latest-news/press-releases/datadog-expands-log-management-offering-with-new-long-term-retention-search-and-data-residency-capabilities/)
- [Axiom pricing](https://axiom.co/pricing)
- [Axiom new-pricing blog](https://axiom.co/blog/new-pricing-axiom-starts-lower-stays-lower)
- [Hydrolix home](https://hydrolix.io/)
- [Hydrolix pricing](https://hydrolix.io/pricing/)

AWS pricing references (verified 2026-04-21 guides):

- [S3 pricing — AWS](https://aws.amazon.com/s3/pricing/)
- [The Ultimate Guide to AWS S3 Pricing 2026 — cloudchipr](https://cloudchipr.com/blog/amazon-s3-pricing-explained)
- [A 2026 Guide To Amazon S3 Pricing — cloudzero](https://www.cloudzero.com/blog/s3-pricing/)

Compliance-archive incumbents (public-review-site summaries;
price lists not published):

- [Iron Mountain cloud storage](https://www.ironmountain.com/services/iron-cloud-data-management/cloud-storage-and-migration)
- [Smarsh on Capterra](https://www.capterra.com/p/130954/Email-Archiving/)
- [Global Relay on Capterra](https://www.capterra.com/p/85993/Global-Relay-Archive/)

Research / adjacent:

- [Dropbox Lepton](https://github.com/dropbox/lepton)
- [ts_zip — Bellard](https://bellard.org/ts_zip/)
- [NNCP — Bellard](https://www.bellard.org/nncp/)
- [Language Modeling Is Compression (DeepMind, ICLR 2024)](https://arxiv.org/abs/2309.10668)
- [LMCompress — Nature MI 2025](https://www.nature.com/articles/s42256-025-01033-7)
- [IBM ZipNN blog](https://research.ibm.com/blog/Zip-NN-AI-compression)
- [Google TurboQuant blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

YC companies checked:

- [Terark (YC W17)](https://www.ycombinator.com/companies/terark)
- [Compresr (YC)](https://www.ycombinator.com/companies/compresr)
- [The Token Company (YC, 2025)](https://www.ycombinator.com/companies/the-token-company)
- [Archil (YC)](https://www.ycombinator.com/companies/archil)

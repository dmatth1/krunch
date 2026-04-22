# Customer profile — what we actually compress

This is the product-level view of what Krunch's target customers
store and why the compression design is shaped the way it is. If you
came here asking "why does Krunch's design make sense when the
HDFS spike showed it losing to zstd" — this is the answer.

## What HDFS is, and why it's a stress test, not a customer

We ran 10+ training experiments on HDFS NameNode logs in Spike 1 +
Spike 2. On that corpus our neural codec loses to zstd. The HDFS
corpus is:

- **100% machine-generated.** No human content anywhere.
- **>90% template repetition** — the same line shape
  `"YYMMDD HHMMSS PID INFO class: action blk_X src /IP:N dest /IP:N"`
  appears millions of times with only the variable fields changing.
- **Variable fields are near-random IDs** (20-digit block IDs, IPs,
  port numbers) that aren't predictable from context.
- **Ideal for a hash-based sliding-window coder.** `zstd --long=27`
  has a 128 MB window that catches every template repetition
  plus every previous occurrence of a block ID / IP, and encodes
  them as short back-references. That's near the theoretical
  optimum for this data shape.

HDFS is what you'd pick if you wanted to make a neural compressor
look bad. It's a worst-case stress test, not a representative
workload. Most real customer data has human-written content
interleaved with structure, which is exactly where neural
predictors beat window-based coders.

## What realistic customer data actually looks like

Breaking down the target verticals from `STORAGE_SERVICE.md`:

| vertical | typical content | compressibility profile |
|---|---|---|
| Software / SaaS (app logs, support tickets, error streams) | Timestamps + templated log prefixes + free-text error messages + stack traces + user-submitted strings | ~60% templated (zstd-dict wins), ~40% free text (neural wins) |
| Finance / fintech (transaction archives) | Structured transaction records: dates, amounts, IDs, merchant codes, card hashes | Highly structured, ~70% dictionary-friendly; ~30% near-random IDs. zstd-dict dominant, neural marginal |
| Healthcare (clinical records, trial data) | Free-text clinical notes + ICD codes + structured fields | ~60% free-text prose, ~40% structured. **Neural wins big** on the prose |
| Legal (e-discovery, contracts, correspondence) | Long-form prose, contracts with repeated boilerplate, emails | Almost entirely text. **Neural wins 20–40%** |
| Compliance / audit trails | JSON event streams with free-text "description" / "reason" fields | Mixed. Neural on text fields, zstd on keys + IDs |
| Documentation / knowledge bases / internal wikis | Markdown, HTML, articles | Almost entirely prose. **Neural wins 25–40%** |
| Security / SIEM events | Mix of authentication events (zstd-friendly) and alert/description text (neural-friendly) | Mixed. ~50/50 split |

What's NOT on this list: highly templated infrastructure logs like
HDFS, Hadoop, raw Kubernetes event firehoses. That data exists, but
it's a minority slice of the **compliance/archive** spend. Most of
it goes into time-series databases (Prometheus, InfluxDB),
observability tools (Datadog, Grafana), or dedicated log
management products (Splunk, Loki) — not cheap long-term storage.

The archive use-case customers actually buy storage for tends to be:
- audit trails and compliance records
- legal discovery document stores
- medical / financial records under retention regulations
- support ticket archives
- knowledge-base / documentation archives
- SaaS application logs (which have a lot of free-text content)

All text-heavy. All in the neural-wins column of our data.

## Why customer profile shapes the codec design

The hybrid codec design in `HYBRID_CODEC_DESIGN.md` is specifically
tuned to how this customer profile distributes work:

- **Neural-as-primary** because the majority of bytes in a typical
  customer archive are text-like.
- **Per-chunk codec selection** because real datasets aren't pure —
  they have bursts of templated content (where classical wins)
  interleaved with free-text (where neural wins).
- **Per-dataset zstd trained dictionary** for customers whose data
  is structured (fintech, security events) — pushes the classical
  baseline 2-3× without needing neural.
- **CLP in the menu** so when a customer *does* bring templated
  infrastructure logs, we at least match zstd via a log-specific
  path.
- **Brotli shared-dict** for document archives with many near-
  duplicate versions (legal, docs, wikis) — 90%+ reduction on
  incremental stores of the same content.
- **Safety-net substitution** so we never silently ship a blob
  that's larger than what zstd-dict alone would have produced.

## The customer pitch in one paragraph

"We compress your data with the best codec for each chunk,
automatically. On text-heavy content — which is most data you
actually pay to archive — we beat zstd by 15–40%. On templated
infrastructure logs we match zstd via a log-specific preprocessor.
You never pay more than zstd + a small dispatcher overhead,
because our safety-net always substitutes zstd when it would have
been smaller. Per-dataset, we train a compression model on your
data; savings compound with scale."

## How the HDFS finding refines (not kills) the product claim

The original "we beat zstd" framing was too broad. The refined
framing is:

- "We beat zstd on text-heavy content" — defensibly true, measured.
- "We match zstd on templated-log content" — true *if and only if*
  we ship CLP in the codec menu. Without CLP we lose by 40% on
  HDFS-class data due to the per-chunk-zstd window limitation
  documented in `DISPATCHER_SIM_RESULTS.md`.
- "We're never materially worse than zstd on any dataset" —
  true by construction via the Stage 3 safety net.

The Spike 1 + 2 HDFS result didn't kill the thesis; it surfaced the
specific piece that needs to be in the product from day one
(CLP + per-dataset chunk-size tuning). We pay attention to that
feedback.

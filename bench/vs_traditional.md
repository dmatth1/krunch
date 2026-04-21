# l3tc vs traditional compressors

## Release gate

PHASE_14: l3tc ratio must be BETTER than the best traditional compressor on ALL 7 specialist domains. If l3tc loses on any domain, that specialist or its tokenizer is a release blocker.

**No l3tc numbers available** — run `bench/e2e_moe.py` first so `bench/e2e_moe.json` exists.

## Per-domain head-to-head (ratio, lower is better)

| domain | l3tc | gzip-9 | brotli-11 | zstd-22 | xz-9e | best-trad | winner |
|---|---|---|---|---|---|---|---|
| prose | — | 0.3593 | 0.2908 | 0.3078 | 0.3019 | 0.2908 (brotli-11) | — |
| code | — | 0.2418 | 0.1991 | 0.2129 | 0.2061 | 0.1991 (brotli-11) | — |
| structured | — | 0.1693 | 0.1309 | 0.1413 | 0.1357 | 0.1309 (brotli-11) | — |
| logs | — | 0.0983 | 0.0791 | 0.0827 | 0.0779 | 0.0779 (xz-9e) | — |
| tabular | — | 0.3292 | 0.2287 | 0.2398 | 0.2295 | 0.2287 (brotli-11) | — |
| markup | — | 0.0822 | 0.0633 | 0.0688 | 0.0662 | 0.0633 (brotli-11) | — |
| fallback | — | 0.8834 | 0.8620 | 0.8626 | 0.8687 | 0.8620 (brotli-11) | — |

## Per-compressor per-domain detail

### brotli-11

| domain | files | in MB | ratio | compress KB/s |
|---|---:|---:|---:|---:|
| prose | 8 | 2.62 | 0.2908 | 267.3 |
| code | 8 | 1.40 | 0.1991 | 264.2 |
| structured | 8 | 1.62 | 0.1309 | 169.9 |
| logs | 8 | 1.57 | 0.0791 | 362.8 |
| tabular | 6 | 4.12 | 0.2287 | 258.9 |
| markup | 7 | 0.71 | 0.0633 | 474.6 |
| fallback | 6 | 0.35 | 0.8620 | 61.1 |
| mixed | 4 | 0.41 | 0.0481 | 540.7 |

### gzip-9

| domain | files | in MB | ratio | compress KB/s |
|---|---:|---:|---:|---:|
| prose | 8 | 2.62 | 0.3593 | 3533.6 |
| code | 8 | 1.40 | 0.2418 | 551.2 |
| structured | 8 | 1.62 | 0.1693 | 239.0 |
| logs | 8 | 1.57 | 0.0983 | 966.3 |
| tabular | 6 | 4.12 | 0.3292 | 1579.4 |
| markup | 7 | 0.71 | 0.0822 | 920.2 |
| fallback | 6 | 0.35 | 0.8834 | 93.9 |
| mixed | 4 | 0.41 | 0.0598 | 2039.7 |

### xz-9e

| domain | files | in MB | ratio | compress KB/s |
|---|---:|---:|---:|---:|
| prose | 8 | 2.62 | 0.3019 | 1427.8 |
| code | 8 | 1.40 | 0.2061 | 693.7 |
| structured | 8 | 1.62 | 0.1357 | 409.9 |
| logs | 8 | 1.57 | 0.0779 | 602.0 |
| tabular | 6 | 4.12 | 0.2295 | 684.1 |
| markup | 7 | 0.71 | 0.0662 | 578.8 |
| fallback | 6 | 0.35 | 0.8687 | 98.0 |
| mixed | 4 | 0.41 | 0.0484 | 1384.0 |

### zstd-22

| domain | files | in MB | ratio | compress KB/s |
|---|---:|---:|---:|---:|
| prose | 8 | 2.62 | 0.3078 | 1881.8 |
| code | 8 | 1.40 | 0.2129 | 501.0 |
| structured | 8 | 1.62 | 0.1413 | 425.1 |
| logs | 8 | 1.57 | 0.0827 | 838.1 |
| tabular | 6 | 4.12 | 0.2398 | 910.9 |
| markup | 7 | 0.71 | 0.0688 | 1095.1 |
| fallback | 6 | 0.35 | 0.8626 | 411.4 |
| mixed | 4 | 0.41 | 0.0517 | 1152.3 |

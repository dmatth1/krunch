# enwik8 — l3tc-rust (Phase 2.5)

Full 100 MB enwik8 corpus, single measurement, binary built with
`target-cpu=native` on Apple Silicon (aarch64). Round-trip verified
byte-identical.

| metric | value |
|---|---|
| input | 100,000,000 bytes |
| output | 21,660,095 bytes |
| **ratio** | **0.2166** |
| **compress** | **110.65 KB/s** |
| **decompress** | **117.50 KB/s** |
| round-trip | OK |

Scaling vs smaller corpora:

| corpus | size | ratio | compress KB/s |
|---|---:|---:|---:|
| enwik6-50k | 50 KB | 0.1815 | 81.3 |
| enwik6 | 1 MB | 0.2061 | 116 |
| **enwik8** | **100 MB** | **0.2166** | **110.65** |

Throughput is stable between 1 MB and 100 MB (116 → 110 KB/s, a
5% drop from I/O and OS noise over a 30-minute run). Ratio
degrades ~1 percentage point from 1 MB → 100 MB because enwik8 has
more template boilerplate and varied languages in the tail that
the 200K parameter model doesn't compress as well as the cleaner
first megabyte.

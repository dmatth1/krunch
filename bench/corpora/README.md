# Test corpora

This directory holds the test corpora used by the benchmark harness.
Corpora themselves are gitignored because they're large (enwik9 is 1 GB).

## Standard corpora

These are the canonical lossless-compression benchmarks. Download them
with `scripts/download_corpora.sh`.

| Corpus | Size | Source | Description |
|---|---|---|---|
| `enwik6` | 1 MB | first 10^6 bytes of enwik9 | Fast smoke tests |
| `enwik8` | 100 MB | [mattmahoney.net](http://mattmahoney.net/dc/enwik8.zip) | First 10^8 bytes of English Wikipedia XML dump. Hutter Prize test corpus. |
| `enwik9` | 1 GB | [mattmahoney.net](http://mattmahoney.net/dc/enwik9.zip) | First 10^9 bytes of the same dump. Main Hutter Prize corpus. |
| `silesia/*` | 202 MB | [Silesia](http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia) | Heterogeneous corpus: text, code, DNA, executables, databases, XML |
| `canterbury/*` | 2.8 MB | [Canterbury](https://corpus.canterbury.ac.nz/descriptions/) | Small varied files: text, code, technical writing, DNA, etc. |

## Custom corpora

For our own testing, we'd also want a `mixed` corpus that represents the
real-world distribution of data a general-purpose compressor sees:

- Source code (various languages)
- Log files
- JSON / YAML configuration
- HTML / Markdown documentation
- CSV data
- Binary blobs (small)

This is deferred until after classical and L3TC measurements against
the standard corpora are committed. We don't want to debug custom
corpus issues at the same time as initial L3TC setup.

## File naming

The benchmark harness treats every file in `bench/corpora/` (not
starting with `.` and not named `README.md`) as a corpus when
`--all` is passed. Put subdirectories here if you want to group
related files (e.g. `bench/corpora/silesia/mozilla`).

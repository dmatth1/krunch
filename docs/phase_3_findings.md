# Phase 3 findings

Phase 3 took l3tc-rust from a 1 MB-enwik6 toy with no integrity
check, no streaming, and no support for non-UTF-8 inputs to
something you could (almost) drop in next to zstd. The "almost"
is the OOD failure mode and two UTF-8 robustness gaps that
Silesia surfaced — addressed in Phase 4a.

## Headline numbers

| corpus | size | l3tc-rust ratio | compress KB/s | path |
|---|---:|---:|---:|---|
| enwik6 | 1 MB | **0.2061** | **116** | tokenized |
| **enwik8** | **100 MB** | **0.2166** | **110.65** | tokenized |
| Silesia / dickens | 10 MB | 1.0000 | (raw) | raw-store (bug A) |
| Silesia / webster | 41 MB | **1.2613** | 290 | tokenized — *worse than raw* |
| Silesia / nci | 33 MB | 0.4877 | 54 | tokenized |
| Silesia / reymont | 6.6 MB | (failed) | — | mid-stream UTF-8 crash |
| Silesia / xml | 5.3 MB | (failed) | — | mid-stream UTF-8 crash |
| Silesia / mozilla, mr, ooffice, osdb, samba, sao, x-ray | 6-51 MB each | 1.0000 | (raw) | raw-store |

For comparison on the in-distribution side, l3tc-rust is still
**the best-compressing tool in our bench suite** on enwik6/8 — 27%
better than bzip2-9 on enwik6, 13% better on enwik8.

## What shipped

| commit | what | impact |
|---|---|---|
| `641dd19` | **3a — File format v3 + CRC32** | bit-flip detection, magic bytes, stable framing |
| `e465d03` | **3b — Streaming encode** | bounded ~4 MB RSS regardless of input size |
| `21929bd` | **3b — Streaming decode (raw-store)** | bounded RSS on binary inputs |
| `3f5bc0d` | **Binary input support (FLAG_RAW_STORE)** | non-UTF-8 inputs round-trip instead of erroring |
| `6dd2296` | **enwik8 baseline** | first real-sized number we ship |
| `657e1a4` | **CLAUDE.md** | codifies the two project goals + regression gates |

Plus PHASE_3/4/5/6 docs, an updated README headline table, and the
new bench result files (`bench/results/enwik8-l3tc.md`).

## What we learned about model-based compression

Phase 3's most important lesson came not from any single commit
but from running the compressor on a corpus we hadn't seen.

### 1. The model is the compressor

`webster` is a 41 MB plain-ASCII English dictionary. By every
classical metric it should be highly compressible — gzip and zstd
both get it under 0.30. l3tc-rust sent it through the tokenized
path and produced **52 MB of output** — 27% larger than the
input. The arithmetic coder is exactly as good as the model's
per-token predictions, and the L3TC-200K model is trained on
enwik8 prose, so webster's idiosyncratic dictionary format
(phonetic notation, accent marks, structured headwords) is wildly
out of distribution. The model assigns low probability to almost
every token; the AC needs more bits per token than raw bytes
have; the "compressed" output ends up bigger than what we started
with.

This isn't a bug. It's the fundamental tradeoff of model-based
compression: **ratio ≈ how well the model matches the input
distribution.** Everything else is implementation detail.

The fix is structural, not local:

- **Phase 4a (next):** ship a hybrid path that picks
  `min(LM_output, zstd_output)` per file. Strict lower bound on
  ratio = best of {LM, zstd}. Worst case stops being embarrassing.
- **Phase 5 (later):** retrain on a broader corpus (The Pile or a
  bespoke domain mix) so the LM wins on more inputs in the first
  place and the fallback fires less often.

### 2. UTF-8 detection at 4 MB granularity is too coarse

The current `encode_reader` peeks the first batch (~4 MB) for
UTF-8 validity. If the first batch is valid, the whole file goes
down the text path; if not, it goes to raw-store. Two failure
modes surfaced on Silesia:

**Bug A — stray byte poisons whole file** (dickens). dickens has
8 stray non-UTF-8 bytes scattered across 10 MB of otherwise-clean
ASCII (probably scanning artifacts). The first batch contains one
of them, the UTF-8 probe fails, the entire file gets routed to
raw-store, and our advertised compression ratio on a corpus
that's 99.9999% English text is **1.0000**. 8 bytes of dirt cost
us all 8 megabytes of potential compression.

**Bug B — mid-stream UTF-8 failure crashes encode_reader**
(reymont, xml). reymont is a PDF — its first 4 MB look like
ASCII (PDF starts with `%PDF-1.3`), so the UTF-8 probe passes.
xml is well-formed XML but contains some binary payload past
byte 655925. In both cases, encode_reader takes the text path,
calls `std::str::from_utf8` mid-stream on a later batch, hits an
invalid byte, and returns an error to the CLI. The half-written
output file is left on disk in a partial state.

Both fixes belong together: **per-segment UTF-8 detection**, with
invalid-byte regions routed through the existing
`needs_raw_fallback` segment path. The plumbing already exists
for SPM-normalization fallback (Persian/Arabic ZWNJ); we just
need to extend it to invalid-UTF-8 stretches and to never crash
the encoder. Bundled into Phase 4a alongside the classical
fallback.

### 3. File format integrity is cheap and high-value

The CRC32 trailer (3a) added ~0.5 KB/s of throughput cost on
enwik6 — within measurement noise. In return:

- A single bit flip in the compressed file is detected up front
  with a clear error message, instead of crashing deep in the AC
  or silently decoding garbage.
- The format now has magic bytes, version negotiation, and a
  reader that accepts both the previous and current versions —
  the basics of a stable on-disk format.

If you only do one production-hardening commit on a compressor,
do this one.

### 4. Streaming encode pays for itself on the first big file

The pre-3b CLI loaded the full input into memory before
compressing. enwik8 (100 MB) costs 100 MB of peak RSS plus the
compressed body — fine on a workstation, embarrassing on a
container or a Raspberry Pi. The 3b streaming encoder caps RSS
at the model + tokenizer + a 4 MB carry buffer + in-flight
segment bodies, regardless of input size. The CLI's enwik8 run
peaked at ~30 MB RSS instead of ~250 MB. No ratio or speed
regression.

The trick was the `N_SEGMENTS_IMPLICIT` sentinel: writing
`u32::MAX` in the header where the segment count would normally
go, and teaching the decoder to walk segments until it sees the
trailer magic. No `Seek` requirement on the writer, so the
compressor can pipe to stdout as well as a file.

## What's deferred and why

| deferred to | item | reason |
|---|---|---|
| Phase 4a | True streaming decode for tokenized files | Compressed bodies are ~5× smaller than the output; only matters for absurdly large inputs |
| Phase 4a | Bug A (stray byte poison) + Bug B (mid-stream crash) | Bundled with the per-segment fallback work that classical-fallback also wants |
| Phase 4a | Per-file classical-fallback (`min(LM, zstd)`) | Fixes the OOD failure mode visibly |
| Phase 4b | Closing the 4 pp ratio gap to Python L3TC | The headline goal of CLAUDE.md; needs measurement before any commit |
| Phase 5 | Broader training corpus | Weeks of compute + data work; deferred until 4 ships |
| Phase 6 | Multi-platform release artifacts | Plumbing; lower per-hour value than Phase 4 ratio work |

## State of the project after Phase 3

- **Speed:** 116 KB/s compress on enwik6, 110 KB/s on enwik8.
  7.4× faster than Python L3TC-200K. Same as end of Phase 2.5 —
  Phase 3 was about correctness and surface area, not speed.
- **Ratio:** 0.2061 on enwik6, 0.2166 on enwik8. Best in the
  bench suite on Wikipedia-style text. *Worse than no
  compression* on at least one Silesia file (webster). Phase 4
  is about both fixing the latter and closing the gap to Python
  on the former.
- **Surface area:** v3 file format with magic + version + CRC,
  streaming encode + raw-store streaming decode, binary input
  via raw-store, CLAUDE.md regression gates, README with real
  enwik8 numbers, full PHASE_3/4/5/6 plans.
- **36 unit tests + 4 end-to-end integration tests passing.**
  Byte-identical round trip on every test corpus including
  enwik6 (Persian/Arabic), enwik8 (100 MB), and 7 of 12 Silesia
  files. The 5 Silesia failures (dickens UTF-8 routing, webster
  ratio>1, nci ratio 0.49, reymont/xml mid-stream crash) are
  exactly the inputs that motivate Phase 4.

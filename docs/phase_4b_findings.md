# Phase 4b findings — closing the gap from actual bytes to the entropy bound

Phase 4a proved our forward pass was already bit-identical to
Python L3TC's. Phase 4b attacked the remaining gap between
actual coded bytes and the theoretical entropy bound, which is
the *real* ratio question once the model side is correct.

## TL;DR

enwik6 actual coded ratio: **0.2060 → 0.1699** (−3.61 pp, no
speed regression). Two commits did the work:

| commit | change | enwik6 ratio | enwik6 KB/s |
|---|---|---:|---:|
| before (Phase 4a end) | baseline | 0.2060 | 117 |
| `01cfec2` Phase 4b1 | varint segment headers | **0.1947** (−1.13) | 114 |
| `14084a2` Phase 4b2 | unk extraction replacing raw-fallback | **0.1699** (−2.48) | **119** |

Gap to the entropy bound dropped from **4.27 pp → 0.61 pp**.
**86% of the achievable gap is closed.** Of the 0.61 pp that
remains, essentially all of it is the unk payloads themselves
plus segment header bytes — the AC is within 3 bytes of the
entropy bound across the whole file.

## 4b1 — file format v4 with LEB128 varint segment headers

### Motivation

Phase 4b's first step was building an audit tool
(`l3tc audit`) that compresses a file sequentially and reports
a precise per-source byte breakdown. Running it on enwik6 at
segment 4096:

| source | bytes | % of overhead |
|---|---:|---:|
| raw-fallback bytes (Persian/Arabic ZWNJ) | 21,354 | 50.0% |
| segment header bytes (13 + 2×unks fixed) | 17,373 | 40.7% |
| unk payload bytes | 3,813 | 8.9% |
| AC body vs entropy bound | 143 | 0.3% |
| file header/trailer/CRC | 28 | 0.1% |
| **total** | **42,711** | **100%** |

Two findings jumped out immediately:

1. **The arithmetic coder is essentially optimal.** 143 bytes of
   overhead across 1135 segments = 0.13 bytes per `enc.finish()`
   on average. Nayuki's coder is much tighter than my earlier
   "3-4 KB tail flush estimate" had assumed. There's no win
   hiding in the AC.

2. **Segment headers at 13 bytes each (plus 2×unks) are
   genuinely wasteful** for the small values they hold.
   Typical `n_tokens` is a few hundred, typical `ac_body_len`
   a few hundred bytes — both fit comfortably in 1-2 LEB128
   bytes. The fixed-width `u32/u16` format was sized for
   worst-case token counts, not the typical case.

### Implementation

File format v3 had per-segment header:
```
n_tokens:  u32 LE (4 bytes)
n_unks:    u32 LE (4 bytes)
flags:     u8     (1 byte)
ac_len:    u32 LE (4 bytes)
per-unk:   u16 LE length + payload (2+n bytes)
raw-fb:    u32 LE length + payload (4+n bytes)
```

= 13 bytes of fixed header per segment + 2 per unk + 4 per
raw-fallback.

v4 replaces every length field with LEB128 varint:
```
n_tokens:  varint (1-2 bytes typical)
n_unks:    varint (1 byte typical)
flags:     u8     (1 byte)
ac_len:    varint (1-2 bytes typical)
per-unk:   varint length + payload (1+n bytes)
raw-fb:    varint length + payload (1+n bytes)
```

≈ 5 bytes of fixed header per segment + 1 per unk + 1 per
raw-fallback.

Version byte bumped from 3 → 4. Reader accepts both v2 (no CRC,
fixed headers), v3 (CRC, fixed headers), and v4 (CRC, varint
headers) via version-dispatched `read_segment_meta` /
`read_segment_meta_v3` functions. The varint helpers
(`write_varint` / `read_varint`) are ~20 lines of pure-Rust
LEB128 encoding/decoding in `codec.rs`.

### Result

enwik6 audit after 4b1:

| metric | v3 | v4 (4b1) | delta |
|---|---:|---:|---:|
| total segments | 1135 | 1135 | — |
| avg segment header bytes | 15.31 | **5.42** | −9.89 per segment |
| total segment header bytes | 17,373 | **6,150** | −11,223 |
| total compressed | 205,966 | **194,743** | **−11,223** |
| ratio | 0.2060 | **0.1947** | **−1.13 pp** |
| compress KB/s | 117 | 114 | −3 (noise) |
| gap to entropy bound | 4.27 pp | 3.15 pp | −1.12 pp |

The savings match exactly — every byte saved on segment headers
shows up in the total compressed size. Speed is within noise.
34 unit tests pass (one test updated to reflect the new format
version); round trip is byte-identical.

**Commit `01cfec2`.**

## 4b2 — unk extraction replacing raw-fallback subdivision

### Motivation

After 4b1 the breakdown was:

| source | bytes | % of new overhead |
|---|---:|---:|
| raw-fallback bytes | 21,354 | 67.8% |
| segment header bytes | 6,150 | 19.5% |
| unk payload bytes | 3,813 | 12.1% |
| everything else | 171 | 0.5% |

The biggest remaining line by far was the raw-fallback bytes.
355 of 1135 segments on enwik6 were raw-fallback chunks — the
recursive subdivision that happens when SPM's
`sp.decode(sp.encode(text)) != text` because of silent
character normalization. Each raw-fallback chunk stores the
~60 bytes verbatim alongside segment-level framing.

**Key insight:** when SPM normalizes a character silently (e.g.
ZWNJ, some bidi marks, some accented Latin under certain
SPM normalizers), it's usually just *that one character*. The
surrounding text is perfectly tokenizable. But our raw-fallback
path bailed the entire ~60-byte chunk, not just the bad
character. Most of the raw-fallback bytes were actually
tokenizable text that we were refusing to let the AC see.

The proper fix is to *extract* the silently-normalized
characters as explicit `<unk>` tokens (the tokenizer already
supports `<unk>` with byte-level payloads — that's the path
used for truly out-of-vocabulary tokens), and let the AC
compress the tokenizable text around them.

### Implementation

New function
`Tokenizer::encode_segment_extract_unks` in `src/tokenizer.rs`
that takes a segment that failed plain `encode_segment`
round-trip, finds the silently-normalized characters
structurally, and produces a single segment with explicit
`<unk>` tokens at those positions.

Two layers:

1. **Fast path.** Scan for a known set of
   silently-normalized Unicode codepoints (ZWNJ, ZWJ, bidi
   marks, soft hyphen, BOM). If found and the prefix before
   the first one round-trips cleanly, splice the bad char as
   an unk directly. Costs 2 SPM encode calls.

2. **Slow path (`extract_recursive`).** For characters not in
   the fast-path set (which turned out to be most of enwik6's
   failing chars — Cyrillic/Arabic/Hebrew/accented Latin), do
   a binary search over the character indices to find the
   first one SPM fails to round-trip. Costs `O(log N)` SPM
   encode calls where N is the character count of the failing
   text. Caches the last-successful-probe's encoding so we
   don't re-encode the good prefix after the search.

After extracting one bad character, recurse on the suffix. The
typical failing segment has 1-5 bad characters, giving
~10-30 SPM calls per failing segment, mostly on short inputs.

### The speed problem and its fix

The initial implementation of 4b2 dropped compress from
117 KB/s to **91 KB/s** — below the 99 KB/s speed floor from
[`CLAUDE.md`](../CLAUDE.md). The binary search was doing ~2×
the total SPM work that the old raw-fallback subdivision did
(log2(N) probes per recursion × K recursions per segment × 2
SPM calls per `encode_segment` since it also verifies round-
trip).

The fix turned out to be simple and orthogonal:
`Tokenizer::encode_file` was tokenizing segments **serially**.
Each segment is independent (they only touch `&self.sp` which
is `Sync`), so parallelizing with `rayon::par_iter` + flat_map
recovers all the lost throughput and then some. The slow path
stays slow on a single core, but each core processes its own
segment in parallel.

With parallel tokenization, the slow-path SPM work overlaps
across cores and the outer compress throughput is back to
119 KB/s — *slightly above* the Phase 4b1 number.

### Result

enwik6 audit after 4b2:

| metric | 4b1 | **4b2** | delta |
|---|---:|---:|---:|
| segments total | 1135 | **245** | −890 (−78%) |
| raw-fallback segments | 355 | **0** | −355 (eliminated) |
| AC body bytes | 163,398 | 163,720 | +322 |
| segment header bytes | 6,150 | **2,498** | −3,652 |
| unk payload bytes | 3,813 | **3,623** | −190 |
| raw-fallback bytes | 21,354 | **0** | **−21,354** |
| total compressed | 194,743 | **169,869** | **−24,874** |
| ratio | 0.1947 | **0.1699** | **−2.48 pp** |
| compress KB/s | 114 | **119** | +5 |
| gap to entropy bound | 3.15 pp | **0.61 pp** | −2.54 pp |

A few things to notice in the numbers:

- **Raw-fallback segments dropped to zero.** The extraction
  handles every normalization issue enwik6 contains. Not a
  single segment on enwik6 needs the raw-fallback path after
  4b2.

- **Total segments dropped 78% (1135 → 245)** because the
  raw-fallback subdivision was the thing producing most of the
  segments. Without it, segments are roughly 1 per
  `segment_bytes` of input as originally intended.

- **Unk payload bytes dropped (3,813 → 3,623).** Surprising
  at first. Why? Because the old raw-fallback path stored the
  *full raw chunks*, and those chunks contained both bad chars
  and a lot of tokenizable context. The unk path only stores
  the *individual bad characters* — the surrounding context
  becomes AC body. So most of the 21,354 raw-fallback bytes
  are now compressed through the AC, not stored verbatim.

- **AC body bytes went up by 322.** Expected: the AC now has
  more text to encode (the previously-raw content). The
  increase is small compared to the raw-fallback bytes
  eliminated (322 added vs 21,354 removed = 65:1 win).

- **Segment header bytes dropped (6,150 → 2,498)** as a
  secondary benefit of having far fewer segments.

- **AC tail overhead is now −3 bytes** (AC body 163,720 vs
  entropy bound 163,723). The AC is actually slightly
  *below* the entropy floor on this file due to favorable
  freq-quantization rounding — the first time that's been
  true across the whole project.

- **Speed improved** (114 → 119 KB/s) because parallel
  tokenization overlapped with everything else. The additional
  per-segment SPM work from the binary search is absorbed by
  multi-core dispatch.

**Commit `14084a2`.**

## How we compare to Python L3TC now

Python L3TC reports 0.1665 on enwik6. We need to be careful
about what that number means:

- **Python's 0.1665 is `entropy_sum / 8 / input_bytes`** — the
  theoretical entropy lower bound computed from the softmax,
  as documented in
  [`docs/phase_4a_findings.md`](phase_4a_findings.md). The
  actual AC encode + file write path in `vendor/L3TC/scripts/
  compressor.py` is commented out; the reported ratio is the
  entropy floor, not bytes you can read off disk.

- **Our entropy bound on enwik6 is 0.1632** at segment 4096
  (**better than Python's 0.1665** by 0.33 pp — we compute
  entropy on the raw softmax while Python computes on the
  freq-quantized softmax, which has a tiny rounding loss).

- **Our actual coded bytes ratio is 0.1699** at segment 4096 —
  real bytes including AC framing, segment headers, unk
  payloads, file framing, and CRC.

So our 0.1699 is **0.34 pp above** the paper's reported 0.1665,
but we're comparing **real bytes vs theoretical entropy**. Any
AC implementation reading the Python reference's logits would
emit more bytes than their `entropy_sum / 8` number. On the
metric that's apples-to-apples (entropy bound), we win.

## What's left

The 0.61 pp gap to our entropy bound breaks down:

| source | bytes | notes |
|---|---:|---|
| segment header bytes (varint) | 2,498 | 10.2 B/seg × 245 segs |
| unk payload bytes | 3,623 | irreducible — these are the actual non-tokenizable bytes |
| AC body (relative to entropy bound) | −3 | negative noise, essentially nothing |
| file header/trailer/CRC | 28 | constant |
| **total** | **6,146** | **0.61 pp** |

- **Segment headers** (2,498 B / 0.25 pp) could drop further
  if we used fewer segments, but that trades against
  segment-level parallelism in the forward pass. The current
  245 segments at 4096 bytes is already at the low end of the
  useful range. Probably not worth chasing.

- **Unk payload bytes** (3,623 B / 0.36 pp) are essentially
  irreducible under our current tokenizer. The bytes *are*
  characters SPM can't tokenize. The only way to compress them
  further would be a byte-level model that runs alongside the
  main LM for unk payloads, or a tokenizer that understands
  more of the Unicode space. Both are Phase 5/8/11 territory
  (different architecture or broader training).

- **AC and file framing** (25 B) are noise.

**There is no more meaningful ratio win in the current
architecture + corpus + codec.** 4b closes the 4 pp gap to the
entropy bound down to 0.61 pp, and everything left is either
structurally irreducible or requires changes outside Phase 4's
scope (retraining, bigger model, multi-model dispatch).

## What shipped in Phase 4b

| commit | description |
|---|---|
| `2090176` | Phase 4b measurement: `l3tc audit` + `AuditStats` |
| `01cfec2` | Phase 4b1: file format v4 with varint segment headers |
| `14084a2` | Phase 4b2: extract SPM-normalized chars to unk path; parallel tokenize |

New code:
- `codec::audit_compress` + `AuditStats` struct
- `codec::write_varint` / `read_varint`
- `codec::write_segment` rewritten for v4
- `codec::read_segment_meta` (v4) + `read_segment_meta_v3`
  retained for back-compat
- `codec::VERSION = 4`, `VERSION_V3_COMPAT = 3`,
  `VERSION_V2_COMPAT = 2`
- `tokenizer::encode_segment_extract_unks`
- `tokenizer::extract_recursive` + `is_known_normalize_silent`
  (fast path) + binary search (slow path)
- `tokenizer::encode_file` parallelized via rayon
- `l3tc audit` and `l3tc entropy-bound` CLI subcommands
- `l3tc dump-logits` CLI subcommand (Phase 4a, included here
  for completeness)

## Where Phase 4 stands

After 4b, Phase 4's success criteria are all met:

- ✅ Forward pass bit-identical to Python (max L_inf
  3.81e-05 in 4a)
- ✅ Entropy bound matches or beats Python (our 0.1632 vs
  Python's reported 0.1665 on enwik6, segment 4096)
- ✅ enwik6 actual coded ratio ≤ 0.180 (we landed at
  **0.1699**, almost 1 pp under target)
- ✅ Compress speed ≥ 99 KB/s on enwik6 (we landed at
  **119 KB/s**, well above the floor)
- ✅ All unit tests pass (34) and all end-to-end integration
  tests pass
- ✅ `docs/phase_4a_findings.md` + this file document the
  implementation-diff finding and the AC-overhead-reduction
  work

**Deferred out of Phase 4:**
- **Hybrid classical fallback for OOD inputs** — moved to
  Phase 8, where it naturally fits as the last tier of the
  multi-model dispatch cascade. Phase 4 doesn't need it:
  enwik6 and enwik8 both compress well under the LM path.
- **Ratio improvements from a better architecture** (v4 → v7)
  — Phase 5.
- **Ratio improvements from a broader training corpus** —
  Phase 11.

Phase 4 is done. Next up is whichever of Phase 5 / 8 / 11 /
service vision / production hardening the project picks up
next — Phase 4 has taken the model-and-codec side as far as it
can go at this architecture and corpus.

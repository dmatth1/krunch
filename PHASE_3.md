# Phase 3 — File format hardening, stream API, real benchmarks

**Starting point (end of Phase 2.5):**
- 116 KB/s compress, 121 KB/s decompress on enwik6 (1 MB)
- Ratio 0.2061 (best of any compressor in the suite)
- 7.4× faster than Python L3TC-200K
- Byte-identical round trip, 35 unit tests + 4 integration tests passing
- Whole-file-into-memory codec API (no streaming yet)
- File format v2: magic + version + segment list + trailer, no CRC

**Phase 3 goals:**

1. **Production-quality file format.** Add CRC32 of the payload so
   decoders can detect corruption. Add a file-level length field
   that cross-checks against the header and the sum of segment
   sizes. Document the format as stable and versioned so tools can
   negotiate.
2. **Streaming API.** Replace the "read whole file into memory"
   codec entry points with a chunked `encode_reader` /
   `decode_writer` pair that handles arbitrarily large inputs with
   bounded memory. Keep the in-memory helpers as thin wrappers.
3. **Real benchmarks.** Run the full 100 MB enwik8, the 1 GB enwik9
   subset, and the Silesia corpus through both l3tc-rust and the
   classical compressors on the same machine. Commit the JSON +
   summary. This is the numbers we advertise going forward.
4. **Multi-platform release builds.** Produce release artifacts for
   macOS-arm64, macOS-x86_64, and linux-x86_64 without
   `target-cpu=native`. Include each target's measured throughput
   in the release notes.

---

## 3a — File format v3 (CRC + length cross-check)

**Why:** v2 has magic bytes but no integrity check. A single bit
flip in the ac_body produces a decode that either crashes deep in
the AC (bad), decodes to garbage silently (worse), or triggers a
length mismatch at the very end (only half the time). A CRC32 over
the full payload catches corruption up front.

**Changes:**
- Bump `VERSION` to 3.
- After the trailer magic, append a `u32` CRC32 computed over the
  full payload (everything between magic and CRC, exclusive of the
  CRC itself).
- On decode: compute the CRC incrementally as bytes are read; after
  reading the trailer, compare against the stored CRC and return
  `Error::BadChecksum` on mismatch.
- Keep v2 read-compat for at least one release so old files still
  decode. Emit a warning on the CLI when reading a v2 file.

**Dependency:** add `crc32fast` (trivial, 0 transitive deps, the
de-facto Rust CRC32 crate).

**Success criteria:**
- All existing round-trip tests still pass with v3.
- New unit test: flipping one byte in a v3 file produces
  `BadChecksum` on decode.
- New unit test: v2 files still decode (with warning on CLI).
- `iter.sh` still reports the same throughput (CRC32 adds
  ~0.5 GB/s throughput, negligible next to our ~100 KB/s).

---

## 3b — Streaming codec API

**Why:** the current `encode` / `decode` functions take `&[u8]`
inputs and return `Vec<u8>`. That's fine for 1 MB enwik6 but pins
the whole file in memory for the duration. A real production
compressor needs to handle gigabyte inputs (enwik9, database
backups, log files) with bounded memory.

**Design:**
- `pub fn encode_reader<R: Read, W: Write>(src: R, dst: W, model: &Model, tok: &Tokenizer, opts: EncodeOpts) -> Result<EncodeStats>`
- `pub fn decode_writer<R: Read, W: Write>(src: R, dst: W, model: &Model, tok: &Tokenizer) -> Result<DecodeStats>`
- Internally, read one segment's worth of bytes at a time, tokenize,
  encode that segment, flush its ac_body + unks, loop.
- The header's `total_bytes` field becomes a running counter; the
  n_segments field we fill in at the end by seeking back (requires
  `Seek`) or switch to a "segment count is implicit, trailer marks
  end" framing. Prefer the latter — it keeps the writer
  non-seekable and is what zstd does.
- Keep `encode_bytes` / `decode_bytes` as in-memory shims that wrap
  the streaming API.

**Success criteria:**
- A 100 MB enwik8 can be encoded with a streaming API using
  <50 MB of peak RSS (model + tokenizer + constant buffer).
- Existing in-memory tests still pass against the shim.
- New integration test: encode enwik8 via the stream API,
  decompress, diff against original.

---

## 3c — Full enwik8 and Silesia benchmarks

**Why:** we've been advertising enwik6 numbers for 3 phases. Time
to publish what happens on 100 MB and on Silesia (the de-facto
compression benchmark corpus).

**Tasks:**
- Extend `bench/compressors.py` with an `l3tc-rust` wrapper that
  shells out to the release CLI with `--time --verify`.
- Download Silesia via `scripts/download_corpora.sh` (already
  supports it — confirm and run).
- Run classical compressors + l3tc-rust on: enwik8 (100 MB), all
  Silesia files, Canterbury (already done, for completeness).
- Commit JSON + markdown summaries to `bench/results/`.
- Update `README.md` headline table with enwik8 numbers.

**Success criteria:**
- `bench/results/enwik8-l3tc.json` committed.
- `bench/results/silesia-all.json` committed.
- `docs/phase_3_findings.md` written with the numbers in context.

---

## 3d — Multi-platform release builds

**Why:** `.cargo/config.toml` currently sets `target-cpu=native`
for development speed, but that makes the binary non-portable.
Production releases need to run on any macOS-arm64 machine, any
recent x86_64 Linux, etc.

**Tasks:**
- Add a `release` cargo profile that disables `target-cpu=native`.
- Build three artifacts: `aarch64-apple-darwin`,
  `x86_64-apple-darwin`, `x86_64-unknown-linux-gnu`.
- Measure each on a representative machine (CI or local VM) and
  record the throughput delta vs the native build.
- Set up a GitHub Actions workflow that builds and attaches the
  artifacts to a release tag.

**Success criteria:**
- Three release binaries build cleanly on a fresh checkout.
- aarch64-apple-darwin release binary stays within 10% of the
  native build's throughput on enwik6 (NEON intrinsics are already
  gated on `target_arch = "aarch64"`, so they should still fire).
- A tagged GitHub release has all three artifacts attached.

---

## Execution order

1. **3c-lite — enwik8 benchmark** (no code changes, run existing
   CLI on enwik8, commit numbers). Low risk, gives us a real-sized
   reference point immediately.
2. **3a — File format v3 with CRC** (small code change, adds
   integrity check). Bump version, add crc32fast dep, write tests.
3. **3c-full — Silesia benchmark** (needs the l3tc-rust wrapper in
   `bench/compressors.py` to be efficient across many small files).
4. **3b — Streaming API** (largest change, needs care around the
   segment framing). Do this last because the benchmarks and
   format work inform the API design.
5. **3d — Multi-platform release** (mostly CI + docs).

## Non-goals

- Retraining the model (Phase 4).
- Changing the model architecture.
- Adding new classical compressors to the harness — the existing
  set is enough for headline comparisons.
- Fuzzing the decoder (Phase 5).

## Success criteria (Phase 3 exit)

Phase 3 is done when:
- File format is v3 with CRC32, documented as stable.
- Streaming API handles arbitrarily large inputs with bounded RSS.
- enwik8 and Silesia numbers are committed to `bench/results/`.
- Three release binaries exist and are measured on real hardware.
- `docs/phase_3_findings.md` summarizes the numbers and any
  surprises.

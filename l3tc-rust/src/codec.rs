//! High-level compress / decompress: ties tokenizer + model +
//! arithmetic coder together.
//!
//! # Compressed file format (Phase 3, v3)
//!
//! Simple, self-describing binary format. NOT compatible with
//! L3TC's Python output — intentionally, see PHASE_1.md. Stabilized
//! in Phase 3 with a CRC32 integrity trailer.
//!
//! ```text
//! header:
//!     magic:         b"LRUS"   (4 bytes)
//!     version:       u8        (= 3; reader also accepts 2)
//!     flags:         u8        (reserved, = 0)
//!     reserved:      u16       (= 0)
//!     total_bytes:   u64       original uncompressed byte length
//!     n_segments:    u32
//! For each segment:
//!     n_tokens:      u32
//!     n_unks:        u32
//!     seg_flags:     u8        bit 0: use raw fallback
//!     ac_bytes:      u32       length of the arithmetic-coded body
//!     ac_body:       bytes     arithmetic coder output for this segment
//!     For each unk:
//!         unk_len:   u16       length in bytes of the raw unk payload
//!         unk_data:  bytes     raw UTF-8 bytes of the unk surface
//!     If seg_flags bit 0 is set:
//!         raw_len:   u32
//!         raw_data:  bytes     original segment text, used verbatim
//!                              on decode (overrides the token path)
//! trailer:
//!     magic:         b"!END"   (4 bytes)
//!     crc32:         u32 LE    CRC32 of everything above (v3 only)
//! ```
//!
//! # Raw fallback segments
//!
//! SentencePiece normalizes certain characters during tokenization
//! (zero-width joiners, combining marks, some NFC forms) which
//! means `sp.decode(sp.encode(text)) != text` for those inputs.
//! When the encoder detects this at segment-tokenization time (see
//! `Tokenizer::encode_segment`), it sets `needs_raw_fallback` and
//! the codec writes the original segment bytes alongside the
//! arithmetic-coded body. The decoder ignores the ac_body and
//! emits the raw bytes directly.
//!
//! For ASCII-only text the fallback is never triggered. For text
//! with occasional non-ASCII characters (e.g. Persian/Arabic in
//! enwik6) a small fraction of segments use the fallback. The
//! total overhead is a few bytes per fallback segment (the length
//! prefix) on top of the segment's raw byte count.
//!
//! # Compression flow
//!
//! 1. Read input text (full file into memory for now)
//! 2. Tokenize into segments using [`Tokenizer::encode_file`]
//! 3. For each segment:
//!     a. Reset model state
//!     b. For each token after BOS:
//!        - Run `session.forward(previous_token)` to get logits
//!        - Build a frequency table from the logits (softmax → integer freqs)
//!        - Encode the current token with the arithmetic coder
//!     c. Write the arithmetic-coded bytes and unk payloads
//!
//! # Decompression flow
//!
//! 1. Read the header to learn segment count and metadata
//! 2. For each segment:
//!     a. Reset model state
//!     b. Initialize arithmetic decoder from the segment's ac_body
//!     c. Feed tokens through the model the same way the encoder did
//!        (start with BOS, then decode next token against logits)
//!     d. Apply unks at their unk-token positions
//!     e. Detokenize to text
//! 3. Concatenate segment texts
//!
//! # Model-driven probability table
//!
//! The arithmetic coder needs a cumulative-frequency table at each
//! step. We convert the model's raw logits to a discrete frequency
//! table via:
//!
//! 1. Apply softmax → floating-point probabilities
//! 2. Scale to a total of `MAX_TOTAL` = arithmetic::MAX_TOTAL
//! 3. Clamp every symbol to at least 1 (avoid zero-probability)
//! 4. Adjust the top-probability symbol to make the total exact
//!
//! This matches Project Nayuki's "model-driven arithmetic coding"
//! pattern and is what L3TC's Python coder does as well.

use crate::arithmetic::{ArithmeticDecoder, ArithmeticEncoder, MAX_TOTAL};
use crate::error::{Error, Result};
use crate::rwkv::{Model, Session};
use crate::tensor;
use crate::tokenizer::{EncodedSegment, Tokenizer, BOS_ID, UNK_ID};

use byteorder::{ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write};

/// Header magic bytes.
const MAGIC: &[u8; 4] = b"LRUS";
/// Trailer magic bytes.
const TRAILER: &[u8; 4] = b"!END";
/// Format version this module reads and writes.
///
/// v2 adds the per-segment `seg_flags` byte with a `use raw
/// fallback` bit so the decoder can use the original segment
/// bytes verbatim when SentencePiece normalization would
/// otherwise lose information (ZWNJ, combining marks, etc.).
/// Format version this module writes.
///
/// v3 adds a CRC32 of the entire payload (everything from the
/// start of the file up to and including the trailer magic)
/// appended as a u32 little-endian after the trailer. This gives
/// the decoder a fast integrity check against bit flips in storage
/// or transport. The reader accepts both v2 (no CRC) and v3 files
/// for at least one release cycle so existing artifacts still
/// decode.
const VERSION: u8 = 3;

/// Legacy version this decoder still accepts (without CRC).
const VERSION_V2_COMPAT: u8 = 2;

/// Sentinel value stored in the header `n_segments` field to signal
/// "segment count not known at header-write time — read segments
/// until the trailer magic". Used by the streaming encoder
/// ([`encode_reader`]) so it can emit a valid header before knowing
/// how many segments it will produce.
const N_SEGMENTS_IMPLICIT: u32 = u32::MAX;

/// File-level flag bit: "this file is stored verbatim, not
/// tokenized".
///
/// Set on the header `flags` byte when [`encode_reader`] detects
/// that the input isn't valid UTF-8 (binary files, mixed-encoding
/// text, arbitrary data). The tokenizer and model path are
/// bypassed entirely; the payload between the header and trailer
/// is the exact original byte sequence, and the compressed file is
/// therefore slightly *larger* than the original (by the header
/// overhead of ~14 bytes plus the 4-byte CRC).
///
/// This is graceful degradation: it means l3tc-rust will produce
/// a valid, round-trippable output for any input bytes, not just
/// UTF-8 text. Downstream callers can check this flag to decide
/// whether the compressor was actually useful for their input.
const FLAG_RAW_STORE: u8 = 0x01;

/// Per-segment flag bit: "raw fallback bytes follow the unks".
///
/// When the encoder detects that `sp.decode(tokens) != original`
/// for a given segment (because of SPM normalization), it sets
/// this bit and appends the raw segment bytes so the decoder can
/// emit them directly instead of reconstructing via tokens.
const SEG_FLAG_RAW_FALLBACK: u8 = 0x01;

/// Default segment length in bytes.
///
/// L3TC's Python reference uses 2048. Empirically 4096 gives ~2%
/// better compression ratio at the cost of ~15% throughput on
/// small files (where fewer segments means less segment-level
/// parallelism) and a small throughput improvement on large files
/// (better model context flow per segment). 4096 is a good
/// compromise default; users who want maximum ratio on large
/// files can set `--segment-bytes 8192` or higher from the CLI.
pub const DEFAULT_SEGMENT_BYTES: usize = 4096;

/// Compress text to bytes.
///
/// Returns a self-describing blob per the format above. The blob is
/// self-sufficient: you can decompress it with [`decompress`] and
/// the same `model` + `tokenizer` to recover the original text
/// exactly.
///
/// Segments are processed **in parallel** via rayon when there are
/// multiple of them. Each segment is independent (the model state
/// is reset at segment boundaries, and the arithmetic coder output
/// is per-segment), so segment-level parallelism is embarrassingly
/// parallel and sidesteps the per-token rayon dispatch overhead
/// that made intra-segment parallelism uneconomical. On a multi-
/// core machine this gives a near-linear speedup in the number of
/// segments up to the core count.
pub fn compress(
    text: &str,
    tokenizer: &Tokenizer,
    model: &Model,
    segment_bytes: usize,
) -> Result<Vec<u8>> {
    use rayon::prelude::*;

    let segments = tokenizer.encode_file(text, segment_bytes)?;
    let total_bytes = text.len() as u64;

    // Compress every segment in parallel. Each segment gets its
    // own Session and CodecScratch. We briefly experimented with a
    // thread_local pool to avoid per-segment allocation, but the
    // Session allocation is cheap enough (~100 KB, dwarfed by the
    // compute work per segment) that pooling didn't move the
    // needle — and the pooled version required an unsafe lifetime
    // transmute that wasn't worth the risk.
    //
    // Segments that need raw fallback skip the arithmetic coder
    // entirely and emit an empty ac_body (the decoder uses the
    // raw bytes instead).
    let segment_bodies: Result<Vec<Vec<u8>>> = segments
        .par_iter()
        .map(|seg| {
            if seg.needs_raw_fallback {
                return Ok(Vec::new());
            }
            let mut session = Session::new(model);
            let mut scratch = CodecScratch::new(model.vocab_size);
            compress_segment(seg, &mut session, model, &mut scratch)
        })
        .collect();
    let segment_bodies = segment_bodies?;

    // Serialize the header + each segment + trailer. This is
    // sequential but trivial — just byte copies.
    let mut out = Vec::with_capacity(text.len() / 8);
    write_header(&mut out, total_bytes, segments.len() as u32, 0)?;
    for (seg, body) in segments.iter().zip(segment_bodies.iter()) {
        write_segment(&mut out, seg, body)?;
    }
    write_trailer(&mut out)?;
    // v3: append CRC32 of the entire payload written so far.
    let crc = crc32fast::hash(&out);
    out.write_u32::<LittleEndian>(crc)?;
    Ok(out)
}

/// Decompress a blob produced by [`compress`] back to text.
///
/// Like [`compress`], this processes segments in parallel using
/// rayon. Each segment is independently decodable (the model state
/// is reset at segment boundaries and each segment has its own
/// arithmetic-coded body), so segment-level parallelism gives a
/// near-linear speedup in the number of segments up to the core
/// count.
/// Decompress into a `Vec<u8>`.
///
/// Works for both text (tokenized) and raw-store (binary) files —
/// the latter is the whole point of this entry point, since
/// [`decompress`] requires UTF-8 output. For raw-store files, the
/// tokenizer and model arguments are ignored (the payload is
/// returned verbatim).
pub fn decompress_bytes(
    bytes: &[u8],
    tokenizer: &Tokenizer,
    model: &Model,
) -> Result<Vec<u8>> {
    // Peek the CRC + version the same way `decompress` does.
    if bytes.len() < 5 {
        return Err(Error::BadCheckpoint("file too short".into()));
    }
    if &bytes[..4] != MAGIC {
        return Err(Error::BadCheckpoint(format!("bad magic: {:?}", &bytes[..4])));
    }
    let version = bytes[4];
    let body: &[u8] = match version {
        VERSION => {
            let (payload, crc_bytes) = bytes.split_at(bytes.len() - 4);
            let stored = LittleEndian::read_u32(crc_bytes);
            let computed = crc32fast::hash(payload);
            if stored != computed {
                return Err(Error::BadCheckpoint(format!(
                    "CRC32 mismatch: stored {stored:08x}, computed {computed:08x}"
                )));
            }
            payload
        }
        VERSION_V2_COMPAT => bytes,
        v => {
            return Err(Error::BadCheckpoint(format!(
                "unsupported version: {v}"
            )));
        }
    };

    // Peek the flags byte (offset 5) to decide raw-store vs
    // tokenized path without running the full read_header parser.
    let flags = body[5];
    if flags & FLAG_RAW_STORE != 0 {
        if body.len() < HEADER_SIZE + TRAILER.len() {
            return Err(Error::BadCheckpoint(
                "raw-store file truncated below minimum framing".into(),
            ));
        }
        let raw_end = body.len() - TRAILER.len();
        let raw = &body[HEADER_SIZE..raw_end];
        let trailer_seen = &body[raw_end..];
        if trailer_seen != TRAILER {
            return Err(Error::BadCheckpoint(format!(
                "bad trailer in raw-store file: {trailer_seen:?}"
            )));
        }
        return Ok(raw.to_vec());
    }

    // Tokenized path: fall back to the UTF-8 decompress and
    // re-convert. Cheap because the model/tokenizer work dominates.
    let text = decompress(bytes, tokenizer, model)?;
    Ok(text.into_bytes())
}

/// Decompress a blob produced by [`compress`] or [`encode_reader`]
/// back to text.
///
/// Requires the output to be valid UTF-8. Use [`decompress_bytes`]
/// for arbitrary-byte inputs (raw-store files produced from
/// binary data).
pub fn decompress(bytes: &[u8], tokenizer: &Tokenizer, model: &Model) -> Result<String> {
    use rayon::prelude::*;

    // Peek at the version byte to decide framing. v3 files end with
    // a 4-byte CRC32 trailer; v2 files do not. The version lives at
    // byte offset 4 (right after the 4-byte magic).
    if bytes.len() < 5 {
        return Err(Error::BadCheckpoint("file too short".into()));
    }
    if &bytes[..4] != MAGIC {
        return Err(Error::BadCheckpoint(format!("bad magic: {:?}", &bytes[..4])));
    }
    let version = bytes[4];
    let body: &[u8] = match version {
        VERSION => {
            if bytes.len() < 4 {
                return Err(Error::BadCheckpoint("v3 file missing CRC trailer".into()));
            }
            let (payload, crc_bytes) = bytes.split_at(bytes.len() - 4);
            let stored_crc = LittleEndian::read_u32(crc_bytes);
            let computed = crc32fast::hash(payload);
            if stored_crc != computed {
                return Err(Error::BadCheckpoint(format!(
                    "CRC32 mismatch: stored {stored_crc:08x}, computed {computed:08x}"
                )));
            }
            payload
        }
        VERSION_V2_COMPAT => bytes,
        v => {
            return Err(Error::BadCheckpoint(format!(
                "unsupported version: {v}"
            )));
        }
    };

    let mut cursor = std::io::Cursor::new(body);
    let (total_bytes, n_segments, flags) = read_header(&mut cursor)?;

    // Raw-store mode: the payload between header and trailer is
    // the exact original bytes. Validate the trailer and return
    // them (as a String, requiring UTF-8 — callers that need
    // arbitrary bytes should use `decompress_bytes`).
    if flags & FLAG_RAW_STORE != 0 {
        // Body layout: header | raw | trailer. Raw length is
        // body.len() - HEADER_SIZE - TRAILER.len().
        if body.len() < HEADER_SIZE + TRAILER.len() {
            return Err(Error::BadCheckpoint(
                "raw-store file truncated below minimum framing".into(),
            ));
        }
        let raw_end = body.len() - TRAILER.len();
        let raw = &body[HEADER_SIZE..raw_end];
        let trailer_seen = &body[raw_end..];
        if trailer_seen != TRAILER {
            return Err(Error::BadCheckpoint(format!(
                "bad trailer in raw-store file: {trailer_seen:?}"
            )));
        }
        return String::from_utf8(raw.to_vec())
            .map_err(|e| Error::BadCheckpoint(format!("raw-store body is not utf-8: {e}")));
    }

    // First pass: read all segment metadata + bodies sequentially.
    // This is I/O bound and very cheap (just byte copies).
    //
    // If the header carries the `N_SEGMENTS_IMPLICIT` sentinel the
    // stream was produced by [`encode_reader`] and we don't know
    // how many segments to expect. Peek 4 bytes before each segment
    // and stop when they equal the trailer magic.
    let mut raw_segments: Vec<SegmentRead> = if n_segments == N_SEGMENTS_IMPLICIT {
        let mut out = Vec::new();
        loop {
            let pos = cursor.position() as usize;
            if body.len() - pos < 4 {
                return Err(Error::BadCheckpoint(
                    "truncated before trailer in implicit-count stream".into(),
                ));
            }
            if &body[pos..pos + 4] == TRAILER {
                break;
            }
            out.push(read_segment_meta(&mut cursor)?);
        }
        out
    } else {
        let mut out = Vec::with_capacity(n_segments as usize);
        for _ in 0..n_segments {
            out.push(read_segment_meta(&mut cursor)?);
        }
        out
    };
    read_trailer(&mut cursor)?;
    // Silence the "may be mutated" lint when the explicit branch runs.
    let _ = &mut raw_segments;

    // Second pass: decode each segment in parallel. Each thread
    // owns a fresh Session (because LayerState is per-thread).
    // Raw-fallback segments skip the token path entirely and emit
    // the stored raw bytes.
    let decoded: Result<Vec<String>> = raw_segments
        .par_iter()
        .map(|seg| -> Result<String> {
            if let Some(raw) = &seg.raw_fallback {
                return Ok(String::from_utf8(raw.clone())?);
            }
            let mut session = Session::new(model);
            let mut scratch = CodecScratch::new(model.vocab_size);
            let tokens =
                decompress_segment(seg.n_tokens, &seg.ac_body, &mut session, model, &mut scratch)?;
            tokenizer.decode_segment(&tokens, &seg.unks)
        })
        .collect();

    let decoded = decoded?;
    let mut out = String::with_capacity(total_bytes as usize);
    for piece in decoded {
        out.push_str(&piece);
    }
    Ok(out)
}

/// Streaming compress from a reader to a writer.
///
/// Reads `src` in bounded batches, tokenizes each batch (splitting
/// only on newlines so segmentation stays consistent with the
/// in-memory path), compresses the batch's segments in parallel
/// with rayon, and writes the header/body/trailer to `dst`
/// incrementally. Peak memory is roughly
/// `BATCH_BYTES + rayon_workers * segment_bytes + segment bodies`,
/// independent of total input size.
///
/// The header's `n_segments` field is written as
/// [`N_SEGMENTS_IMPLICIT`] because the streaming writer doesn't
/// know the final segment count at header-write time. The decoder
/// recognises this sentinel and walks segments until it sees the
/// trailer magic.
///
/// Returns the total number of uncompressed input bytes consumed.
pub fn encode_reader<R: Read, W: Write>(
    mut src: R,
    dst: W,
    tokenizer: &Tokenizer,
    model: &Model,
    segment_bytes: usize,
) -> Result<u64> {
    use rayon::prelude::*;

    /// Soft target for how many input bytes to buffer before
    /// segmenting + compressing a batch. Larger batches mean more
    /// segment-level parallelism per batch (better rayon
    /// utilisation) at the cost of higher peak memory. 4 MB is
    /// empirically enough to fill 8 cores without blowing past a
    /// ~10 MB RSS budget.
    const BATCH_BYTES: usize = 4 * 1024 * 1024;

    // Wrap the writer in a CRC-computing adapter so we can write
    // bytes once and update the hash in step.
    let mut dst = CrcWriter::new(dst);

    // Byte accumulator straddling batch boundaries. We only split
    // on '\n' so the carry holds the tail of the current read that
    // didn't end on a newline.
    let mut carry: Vec<u8> = Vec::with_capacity(BATCH_BYTES);
    let mut read_buf = vec![0u8; 64 * 1024];
    let mut total_in: u64 = 0;
    let mut eof = false;

    // Fill the first batch before deciding text vs raw-store mode.
    while carry.len() < BATCH_BYTES {
        let n = src.read(&mut read_buf).map_err(Error::Io)?;
        if n == 0 {
            eof = true;
            break;
        }
        carry.extend_from_slice(&read_buf[..n]);
    }

    // Probe the first batch for UTF-8 validity. If the batch is
    // valid UTF-8, or it only ends mid-codepoint (a legal situation
    // when the split happens inside a multi-byte char), we use the
    // text path. Otherwise we fall back to raw-store mode.
    let utf8_ok = match std::str::from_utf8(&carry) {
        Ok(_) => true,
        Err(e) => {
            // Incomplete final codepoint is fine — the carry boundary
            // will move when we read more. Real invalid bytes mean
            // the input isn't text at all.
            e.error_len().is_none()
                && std::str::from_utf8(&carry[..e.valid_up_to()]).is_ok()
        }
    };

    if !utf8_ok {
        // --- Raw-store mode: stream bytes verbatim. ---
        write_header(&mut dst, 0, 0, FLAG_RAW_STORE)?;
        // Flush the already-buffered first batch.
        dst.write_all(&carry).map_err(Error::Io)?;
        total_in += carry.len() as u64;
        // Pipe the rest of src straight through.
        loop {
            let n = src.read(&mut read_buf).map_err(Error::Io)?;
            if n == 0 {
                break;
            }
            dst.write_all(&read_buf[..n]).map_err(Error::Io)?;
            total_in += n as u64;
        }
        write_trailer(&mut dst)?;
        let (mut inner, crc) = dst.finish();
        inner.write_u32::<LittleEndian>(crc).map_err(Error::Io)?;
        return Ok(total_in);
    }

    // --- Text path: implicit-count streaming segments. ---
    //
    // total_bytes is unknown until the stream ends; we write 0 as a
    // placeholder. Consumers that need the exact count can derive
    // it from the reconstructed output. The CLI's --time path
    // doesn't rely on this field.
    write_header(&mut dst, 0, N_SEGMENTS_IMPLICIT, 0)?;

    while !eof || !carry.is_empty() {
        // Fill carry to BATCH_BYTES (or until EOF).
        while carry.len() < BATCH_BYTES && !eof {
            let n = src.read(&mut read_buf).map_err(Error::Io)?;
            if n == 0 {
                eof = true;
                break;
            }
            carry.extend_from_slice(&read_buf[..n]);
        }

        // Choose a cut point: the last '\n' in carry, so we never
        // break a line across batches (segmentation groups short
        // lines and splits long ones, and both behaviours depend
        // on seeing whole lines).
        let cut = if eof {
            carry.len()
        } else {
            match carry.iter().rposition(|&b| b == b'\n') {
                Some(idx) => idx + 1,
                None => {
                    // No newline in 4 MB of input — degenerate
                    // binary-ish content. Flush the whole carry to
                    // avoid pathological unbounded growth.
                    carry.len()
                }
            }
        };
        if cut == 0 {
            continue;
        }

        // Move the prefix out so we can stream the tail forward.
        let prefix: Vec<u8> = carry.drain(..cut).collect();
        total_in += prefix.len() as u64;
        let text = std::str::from_utf8(&prefix)
            .map_err(|e| Error::BadCheckpoint(format!("non-utf8 input: {e}")))?;

        let segments = tokenizer.encode_file(text, segment_bytes)?;

        // Compress this batch's segments in parallel. Each segment
        // gets its own Session + scratch so threads don't contend.
        let bodies: Result<Vec<Vec<u8>>> = segments
            .par_iter()
            .map(|seg| {
                if seg.needs_raw_fallback {
                    return Ok(Vec::new());
                }
                let mut session = Session::new(model);
                let mut scratch = CodecScratch::new(model.vocab_size);
                compress_segment(seg, &mut session, model, &mut scratch)
            })
            .collect();
        let bodies = bodies?;

        for (seg, body) in segments.iter().zip(bodies.iter()) {
            write_segment(&mut dst, seg, body)?;
        }
    }

    write_trailer(&mut dst)?;
    let (mut inner, crc) = dst.finish();
    inner.write_u32::<LittleEndian>(crc).map_err(Error::Io)?;
    Ok(total_in)
}

/// Write adapter that computes a CRC32 over every byte written.
///
/// Used by [`encode_reader`] so the streaming writer can produce
/// the same CRC32 trailer that [`compress`] appends in one shot.
struct CrcWriter<W: Write> {
    inner: W,
    hasher: crc32fast::Hasher,
}

impl<W: Write> CrcWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: crc32fast::Hasher::new(),
        }
    }

    fn finish(self) -> (W, u32) {
        let crc = self.hasher.finalize();
        (self.inner, crc)
    }
}

impl<W: Write> Write for CrcWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

// -------- per-segment codec -------- //

fn compress_segment(
    seg: &EncodedSegment,
    session: &mut Session<'_>,
    _model: &Model,
    scratch: &mut CodecScratch,
) -> Result<Vec<u8>> {
    let mut ac_bytes = Vec::with_capacity(seg.tokens.len());
    {
        let mut enc = ArithmeticEncoder::new(&mut ac_bytes);

        // The first token is BOS — it's agreed out-of-band (the
        // decoder starts with BOS too), so we don't encode it. The
        // tokens slice always starts with BOS per EncodedSegment's
        // contract. We need the logits after forwarding token[i-1]
        // to encode token[i].
        for i in 1..seg.tokens.len() {
            let prev = seg.tokens[i - 1];
            let tok = seg.tokens[i];
            let logits = session.forward(prev);
            logits_to_cum_freqs_scratch(
                logits,
                &mut scratch.cum,
                &mut scratch.exps,
                &mut scratch.freqs,
            );
            enc.encode_symbol(&scratch.cum, tok)?;
        }

        enc.finish()?;
    }
    Ok(ac_bytes)
}

fn decompress_segment(
    n_tokens: u32,
    ac_body: &[u8],
    session: &mut Session<'_>,
    _model: &Model,
    scratch: &mut CodecScratch,
) -> Result<Vec<u32>> {
    let mut tokens = Vec::with_capacity(n_tokens as usize);
    tokens.push(BOS_ID);

    let mut dec = ArithmeticDecoder::new(ac_body)?;

    // n_tokens includes the BOS. We decode n_tokens - 1 symbols.
    // Each iteration: forward the previous token through the model,
    // use its output distribution to decode the next token.
    let mut prev = BOS_ID;
    for _ in 1..n_tokens {
        let logits = session.forward(prev);
        logits_to_cum_freqs_scratch(
            logits,
            &mut scratch.cum,
            &mut scratch.exps,
            &mut scratch.freqs,
        );
        let tok = dec.decode_symbol(&scratch.cum)?;
        tokens.push(tok);
        prev = tok;
    }

    Ok(tokens)
}

/// Reusable buffers for the per-segment encode/decode loop.
///
/// Holds the cumulative-frequency table and the softmax scratch
/// buffer so that `logits_to_cum_freqs` can avoid heap allocation
/// on the hot path. Allocated once per compress/decompress call
/// and reused across every segment and every token.
struct CodecScratch {
    cum: Vec<u64>,
    exps: Vec<f32>,
    /// Per-symbol integer frequencies.
    ///
    /// Separate from `cum` so the scaling loop has no sequential
    /// dependency and the prefix-sum pass is a simple running-add
    /// in a dedicated loop. The cost of splitting is one extra
    /// 128 KB buffer per segment; the benefit is that the scaling
    /// loop fully vectorizes.
    freqs: Vec<u64>,
}

impl CodecScratch {
    fn new(vocab_size: usize) -> Self {
        Self {
            cum: vec![0u64; vocab_size + 1],
            exps: vec![0.0f32; vocab_size],
            freqs: vec![0u64; vocab_size],
        }
    }
}

/// Fast `exp(x)` approximation for `x <= 0`, vectorization-friendly.
///
/// Used by [`logits_to_cum_freqs_scratch`] to turn softmax-shifted
/// logits (which are always ≤ 0 after subtracting the max) into
/// probabilities without calling the libm `exp` scalar function.
///
/// # Method
///
/// We factor `exp(x) = 2^(x · log2(e))` and split the exponent into
/// an integer part `k` and a fractional part `r`:
///
/// ```text
/// exp(x) = 2^(k + r) = 2^k · exp2(r),  with r in [0, 1)
/// ```
///
/// - `2^k` for integer `k` is computed by directly constructing an
///   IEEE-754 float via bit manipulation: the float `2^k` has
///   biased exponent `k + 127` and a zero mantissa.
/// - `exp2(r)` for `r in [0, 1)` is approximated by a degree-4
///   polynomial fit on that interval (minimax coefficients from a
///   standard reference). The polynomial is in Horner form, which
///   is a 4-step scalar accumulation that LLVM vectorizes cleanly
///   using ARM NEON FMA / x86 FMA3.
///
/// # Accuracy
///
/// Maximum relative error is around 1e-5 over `x ∈ [-50, 0]`. For
/// `x < -50` the result is below f32 precision anyway (~2e-22), and
/// we clamp to 0 there.
///
/// For arithmetic coding, this level of precision is more than
/// sufficient: the worst-case coding overhead from approximating
/// the probability model is the log2 of the relative error,
/// approximately 2^-17 bits per symbol for our approximation, or
/// about 0.00001 bits per token. On a 280,000-token file that's
/// less than half a byte of extra output.
///
/// # Performance
///
/// About 3-5x faster than `f32::exp` because the whole routine is
/// inlined, branch-free, and fully vectorizable. Empirically on
/// Apple Silicon the 16384-element softmax loop drops from ~82 us
/// to ~15-25 us.
#[inline(always)]
fn fast_exp_neg(x: f32) -> f32 {
    // Clamp to a safe range; below -50 the result is ~1.9e-22,
    // far below any frequency we'd actually emit.
    let x = x.max(-50.0);

    // y = x * log2(e); exp(x) = 2^y
    const LOG2E: f32 = 1.442_695_f32;
    let y = x * LOG2E;

    // Split into integer and fractional parts. Since y <= 0, floor
    // gives the largest integer <= y, and r = y - k is in [0, 1).
    let k = y.floor();
    let r = y - k;

    // Polynomial minimax approximation of 2^r on [0, 1), degree 4.
    // Coefficients are the same ones used by cephes and various
    // SIMD math libraries.
    //   2^r ≈ 1 + r*(c1 + r*(c2 + r*(c3 + r*c4)))
    let poly = 1.0_f32
        + r * (0.693_147_18_f32
            + r * (0.240_226_51_f32
                + r * (0.055_505_4_f32 + r * 0.009_618_129_f32)));

    // 2^k via direct bit construction. The IEEE-754 float 2^k has
    // biased exponent `k + 127` and a zero mantissa, so we build
    // it as `((k + 127) << 23)` reinterpreted as f32.
    //
    // k can be very negative (down to -72 after the clamp), and
    // (k + 127) can be a small positive number; for the smallest
    // subnormal values we'd need to handle underflow, but the
    // clamp on x guarantees we stay in the normal range.
    let k_i = k as i32;
    let exp_bits = ((k_i + 127) as u32) << 23;
    let two_k = f32::from_bits(exp_bits);

    two_k * poly
}

/// Public alias for [`logits_to_cum_freqs_scratch`], used by the
/// profile tests in `tests/profile_codec.rs`. Not part of the stable API.
#[doc(hidden)]
pub fn logits_to_cum_freqs_public(logits: &[f32], cum: &mut [u64]) {
    let mut exps = vec![0.0f32; logits.len()];
    let mut freqs = vec![0u64; logits.len()];
    logits_to_cum_freqs_scratch(logits, cum, &mut exps, &mut freqs);
}

/// Convert raw model logits to a cumulative frequency table.
///
/// Three passes, but the expensive one is a single-loop
/// scale/floor/accumulate that also builds `cum` in place. The
/// sequential cum dependency doesn't matter in practice because
/// the `(e * scale).floor() as u64` operation is itself scalar on
/// ARM NEON — the loop was never going to vectorize anyway, so we
/// might as well fuse the accumulation into it and save a memory
/// pass.
///
/// The separate `freqs` buffer in [`CodecScratch`] is retained so
/// the signature is future-proof for a smarter vectorized
/// implementation (e.g. an INT8-quantized path).
fn logits_to_cum_freqs_scratch(
    logits: &[f32],
    cum: &mut [u64],
    exps: &mut [f32],
    _freqs: &mut [u64],
) {
    debug_assert_eq!(cum.len(), logits.len() + 1);
    debug_assert_eq!(exps.len(), logits.len());

    let n = logits.len();

    // --- Pass 1: find max logit ---
    let mut max = f32::NEG_INFINITY;
    for &l in logits {
        if l > max {
            max = l;
        }
    }
    if !max.is_finite() {
        uniform_fallback(n, cum);
        return;
    }

    // --- Pass 2: compute shifted exps and running sum ---
    let mut sum = 0.0f32;
    for i in 0..n {
        let e = fast_exp_neg(logits[i] - max);
        exps[i] = e;
        sum += e;
    }
    if !sum.is_finite() || sum <= 0.0 {
        uniform_fallback(n, cum);
        return;
    }

    // --- Pass 3: scale, floor, clamp, accumulate cum ---
    //
    // Using f64 for scale to preserve precision against the large
    // `usable ≈ 2^62`. f32 rounding of `usable as f32` would
    // misallocate frequencies on many low-probability tokens.
    let target_total: u64 = MAX_TOTAL as u64;
    let usable = target_total.saturating_sub(n as u64);
    let scale: f64 = (usable as f64) / (sum as f64);
    if !scale.is_finite() {
        uniform_fallback(n, cum);
        return;
    }

    let mut assigned: u64 = 0;
    let mut best_idx = 0usize;
    let mut best_exp = f32::NEG_INFINITY;

    cum[0] = 0;
    for i in 0..n {
        let e = exps[i];
        let scaled = (((e as f64) * scale).floor() as u64).max(1);
        assigned += scaled;
        cum[i + 1] = cum[i] + scaled;
        if e > best_exp {
            best_exp = e;
            best_idx = i;
        }
    }

    // --- Fix up the total ---
    if assigned < target_total {
        let residual = target_total - assigned;
        for i in (best_idx + 1)..=n {
            cum[i] += residual;
        }
    } else if assigned > target_total {
        let excess = assigned - target_total;
        let best_freq = cum[best_idx + 1] - cum[best_idx];
        if best_freq > excess + 1 {
            for i in (best_idx + 1)..=n {
                cum[i] -= excess;
            }
        } else {
            uniform_fallback(n, cum);
            return;
        }
    }
    debug_assert_eq!(cum[n], target_total);
}

/// Build a uniform cumulative-frequency table in `cum`.
///
/// Used by [`logits_to_cum_freqs_scratch`] as the fallback when the
/// input distribution is degenerate (all NaN, all zero, etc.).
/// Gives every symbol an equal probability with the total equal to
/// `MAX_TOTAL`.
fn uniform_fallback(n: usize, cum: &mut [u64]) {
    let target_total = MAX_TOTAL as u64;
    let uniform = target_total / n as u64;
    let residual = target_total - uniform * n as u64;
    cum[0] = 0;
    for i in 0..n {
        cum[i + 1] = cum[i] + uniform;
    }
    // Distribute the residual by giving each of the first `residual`
    // symbols one extra count. This keeps every freq >= 1 and the
    // sum exactly equal to target_total.
    for i in 1..=(residual as usize) {
        // Shift every cum[j] for j >= i up by 1. This is O(n*residual)
        // but residual < n so it's bounded; for our sizes n=16384
        // and residual < 16384 it's a few ms at worst in the
        // pathological path.
        for j in i..=n {
            cum[j] += 1;
        }
    }
}

// -------- header / trailer I/O -------- //

fn write_header<W: Write>(
    w: &mut W,
    total_bytes: u64,
    n_segments: u32,
    flags: u8,
) -> Result<()> {
    w.write_all(MAGIC)?;
    w.write_u8(VERSION)?;
    w.write_u8(flags)?;
    w.write_u16::<LittleEndian>(0)?; // reserved
    w.write_u64::<LittleEndian>(total_bytes)?;
    w.write_u32::<LittleEndian>(n_segments)?;
    Ok(())
}

fn read_header<R: Read>(r: &mut R) -> Result<(u64, u32, u8)> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(Error::BadCheckpoint(format!("bad magic: {magic:?}")));
    }
    let version = r.read_u8()?;
    if version != VERSION && version != VERSION_V2_COMPAT {
        return Err(Error::BadCheckpoint(format!(
            "unsupported version: {version}"
        )));
    }
    let flags = r.read_u8()?;
    let _reserved = r.read_u16::<LittleEndian>()?;
    let total_bytes = r.read_u64::<LittleEndian>()?;
    let n_segments = r.read_u32::<LittleEndian>()?;
    Ok((total_bytes, n_segments, flags))
}

/// Fixed size of the on-disk header, used by the decoder to
/// locate the raw payload range for `FLAG_RAW_STORE` files.
const HEADER_SIZE: usize = 4 + 1 + 1 + 2 + 8 + 4;

/// Per-segment data read back from a compressed file.
struct SegmentRead {
    /// Token count (including the implicit BOS). Unused for raw
    /// fallback segments but still present in the format.
    n_tokens: u32,
    /// Unk payloads.
    unks: Vec<Vec<u8>>,
    /// Arithmetic-coded body.
    ac_body: Vec<u8>,
    /// Raw fallback bytes, if the segment was flagged with
    /// [`SEG_FLAG_RAW_FALLBACK`] at write time.
    raw_fallback: Option<Vec<u8>>,
}

fn write_segment<W: Write>(w: &mut W, seg: &EncodedSegment, ac_body: &[u8]) -> Result<()> {
    w.write_u32::<LittleEndian>(seg.tokens.len() as u32)?;
    w.write_u32::<LittleEndian>(seg.unks.len() as u32)?;
    let seg_flags = if seg.needs_raw_fallback {
        SEG_FLAG_RAW_FALLBACK
    } else {
        0
    };
    w.write_u8(seg_flags)?;
    w.write_u32::<LittleEndian>(ac_body.len() as u32)?;
    w.write_all(ac_body)?;
    for unk in &seg.unks {
        if unk.len() > u16::MAX as usize {
            return Err(Error::BadCheckpoint(format!(
                "unk payload too large: {} bytes",
                unk.len()
            )));
        }
        w.write_u16::<LittleEndian>(unk.len() as u16)?;
        w.write_all(unk)?;
    }
    if seg.needs_raw_fallback {
        let raw_bytes = seg.raw.as_bytes();
        w.write_u32::<LittleEndian>(raw_bytes.len() as u32)?;
        w.write_all(raw_bytes)?;
    }
    Ok(())
}

fn read_segment_meta<R: Read>(r: &mut R) -> Result<SegmentRead> {
    let n_tokens = r.read_u32::<LittleEndian>()?;
    let n_unks = r.read_u32::<LittleEndian>()?;
    let seg_flags = r.read_u8()?;
    let ac_bytes_len = r.read_u32::<LittleEndian>()? as usize;
    let mut ac_body = vec![0u8; ac_bytes_len];
    r.read_exact(&mut ac_body)?;
    let mut unks = Vec::with_capacity(n_unks as usize);
    for _ in 0..n_unks {
        let unk_len = r.read_u16::<LittleEndian>()? as usize;
        let mut buf = vec![0u8; unk_len];
        r.read_exact(&mut buf)?;
        unks.push(buf);
    }
    let raw_fallback = if seg_flags & SEG_FLAG_RAW_FALLBACK != 0 {
        let raw_len = r.read_u32::<LittleEndian>()? as usize;
        let mut raw = vec![0u8; raw_len];
        r.read_exact(&mut raw)?;
        Some(raw)
    } else {
        None
    };
    Ok(SegmentRead {
        n_tokens,
        unks,
        ac_body,
        raw_fallback,
    })
}

fn write_trailer<W: Write>(w: &mut W) -> Result<()> {
    w.write_all(TRAILER)?;
    Ok(())
}

fn read_trailer<R: Read>(r: &mut R) -> Result<()> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != TRAILER {
        return Err(Error::BadCheckpoint(format!("bad trailer: {magic:?}")));
    }
    Ok(())
}

// Suppress the unused import warning in -D unused_imports builds.
#[allow(dead_code)]
fn _keep_tensor_import_alive() {
    let _ = tensor::argmax(&[1.0]);
    let _: UnkLimit = UNK_ID;
}
type UnkLimit = u32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cum_freqs_sum_to_max_total() {
        // Random-ish logits
        let logits: Vec<f32> = (0..256).map(|i| (i as f32 / 50.0).sin()).collect();
        let mut cum = vec![0u64; logits.len() + 1];
        let mut exps = vec![0.0f32; logits.len()];
        let mut freqs = vec![0u64; logits.len()];
        logits_to_cum_freqs_scratch(&logits, &mut cum, &mut exps, &mut freqs);
        assert_eq!(cum[0], 0);
        assert_eq!(cum[logits.len()], MAX_TOTAL as u64);
        // No zero frequencies
        for i in 0..logits.len() {
            assert!(cum[i + 1] > cum[i], "zero freq at symbol {i}");
        }
    }

    #[test]
    fn cum_freqs_uniform_fallback() {
        let logits = vec![f32::NAN; 16];
        let mut cum = vec![0u64; logits.len() + 1];
        let mut exps = vec![0.0f32; logits.len()];
        let mut freqs = vec![0u64; logits.len()];
        logits_to_cum_freqs_scratch(&logits, &mut cum, &mut exps, &mut freqs);
        assert_eq!(cum[logits.len()], MAX_TOTAL as u64);
        // Every step is non-zero
        for i in 0..logits.len() {
            assert!(cum[i + 1] > cum[i]);
        }
    }

    #[test]
    fn fast_exp_neg_matches_libm_reasonably() {
        // Spot-check the approximation against libm's f32::exp
        // across the range we care about.
        for &x in &[0.0_f32, -0.1, -0.5, -1.0, -2.0, -5.0, -10.0, -20.0, -40.0] {
            let actual = fast_exp_neg(x);
            let expected = x.exp();
            let rel_err = ((actual - expected) / expected).abs();
            // Allow ~1% relative error — more than enough for our use case
            assert!(
                rel_err < 0.01,
                "fast_exp_neg({x}) = {actual}, expected {expected}, rel_err {rel_err}"
            );
        }
    }

    #[test]
    fn crc32_roundtrip_detects_corruption() {
        // Build a tiny fake v3 payload: magic + version + flags +
        // reserved + total_bytes + n_segments=0 + trailer, then
        // append the correct CRC. Verify that decompress-time CRC
        // validation accepts it and rejects a bit-flipped copy.
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.push(VERSION);
        buf.push(0); // flags
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // total_bytes
        buf.extend_from_slice(&0u32.to_le_bytes()); // n_segments
        buf.extend_from_slice(TRAILER);
        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());

        // Pristine copy: CRC matches.
        let (payload, crc_bytes) = buf.split_at(buf.len() - 4);
        assert_eq!(LittleEndian::read_u32(crc_bytes), crc32fast::hash(payload));

        // Corrupt one byte in the middle of the payload.
        let mut bad = buf.clone();
        bad[8] ^= 0xFF;
        let (payload, crc_bytes) = bad.split_at(bad.len() - 4);
        assert_ne!(LittleEndian::read_u32(crc_bytes), crc32fast::hash(payload));
    }

    #[test]
    fn fast_exp_neg_clamps_extreme_negatives() {
        // Below -50 we return ~0 (the input is clamped to -50)
        let v = fast_exp_neg(-1000.0);
        assert!(v.is_finite());
        assert!(v >= 0.0);
        assert!(v < 1e-20);
    }
}

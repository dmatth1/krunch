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
//! 2. `freqs[i] = round(p[i] * 10_000_000)`
//! 3. `freqs[i] = max(freqs[i], 1)` (avoid zero-probability)
//! 4. Cumulative table is the prefix sum of `freqs`.
//!
//! This is intentionally bit-identical with the Python L3TC
//! reference (`vendor/L3TC/scripts/compressor.py` lines 273-276).
//! Earlier phases used a finer scale (~2^62) with `floor` rounding
//! and a residual fixup pass; that diverged from Python in three
//! ways simultaneously and was 4 percentage points worse on enwik6.
//! Phase 4a closes that gap by matching Python exactly.

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
/// **v4** (Phase 4b1) packs the per-segment header fields as
/// LEB128 varints instead of fixed-width u32/u16. Typical token
/// counts and ac-body lengths fit in 1-2 bytes each, dropping
/// the per-segment header from 13 bytes to ~5 bytes (savings of
/// ~9 KB / 0.9 pp on enwik6). Per-unk lengths and the raw_len
/// field are also varints, saving another ~3 KB / 0.3 pp on
/// enwik6 (mostly from the 355 raw-fallback segments). The
/// file-level CRC32 trailer from v3 is unchanged.
///
/// v3 added the CRC32 of the entire payload appended as a u32
/// little-endian after the trailer. The reader still accepts v3
/// files (and v2 files without CRC) for at least one release
/// cycle so existing artifacts decode.
const VERSION: u8 = 4;

/// Previous format with v3 CRC32 trailer + fixed-width segment
/// headers. Decoder accepts these as long as we can detect them
/// from the version byte.
const VERSION_V3_COMPAT: u8 = 3;

/// Legacy version this decoder still accepts (no CRC, fixed
/// segment headers).
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
        VERSION | VERSION_V3_COMPAT => {
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
        VERSION | VERSION_V3_COMPAT => {
            if bytes.len() < 4 {
                return Err(Error::BadCheckpoint("v3+ file missing CRC trailer".into()));
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

    // v4 uses varint segment headers; v2/v3 use fixed-width.
    // We branch via a bool inside the read loop because the
    // segment readers are generic over `R: Read` and so can't be
    // coerced to a concrete function pointer.
    let varint_segments = version == VERSION;

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
            let seg = if varint_segments {
                read_segment_meta(&mut cursor)?
            } else {
                read_segment_meta_v3(&mut cursor)?
            };
            out.push(seg);
        }
        out
    } else {
        let mut out = Vec::with_capacity(n_segments as usize);
        for _ in 0..n_segments {
            let seg = if varint_segments {
                read_segment_meta(&mut cursor)?
            } else {
                read_segment_meta_v3(&mut cursor)?
            };
            out.push(seg);
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

/// Streaming decompress from a reader to a writer.
///
/// Symmetric counterpart to [`encode_reader`]. The high-value case
/// is **raw-store files** (binary inputs) where the compressed
/// body is the same size as the original — slurping a multi-GB
/// raw-store file into memory just to copy it back out would
/// defeat the purpose. For raw-store streams this function pipes
/// bytes from `src` to `dst` with a constant 8-byte lookahead and
/// validates the CRC32 trailer at EOF.
///
/// **Tokenized files** are read in full into memory because the
/// compressed body is ~5× smaller than the output and decoding
/// any segment requires the model state, which doesn't usefully
/// stream. The output write itself is still incremental.
///
/// Returns the number of uncompressed output bytes written.
pub fn decode_writer<R: Read, W: Write>(
    mut src: R,
    mut dst: W,
    tokenizer: &Tokenizer,
    model: &Model,
) -> Result<u64> {
    // Read the fixed-size header up front so we can branch on
    // version and flags before deciding how to consume the rest.
    let mut header = [0u8; HEADER_SIZE];
    src.read_exact(&mut header).map_err(Error::Io)?;
    if &header[..4] != MAGIC {
        return Err(Error::BadCheckpoint(format!(
            "bad magic: {:?}",
            &header[..4]
        )));
    }
    let version = header[4];
    if version != VERSION && version != VERSION_V3_COMPAT && version != VERSION_V2_COMPAT {
        return Err(Error::BadCheckpoint(format!(
            "unsupported version: {version}"
        )));
    }
    let flags = header[5];
    // v3 introduced the CRC32 trailer; v4 keeps it. Only v2 lacks it.
    let has_crc = version == VERSION || version == VERSION_V3_COMPAT;

    if flags & FLAG_RAW_STORE != 0 {
        // --- Raw-store streaming path ---
        //
        // Layout: header(20) | raw(N) | trailer(4) | crc(4 if v3+).
        // We've consumed the header. Hash it, then stream the body
        // with an 8-byte lookahead so we never write the trailer or
        // CRC to dst.
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&header);
        let tail_size = if has_crc { 8 } else { 4 };
        let mut tail: Vec<u8> = Vec::with_capacity(tail_size + 64 * 1024);
        let mut buf = vec![0u8; 64 * 1024];
        let mut written: u64 = 0;
        loop {
            let n = src.read(&mut buf).map_err(Error::Io)?;
            if n == 0 {
                break;
            }
            tail.extend_from_slice(&buf[..n]);
            // Drain everything that's safely outside the lookahead.
            if tail.len() > tail_size {
                let drain_n = tail.len() - tail_size;
                let drained = &tail[..drain_n];
                hasher.update(drained);
                dst.write_all(drained).map_err(Error::Io)?;
                written += drain_n as u64;
                tail.drain(..drain_n);
            }
        }
        if tail.len() != tail_size {
            return Err(Error::BadCheckpoint(format!(
                "raw-store stream truncated: tail has {} bytes, expected {tail_size}",
                tail.len()
            )));
        }
        let trailer_seen = &tail[..4];
        if trailer_seen != TRAILER {
            return Err(Error::BadCheckpoint(format!(
                "bad trailer in raw-store stream: {trailer_seen:?}"
            )));
        }
        // Trailer bytes are part of the CRC payload.
        hasher.update(trailer_seen);
        if has_crc {
            let stored_crc = LittleEndian::read_u32(&tail[4..8]);
            let computed = hasher.finalize();
            if stored_crc != computed {
                return Err(Error::BadCheckpoint(format!(
                    "CRC32 mismatch in raw-store stream: stored {stored_crc:08x}, computed {computed:08x}"
                )));
            }
        }
        return Ok(written);
    }

    // --- Tokenized path ---
    //
    // Slurp the remainder of src (it's at most ~25% of the output
    // text size) and reuse the existing in-memory decode. The
    // output write is then a single chunk; we could also stream
    // segment outputs as they decode, but parallelism in
    // decompress() makes that nontrivial without losing speed.
    let mut rest = Vec::new();
    src.read_to_end(&mut rest).map_err(Error::Io)?;
    let mut full = Vec::with_capacity(HEADER_SIZE + rest.len());
    full.extend_from_slice(&header);
    full.extend_from_slice(&rest);
    let payload = decompress_bytes(&full, tokenizer, model)?;
    dst.write_all(&payload).map_err(Error::Io)?;
    Ok(payload.len() as u64)
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

/// Per-source byte breakdown returned by [`audit_compress`].
///
/// Used by the `l3tc audit` CLI subcommand to figure out where the
/// gap between actual coded bytes and the theoretical entropy bound
/// actually goes. Phase 4b debugging only — not part of the stable
/// API.
#[derive(Debug, Default, Clone)]
pub struct AuditStats {
    /// Original input size in bytes.
    pub input_bytes: u64,
    /// Number of segments produced by the tokenizer.
    pub n_segments: u64,
    /// Number of segments that took the raw-fallback path.
    pub n_raw_fallback_segments: u64,
    /// Total token count across all segments (including BOS per
    /// segment).
    pub n_tokens: u64,
    /// Number of (current_token, next_token) pairs the AC actually
    /// encodes — i.e. excludes the BOS that opens each segment.
    pub n_predict_steps: u64,
    /// Theoretical entropy lower bound in bits, computed from the
    /// raw softmax. `entropy_bits / 8` is the floor of what any AC
    /// could possibly achieve.
    pub entropy_bits: f64,
    /// Sum of arithmetic-encoded body bytes across all segments.
    pub ac_body_bytes: u64,
    /// Sum of segment-header bytes (the fixed 13-byte struct per
    /// segment + 2 bytes per unk).
    pub segment_header_bytes: u64,
    /// Sum of unk payload bytes across all segments.
    pub unk_payload_bytes: u64,
    /// Sum of raw-fallback payload bytes across all segments.
    pub raw_fallback_bytes: u64,
    /// File-level header (magic + version + flags + reserved +
    /// total_bytes + n_segments).
    pub file_header_bytes: u64,
    /// File-level trailer magic.
    pub file_trailer_bytes: u64,
    /// File-level CRC32 trailer (v3).
    pub file_crc_bytes: u64,
}

impl AuditStats {
    /// Total bytes the compressor would actually emit if we ran the
    /// full pipeline.
    pub fn total_compressed_bytes(&self) -> u64 {
        self.file_header_bytes
            + self.file_trailer_bytes
            + self.file_crc_bytes
            + self.segment_header_bytes
            + self.ac_body_bytes
            + self.unk_payload_bytes
            + self.raw_fallback_bytes
    }

    /// Theoretical entropy floor in bytes (`ceil(entropy_bits / 8)`).
    pub fn entropy_bound_bytes(&self) -> u64 {
        (self.entropy_bits / 8.0).ceil() as u64
    }

    /// Bytes that the AC + framing pays beyond the entropy bound.
    /// This is what Phase 4b is trying to drive down.
    pub fn overhead_bytes(&self) -> i64 {
        self.total_compressed_bytes() as i64 - self.entropy_bound_bytes() as i64
    }
}

/// Phase 4b debug: compress `text` with the per-source byte
/// breakdown filled in. Mirrors [`compress`] sequentially (no rayon)
/// so the per-segment counters can accumulate cleanly.
pub fn audit_compress(
    text: &str,
    tokenizer: &Tokenizer,
    model: &Model,
    segment_bytes: usize,
) -> Result<AuditStats> {
    let segments = tokenizer.encode_file(text, segment_bytes)?;
    let mut stats = AuditStats {
        input_bytes: text.len() as u64,
        n_segments: segments.len() as u64,
        file_header_bytes: HEADER_SIZE as u64,
        file_trailer_bytes: TRAILER.len() as u64,
        file_crc_bytes: 4,
        ..Default::default()
    };

    let mut session = Session::new(model);
    let mut scratch = CodecScratch::new(model.vocab_size);

    // Helper: byte length of LEB128-encoded value (matches
    // `write_varint` exactly).
    fn varint_len(mut v: u64) -> u64 {
        let mut n = 1u64;
        v >>= 7;
        while v > 0 {
            n += 1;
            v >>= 7;
        }
        n
    }

    for seg in &segments {
        // v4 per-segment header (matches `write_segment`):
        //   varint(n_tokens) + varint(n_unks) + 1(flags) + varint(ac_len)
        //   plus per-unk:    varint(unk_len) + payload
        //   plus raw-fallback: varint(raw_len) + payload
        stats.segment_header_bytes += varint_len(seg.tokens.len() as u64);
        stats.segment_header_bytes += varint_len(seg.unks.len() as u64);
        stats.segment_header_bytes += 1; // flags byte
        // ac_len is filled in below once we know it.
        for unk in &seg.unks {
            stats.segment_header_bytes += varint_len(unk.len() as u64);
            stats.unk_payload_bytes += unk.len() as u64;
        }
        if seg.needs_raw_fallback {
            stats.n_raw_fallback_segments += 1;
            stats.segment_header_bytes += varint_len(0); // ac_len = 0
            stats.segment_header_bytes += varint_len(seg.raw.len() as u64);
            stats.raw_fallback_bytes += seg.raw.len() as u64;
            stats.n_tokens += seg.tokens.len() as u64;
            continue;
        }

        // Walk the segment, accumulating entropy + encoding into
        // a fresh AC. We run both passes inline so we can compare
        // per-segment AC bytes to per-segment entropy bits later
        // if we want to.
        session.reset();
        let mut ac_bytes = Vec::with_capacity(seg.tokens.len());
        {
            let mut enc = ArithmeticEncoder::new(&mut ac_bytes);
            for i in 1..seg.tokens.len() {
                let prev = seg.tokens[i - 1];
                let tok = seg.tokens[i];
                let logits = session.forward(prev);

                // Entropy: -log2(softmax_p[tok]) computed in f64.
                let mut max = f32::NEG_INFINITY;
                for &l in logits {
                    if l > max {
                        max = l;
                    }
                }
                let mut sum = 0.0f64;
                for &l in logits {
                    sum += ((l - max) as f64).exp();
                }
                let target = (logits[tok as usize] - max) as f64;
                let log_p = target - sum.ln();
                stats.entropy_bits += -log_p / std::f64::consts::LN_2;
                stats.n_predict_steps += 1;

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
        stats.ac_body_bytes += ac_bytes.len() as u64;
        // varint(ac_len) — accounted for here once we know the body
        // length. Mirrors the order in `write_segment`.
        stats.segment_header_bytes += varint_len(ac_bytes.len() as u64);
        stats.n_tokens += seg.tokens.len() as u64;
    }

    Ok(stats)
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

/// Public alias for [`logits_to_cum_freqs_scratch`], used by the
/// profile tests in `tests/profile_codec.rs`. Not part of the stable API.
#[doc(hidden)]
pub fn logits_to_cum_freqs_public(logits: &[f32], cum: &mut [u64]) {
    let mut exps = vec![0.0f32; logits.len()];
    let mut freqs = vec![0u64; logits.len()];
    logits_to_cum_freqs_scratch(logits, cum, &mut exps, &mut freqs);
}

/// Convert raw model logits to a cumulative frequency table,
/// **matching the Python L3TC reference exactly**.
///
/// The Python reference (`vendor/L3TC/scripts/compressor.py`) does:
///
/// ```python
/// probs = torch.softmax(logits, dim=-1)
/// freqs = torch.round(probs * 10_000_000).int()
/// freqs = torch.max(freqs, freqs.new_ones(freqs.size()))
/// ```
///
/// We mirror this exactly:
///
/// 1. Compute `softmax(logits)` in f32 using libm `exp` (matches
///    PyTorch's softmax — `fast_exp_neg` introduces ~1% relative
///    error per call, which compounds across 16384 symbols and
///    measurably hurts ratio).
/// 2. `freqs[i] = round(p[i] * PYTHON_FREQ_TOTAL).max(1)` where
///    `PYTHON_FREQ_TOTAL = 10_000_000`.
/// 3. Cumulative table = prefix sum of `freqs`.
///
/// The arithmetic coder accepts any `total ≤ MAX_TOTAL = 2^62`,
/// so the ~10M total here is well within bounds. There is **no
/// residual fixup**: whatever the rounding produces is what the
/// AC sees, exactly as in Python.
///
/// Earlier (Phase 1-3) we used a finer scale (~2^62) with `floor`
/// rounding plus a residual fixup pass distributing the truncation
/// error to the top symbol. That looked like it should give better
/// resolution, but it diverged from Python in three ways
/// simultaneously (`floor` vs `round`, scale, fixup) and ended up
/// 4 percentage points worse on enwik6. Phase 4a closes that gap
/// by matching Python bit-for-bit.
///
/// `_freqs` is unused in this implementation but kept in the
/// signature for the hot-path scratch reuse pattern in
/// [`CodecScratch`].
fn logits_to_cum_freqs_scratch(
    logits: &[f32],
    cum: &mut [u64],
    exps: &mut [f32],
    _freqs: &mut [u64],
) {
    debug_assert_eq!(cum.len(), logits.len() + 1);
    debug_assert_eq!(exps.len(), logits.len());

    /// Total target frequency, matching Python L3TC's
    /// `freqs = round(probs * 10_000_000)`.
    const PYTHON_FREQ_TOTAL: u64 = 10_000_000;

    let n = logits.len();

    // --- Pass 1: find max logit (numerical-stability shift) ---
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

    // --- Pass 2: shifted exp + running sum ---
    //
    // Vectorized NEON path (`tensor::softmax_shifted_exp_sum`)
    // on aarch64 via a hand-rolled `exp_f32x4_neon` using a
    // degree-6 minimax polynomial for `2^r`. Max relative error
    // is under 5e-7 — tighter than the ~1% tolerance that broke
    // the Phase 4a diff harness — so the resulting cum_freqs
    // table is bit-identical to the libm path after the
    // Phase 4a `round(p * 10_000_000); max(1)` quantization
    // absorbs the residual.
    //
    // Phase 4a briefly used libm `f32::exp` here because the
    // old `fast_exp_neg` polynomial was too loose; Phase 4c1
    // replaces libm with a tighter polynomial and measures
    // the entropy bound as a regression gate.
    let sum = tensor::softmax_shifted_exp_sum(logits, max, exps);
    if !sum.is_finite() || sum <= 0.0 {
        uniform_fallback(n, cum);
        return;
    }

    // --- Pass 3: round(p * 10M); max(1); accumulate cum ---
    //
    // Mirror Python exactly: per-symbol freq is
    // `max(1, round(prob * 10_000_000))`. No residual fixup.
    let inv_sum = 1.0f32 / sum;
    cum[0] = 0;
    let mut total: u64 = 0;
    for i in 0..n {
        let prob = exps[i] * inv_sum;
        let scaled = (prob * PYTHON_FREQ_TOTAL as f32).round() as i64;
        let freq = scaled.max(1) as u64;
        total += freq;
        cum[i + 1] = total;
    }

    // Sanity: total must be ≤ MAX_TOTAL for the AC to accept it.
    // With PYTHON_FREQ_TOTAL = 10M and n ≤ 16384, the max
    // achievable total is ~10M + 16384 (every freq rounds up by 1
    // and the max(1) clamp pushes near-zero freqs to 1). Both are
    // far below MAX_TOTAL = 2^62, so this never triggers.
    debug_assert!(total <= MAX_TOTAL as u64);
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
    if version != VERSION && version != VERSION_V3_COMPAT && version != VERSION_V2_COMPAT {
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

/// LEB128 unsigned varint encoder. Each byte holds 7 bits of
/// value plus a high "continuation" bit that's 1 if more bytes
/// follow. Used by the v4 segment-header packing.
///
/// Typical sizes:
///   value < 128       → 1 byte
///   value < 16384     → 2 bytes
///   value < 2097152   → 3 bytes
fn write_varint<W: Write>(w: &mut W, mut value: u64) -> Result<()> {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            w.write_u8(byte).map_err(Error::Io)?;
            return Ok(());
        }
        w.write_u8(byte | 0x80).map_err(Error::Io)?;
    }
}

/// LEB128 unsigned varint decoder. Reads up to 10 bytes (enough
/// for any u64). Errors out on a malformed sequence longer than
/// that.
fn read_varint<R: Read>(r: &mut R) -> Result<u64> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    for _ in 0..10 {
        let byte = r.read_u8().map_err(Error::Io)?;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
    }
    Err(Error::BadCheckpoint("varint too long".into()))
}

/// Write one segment in the **v4** format.
///
/// v3 used fixed-width u32/u16 fields totaling 13 bytes per
/// segment header (plus per-unk u16 length and per-raw u32
/// length). v4 packs all length fields as LEB128 varints, which
/// drops a typical text-only segment header from 13 bytes to ~5
/// bytes and a typical raw-fallback segment from 17 bytes to ~7
/// bytes. The actual ac_body / unk payloads / raw payloads are
/// unchanged.
fn write_segment<W: Write>(w: &mut W, seg: &EncodedSegment, ac_body: &[u8]) -> Result<()> {
    write_varint(w, seg.tokens.len() as u64)?;
    write_varint(w, seg.unks.len() as u64)?;
    let seg_flags = if seg.needs_raw_fallback {
        SEG_FLAG_RAW_FALLBACK
    } else {
        0
    };
    w.write_u8(seg_flags).map_err(Error::Io)?;
    write_varint(w, ac_body.len() as u64)?;
    w.write_all(ac_body).map_err(Error::Io)?;
    for unk in &seg.unks {
        write_varint(w, unk.len() as u64)?;
        w.write_all(unk).map_err(Error::Io)?;
    }
    if seg.needs_raw_fallback {
        let raw_bytes = seg.raw.as_bytes();
        write_varint(w, raw_bytes.len() as u64)?;
        w.write_all(raw_bytes).map_err(Error::Io)?;
    }
    Ok(())
}

/// Read one segment in the **v4** format.
fn read_segment_meta<R: Read>(r: &mut R) -> Result<SegmentRead> {
    let n_tokens = read_varint(r)? as u32;
    let n_unks = read_varint(r)? as usize;
    let seg_flags = r.read_u8()?;
    let ac_bytes_len = read_varint(r)? as usize;
    let mut ac_body = vec![0u8; ac_bytes_len];
    r.read_exact(&mut ac_body)?;
    let mut unks = Vec::with_capacity(n_unks);
    for _ in 0..n_unks {
        let unk_len = read_varint(r)? as usize;
        let mut buf = vec![0u8; unk_len];
        r.read_exact(&mut buf)?;
        unks.push(buf);
    }
    let raw_fallback = if seg_flags & SEG_FLAG_RAW_FALLBACK != 0 {
        let raw_len = read_varint(r)? as usize;
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

/// Read one segment in the **v2/v3 fixed-width** format.
///
/// Kept for back-compat: any v2 or v3 file produced before
/// Phase 4b1 still decodes through this path. v4 readers should
/// use [`read_segment_meta`] instead.
fn read_segment_meta_v3<R: Read>(r: &mut R) -> Result<SegmentRead> {
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
    fn cum_freqs_match_python_total() {
        // Random-ish logits. Phase 4a: we now match Python's
        // `freqs = round(probs * 10_000_000); max(1)` scheme, so
        // the total is approximately 10M (not MAX_TOTAL = 2^62).
        // Exact value depends on rounding noise + the max(1) clamp.
        let logits: Vec<f32> = (0..256).map(|i| (i as f32 / 50.0).sin()).collect();
        let mut cum = vec![0u64; logits.len() + 1];
        let mut exps = vec![0.0f32; logits.len()];
        let mut freqs = vec![0u64; logits.len()];
        logits_to_cum_freqs_scratch(&logits, &mut cum, &mut exps, &mut freqs);
        assert_eq!(cum[0], 0);
        // Total is bounded above by 10M + n (every freq could
        // round up by 1) and bounded below by max(n, 10M - n).
        let total = cum[logits.len()];
        assert!(total <= 10_000_000 + logits.len() as u64);
        assert!(total >= logits.len() as u64);
        assert!(total <= MAX_TOTAL as u64);
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
        // The uniform fallback still uses MAX_TOTAL because it
        // takes a different code path (for degenerate inputs).
        assert_eq!(cum[logits.len()], MAX_TOTAL as u64);
        // Every step is non-zero
        for i in 0..logits.len() {
            assert!(cum[i + 1] > cum[i]);
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

}

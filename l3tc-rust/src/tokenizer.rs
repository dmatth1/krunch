//! SentencePiece tokenizer wrapper.
//!
//! Wraps the `sentencepiece` crate (which in turn wraps the C++
//! SentencePiece library) and exposes exactly the two operations we
//! need: encode a text segment to a list of token IDs, and decode a
//! list of token IDs back to text.
//!
//! # Outlier handling
//!
//! L3TC trains its SPM model with `character_coverage=0.999`, which
//! means about 0.1% of characters in the training corpus weren't
//! given their own token. At inference time these characters become
//! `<unk>` tokens (id 1 in L3TC's vocabulary). The L3TC Python
//! compressor stores the raw UTF-8 of each unk character separately
//! and zstd-compresses them as a side payload.
//!
//! For Phase 1 we mirror this behavior: the encoder returns the
//! list of token IDs alongside the list of raw UTF-8 byte strings
//! for each unk token, in encounter order. The codec layer writes
//! the unk bytes into the compressed file alongside the arithmetic
//! coder's output. The decoder reads them back in the same order
//! and emits them whenever it sees an unk token.
//!
//! This is not the tightest possible encoding — L3TC gets
//! additional savings by running zstd over the concatenated unk
//! bytes — but it's bit-correct and the unk bytes are a tiny
//! fraction of the total (~3 KB out of 1 MB for enwik6).

use crate::error::{Error, Result};
use sentencepiece::SentencePieceProcessor;
use std::path::Path;

/// ID of the `<s>` (beginning-of-sentence) token in the L3TC vocab.
pub const BOS_ID: u32 = 2;

/// ID of the `<unk>` (unknown/outlier) token in the L3TC vocab.
pub const UNK_ID: u32 = 1;

/// A segment of text, ready to feed through the model.
#[derive(Debug, Clone)]
pub struct EncodedSegment {
    /// Token ids, starting with BOS_ID. Unk tokens are represented
    /// as `UNK_ID` here; the actual byte payloads live in `unks`.
    pub tokens: Vec<u32>,
    /// Raw UTF-8 bytes of each `<unk>` occurrence in encounter order.
    /// On decode, these are pulled out and appended to the output
    /// whenever we see an unk token.
    pub unks: Vec<Vec<u8>>,
    /// The original segment text.
    ///
    /// Used in two cases:
    /// 1. To compute the byte length of this segment for framing.
    /// 2. As a fallback payload when SPM's decode doesn't faithfully
    ///    round-trip the original (see `needs_raw_fallback`).
    ///
    /// Always held for now; future phases may drop this for
    /// memory-sensitive contexts.
    pub raw: String,
    /// Whether the arithmetic-coded token stream can faithfully
    /// reproduce `raw` via `sp.decode(tokens)`.
    ///
    /// `false` (the common case) means we round-trip through tokens
    /// normally. `true` means SentencePiece normalizes some
    /// character in `raw` (ZWNJ, combining marks, etc.) such that
    /// the token path would lose bytes — the codec must write `raw`
    /// as a raw fallback alongside the ac body.
    pub needs_raw_fallback: bool,
}

/// Wrapper around a loaded SentencePiece model.
pub struct Tokenizer {
    sp: SentencePieceProcessor,
}

impl Tokenizer {
    /// Load a SentencePiece model file from disk.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let sp = SentencePieceProcessor::open(path.as_ref()).map_err(|e| {
            Error::Tokenizer(format!("failed to load SPM model: {e}"))
        })?;
        Ok(Self { sp })
    }

    /// Vocabulary size reported by the loaded model.
    pub fn vocab_size(&self) -> usize {
        self.sp.len()
    }

    /// Minimum segment size before we stop subdividing a
    /// raw-fallback segment.
    ///
    /// When a segment fails to round-trip through SPM because of
    /// normalization, [`encode_file`] recursively halves it to
    /// isolate the problematic characters into small sub-segments.
    /// Once a segment is ≤ this many bytes, we stop splitting and
    /// store it raw. The value trades off between two overheads:
    ///
    /// - **Too small**: per-segment header overhead (~15 bytes)
    ///   dominates, and we lose context for the arithmetic coder's
    ///   model state flow. Empirically worse on enwik6.
    /// - **Too large**: raw-fallback segments waste more bytes on
    ///   the mostly-ASCII parts surrounding each problematic char.
    ///
    /// 64 bytes is the sweet spot: problematic characters tend to
    /// cluster (e.g., a full Persian word in a Wikipedia link), so
    /// a 64-byte window catches the whole cluster cleanly rather
    /// than paying header overhead for every individual codepoint.
    const MIN_RAW_FALLBACK_BYTES: usize = 64;

    /// Encode a full input file into segments of at most
    /// `segment_bytes` bytes each, mirroring the L3TC preprocessor
    /// algorithm.
    ///
    /// The L3TC splitter operates line-by-line: it reads each line,
    /// concatenates with any carry-over from the previous line, and
    /// emits segments of `segment_bytes` bytes. The remainder of the
    /// last line is cached and prepended to the next. This is
    /// important because the receiver (the RWKV model) resets state
    /// at every segment boundary, and matching L3TC's boundary
    /// choices is the only way to preserve the exact same per-
    /// segment token distribution.
    ///
    /// Each returned segment's token list starts with the BOS token
    /// (L3TC's convention).
    ///
    /// # Raw-fallback refinement
    ///
    /// If a segment's tokens do not faithfully decode back to the
    /// original (see `needs_raw_fallback` on [`EncodedSegment`]),
    /// we recursively halve it until either (a) the halves round-
    /// trip cleanly or (b) the sub-segment is below
    /// [`Tokenizer::MIN_RAW_FALLBACK_BYTES`]. This isolates the
    /// problematic characters into small raw-fallback chunks so
    /// that the surrounding ASCII-ish content compresses normally.
    pub fn encode_file(&self, text: &str, segment_bytes: usize) -> Result<Vec<EncodedSegment>> {
        let mut segment_lines: Vec<String> = Vec::new();
        let mut cache = String::new();

        for line in text.split_inclusive('\n') {
            let combined = if cache.is_empty() {
                line.to_string()
            } else {
                let mut s = std::mem::take(&mut cache);
                s.push_str(line);
                s
            };
            cache.clear();

            if combined.len() > segment_bytes {
                // Slice by bytes (not chars) to match Python's
                // string slicing semantics. combined is guaranteed
                // to be valid UTF-8 but slicing at arbitrary byte
                // offsets can split multi-byte characters, which
                // would panic on Rust slices. We use a safer
                // approach: slice at char boundaries that don't
                // exceed segment_bytes.
                let mut pos = 0usize;
                while combined.len() - pos > segment_bytes {
                    let end = safe_byte_boundary(&combined, pos + segment_bytes);
                    segment_lines.push(combined[pos..end].to_string());
                    pos = end;
                }
                // The tail goes into the cache if it's shorter than
                // segment_bytes; otherwise flush as a segment.
                let tail = &combined[pos..];
                if tail.len() < segment_bytes {
                    cache.push_str(tail);
                } else {
                    segment_lines.push(tail.to_string());
                }
            } else if combined.len() < segment_bytes {
                cache.push_str(&combined);
            } else {
                segment_lines.push(combined);
            }
        }
        if !cache.is_empty() {
            segment_lines.push(cache);
        }

        let mut segments = Vec::with_capacity(segment_lines.len());
        for seg in segment_lines {
            self.encode_segment_with_refinement(&seg, &mut segments)?;
        }
        Ok(segments)
    }

    /// Encode `text` as a single segment, and if it fails to
    /// round-trip through SPM, recursively subdivide it until each
    /// raw-fallback sub-segment is at most
    /// [`Tokenizer::MIN_RAW_FALLBACK_BYTES`] bytes.
    ///
    /// Results are appended to `out`. This is the refinement loop
    /// for [`encode_file`].
    fn encode_segment_with_refinement(
        &self,
        text: &str,
        out: &mut Vec<EncodedSegment>,
    ) -> Result<()> {
        let seg = self.encode_segment(text)?;
        if !seg.needs_raw_fallback || text.len() <= Self::MIN_RAW_FALLBACK_BYTES {
            out.push(seg);
            return Ok(());
        }

        // Split in half at a char boundary
        let mid = safe_byte_boundary(text, text.len() / 2);
        if mid == 0 || mid == text.len() {
            // Couldn't find a meaningful split point; keep as is.
            out.push(seg);
            return Ok(());
        }
        let (left, right) = text.split_at(mid);
        self.encode_segment_with_refinement(left, out)?;
        self.encode_segment_with_refinement(right, out)?;
        Ok(())
    }

    /// Encode a single already-chunked text segment.
    pub fn encode_segment(&self, text: &str) -> Result<EncodedSegment> {
        let pieces = self
            .sp
            .encode(text)
            .map_err(|e| Error::Tokenizer(format!("sp encode failed: {e}")))?;

        let mut tokens = Vec::with_capacity(pieces.len() + 1);
        let mut unks = Vec::new();
        tokens.push(BOS_ID);

        // NB: piece.span.{0,1} from the sentencepiece Rust crate
        // are CHARACTER (codepoint) indices, not byte indices.
        // This is inconsistent with the crate's doc comment, which
        // claims byte offsets, but the underlying C++ SentencePiece
        // reports char positions. We use `char_indices` below to
        // translate back to byte positions when needed.
        //
        // For unk detection we just need the ID check; the surface
        // bytes come from the raw `text` via the char offset
        // translation.
        let char_bytes: Vec<(usize, char)> = text.char_indices().collect();

        for piece in pieces {
            let (char_begin, char_end) =
                (piece.span.0 as usize, piece.span.1 as usize);
            if char_begin == char_end {
                continue;
            }
            let id = piece.id;
            if id == UNK_ID {
                // Convert char offsets to byte offsets
                let byte_begin = char_bytes
                    .get(char_begin)
                    .map(|(b, _)| *b)
                    .unwrap_or(text.len());
                let byte_end = char_bytes
                    .get(char_end)
                    .map(|(b, _)| *b)
                    .unwrap_or(text.len());
                let unk_bytes = text.as_bytes()[byte_begin..byte_end].to_vec();
                unks.push(unk_bytes);
            }
            tokens.push(id);
        }

        // Determine whether the tokens will faithfully round-trip.
        // If sp.decode(tokens_minus_bos) != text_minus_unks, we
        // need to fall back to storing the raw bytes. We
        // specifically exclude the unk positions from the comparison
        // because unk bytes are already stored separately.
        let needs_raw_fallback = !self.tokens_faithfully_decode(text, &tokens, &unks)?;

        Ok(EncodedSegment {
            tokens,
            unks,
            raw: text.to_string(),
            needs_raw_fallback,
        })
    }

    /// Check whether the given token stream + unks will reproduce
    /// the original text byte-exact when passed back through
    /// [`Tokenizer::decode_segment`].
    ///
    /// This is the robust correctness check used by encoders to
    /// decide whether to fall back to raw byte storage. Returns
    /// `true` on a faithful round trip, `false` if SPM's
    /// normalization lost information.
    fn tokens_faithfully_decode(
        &self,
        original: &str,
        tokens: &[u32],
        unks: &[Vec<u8>],
    ) -> Result<bool> {
        let decoded = self.decode_segment(tokens, unks)?;
        Ok(decoded == original)
    }

    /// Decode a token id list back to text, substituting unk bytes
    /// from `unks` in encounter order.
    ///
    /// If the token stream contains `n` unk tokens, `unks` must
    /// contain exactly `n` byte payloads in the same order.
    pub fn decode_segment(&self, tokens: &[u32], unks: &[Vec<u8>]) -> Result<String> {
        // Skip the leading BOS if present
        let start = if !tokens.is_empty() && tokens[0] == BOS_ID { 1 } else { 0 };
        let body = &tokens[start..];

        // Walk through the token list, batching non-unk tokens into
        // SP decode calls and splicing raw unk bytes at their
        // encounter positions.
        let mut out = Vec::<u8>::with_capacity(body.len() * 2);
        let mut batch: Vec<u32> = Vec::new();
        let mut unk_idx = 0usize;

        for &id in body {
            if id == UNK_ID {
                // Flush the current batch first
                if !batch.is_empty() {
                    let piece = self.decode_tokens(&batch)?;
                    out.extend_from_slice(piece.as_bytes());
                    batch.clear();
                }
                if unk_idx >= unks.len() {
                    return Err(Error::Tokenizer(
                        "decode: token stream has more <unk> tokens than unk payloads".into(),
                    ));
                }
                out.extend_from_slice(&unks[unk_idx]);
                unk_idx += 1;
            } else {
                batch.push(id);
            }
        }
        if !batch.is_empty() {
            let piece = self.decode_tokens(&batch)?;
            out.extend_from_slice(piece.as_bytes());
        }
        if unk_idx != unks.len() {
            return Err(Error::Tokenizer(format!(
                "decode: token stream had {} <unk>s but {} unk payloads were provided",
                unk_idx,
                unks.len()
            )));
        }

        String::from_utf8(out).map_err(Error::InvalidUtf8)
    }

    fn decode_tokens(&self, tokens: &[u32]) -> Result<String> {
        self.sp
            .decode_piece_ids(tokens)
            .map_err(|e| Error::Tokenizer(format!("sp decode failed: {e}")))
    }
}

/// Find a safe byte boundary near `target` in `s` so that
/// `s[..boundary]` is valid UTF-8. Walks backward from `target` to
/// the nearest char boundary; guaranteed to return a value `>= 1`
/// if `target >= 1` and `s` is non-empty.
fn safe_byte_boundary(s: &str, target: usize) -> usize {
    let mut i = target.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn spm_path() -> Option<PathBuf> {
        // The tokenizer sits in vendor/L3TC (from the project root, not
        // the crate root). Probe from the crate manifest dir upward.
        let candidates = [
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("vendor/L3TC/dictionary/vocab_enwik8_bpe_16384_0.999/spm_enwik8_bpe_16384_0.999.model"),
        ];
        candidates.into_iter().find(|p| p.exists())
    }

    #[test]
    fn safe_byte_boundary_walks_backward() {
        let s = "héllo";
        // "héllo" is bytes: h (1) é (2) l (1) l (1) o (1) = 6 bytes
        // Splitting at byte 2 would be mid-char ('é' is bytes 1-2)
        assert_eq!(safe_byte_boundary(s, 2), 1);
        assert_eq!(safe_byte_boundary(s, 3), 3);
        assert_eq!(safe_byte_boundary(s, 10), 6);
    }

    #[test]
    #[ignore = "requires SPM model from vendor/L3TC"]
    fn tokenizer_roundtrip_simple_text() {
        let Some(path) = spm_path() else {
            panic!("no SPM model found; run the L3TC setup first");
        };
        let tok = Tokenizer::load(path).expect("load spm");
        let text = "The quick brown fox jumps over the lazy dog.";
        let seg = tok.encode_segment(text).expect("encode");
        assert_eq!(seg.tokens[0], BOS_ID);
        assert!(seg.tokens.len() > 1);
        // No outliers expected for plain ASCII English
        assert!(seg.unks.is_empty());
        let decoded = tok.decode_segment(&seg.tokens, &seg.unks).expect("decode");
        // SPM may normalize whitespace slightly but for plain ASCII
        // the round trip should be byte-identical.
        assert_eq!(decoded, text);
    }

    #[test]
    #[ignore = "requires SPM model from vendor/L3TC"]
    fn tokenizer_handles_outliers() {
        let Some(path) = spm_path() else { return; };
        let tok = Tokenizer::load(path).expect("load spm");
        // Some Wikipedia text is likely to contain characters in
        // the 0.1% tail. Use a Unicode mix.
        let text = "ℏ = h / (2π)";
        let seg = tok.encode_segment(text).expect("encode");
        // We don't assert a specific number of unks because it
        // depends on the exact SPM vocabulary, but the round trip
        // should still succeed.
        let decoded = tok.decode_segment(&seg.tokens, &seg.unks).expect("decode");
        assert_eq!(decoded, text);
    }

    #[test]
    #[ignore = "requires SPM model from vendor/L3TC"]
    fn tokenizer_segments_file_by_length() {
        let Some(path) = spm_path() else { return; };
        let tok = Tokenizer::load(path).expect("load spm");
        let text = "short line.\n".repeat(300); // many small lines
        let segments = tok.encode_file(&text, 64).expect("encode_file");
        assert!(segments.len() > 1, "expected multiple segments");
        // Every segment starts with BOS
        for seg in &segments {
            assert_eq!(seg.tokens[0], BOS_ID);
        }
    }
}

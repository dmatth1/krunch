//! Arithmetic coding.
//!
//! Port of Project Nayuki's reference arithmetic coder, which is the
//! same implementation L3TC uses in `util/arithmeticcoding.py`.
//! This is pure integer math on `u64` state variables — no floating
//! point, fully deterministic, same output on every platform.
//!
//! # Contract
//!
//! - The encoder and decoder must be driven with the **same
//!   probability model** in the same order, symbol by symbol. If
//!   the model produces different cumulative frequencies on the
//!   two sides, the decoded stream will diverge.
//! - "Frequency" here is a raw integer. The coder normalizes to a
//!   total; sum of frequencies at each step is the `total` argument.
//! - `total` must be ≤ [`MAX_TOTAL`] to maintain the coder's
//!   invariants.
//!
//! # Termination
//!
//! The encoder must call [`ArithmeticEncoder::finish`] after its
//! last symbol. The decoder naturally terminates when it has seen
//! the expected number of symbols (which is a pre-agreed contract
//! between encoder and decoder, not something stored in the stream).
//!
//! # References
//!
//! - <https://www.nayuki.io/page/reference-arithmetic-coding>
//! - L3TC `util/arithmeticcoding.py`

use crate::bitio::{BitReader, BitWriter};
use crate::error::{Error, Result};
use std::io::{Read, Write};

/// Number of bits in `low` and `high` state variables.
///
/// L3TC uses 64 bits (matching the Python `STATE_SIZE = 64`). Must
/// match between encoder and decoder.
pub const STATE_SIZE: u32 = 64;

/// Maximum value representable in `STATE_SIZE` bits plus one.
/// Conceptually `2^STATE_SIZE`.
///
/// We store `low` and `high` as `u64`, so "max range" is
/// `1u128 << STATE_SIZE` and overflow-safe operations use `u128`
/// for intermediates where needed. In practice with `STATE_SIZE = 64`,
/// the arithmetic fits in `u64` once the shift semantics are handled
/// carefully.
pub const MAX_RANGE: u128 = 1u128 << STATE_SIZE;

/// Minimum range during coding. See Nayuki's page for the derivation.
pub const MIN_RANGE: u128 = (MAX_RANGE >> 2) + 2;

/// Maximum allowed total frequency per update.
///
/// If a caller passes a probability model whose cumulative total
/// exceeds this, the coder returns an error rather than producing
/// silently wrong output.
pub const MAX_TOTAL: u128 = MIN_RANGE;

/// Mask of `STATE_SIZE` ones.
const MASK: u128 = MAX_RANGE - 1;

/// Mask of the top bit at width `STATE_SIZE`.
const TOP_MASK: u128 = MAX_RANGE >> 1;

/// Mask of the second-highest bit at width `STATE_SIZE`.
const SECOND_MASK: u128 = TOP_MASK >> 1;

/// Encoder: maps a sequence of (symbol, frequency table) pairs to a
/// bit stream.
///
/// Call [`ArithmeticEncoder::encode_symbol`] once per symbol, then
/// [`ArithmeticEncoder::finish`] exactly once at the end.
pub struct ArithmeticEncoder<W: Write> {
    writer: BitWriter<W>,
    low: u128,
    high: u128,
    /// Number of underflow bits we've deferred — see Nayuki's page
    /// for the "E3 scaling" explanation.
    num_underflow: u64,
}

impl<W: Write> ArithmeticEncoder<W> {
    /// Construct a new encoder writing to `sink`.
    pub fn new(sink: W) -> Self {
        Self {
            writer: BitWriter::new(sink),
            low: 0,
            high: MASK,
            num_underflow: 0,
        }
    }

    /// Encode one symbol against a cumulative frequency table.
    ///
    /// `cum_freqs[i]` is the total count of symbols strictly less
    /// than `i`, so `cum_freqs[0] = 0` and
    /// `cum_freqs[n_symbols] = total`. This matches the Nayuki
    /// reference coder's calling convention.
    ///
    /// `symbol` must be a valid index into `cum_freqs` (less than
    /// `cum_freqs.len() - 1`).
    ///
    /// Returns [`Error::ZeroProbability`] if the selected symbol has
    /// zero frequency (i.e. `cum_freqs[symbol + 1] == cum_freqs[symbol]`).
    pub fn encode_symbol(&mut self, cum_freqs: &[u64], symbol: u32) -> Result<()> {
        let idx = symbol as usize;
        debug_assert!(
            idx + 1 < cum_freqs.len(),
            "symbol {symbol} out of range for cum_freqs of len {}",
            cum_freqs.len()
        );
        let total = cum_freqs[cum_freqs.len() - 1] as u128;
        if total == 0 || total > MAX_TOTAL {
            return Err(Error::ZeroProbability { symbol });
        }
        let sym_low = cum_freqs[idx] as u128;
        let sym_high = cum_freqs[idx + 1] as u128;
        if sym_high == sym_low {
            return Err(Error::ZeroProbability { symbol });
        }

        // Update range
        let range = self.high - self.low + 1;
        let new_high = self.low + (range * sym_high) / total - 1;
        let new_low = self.low + (range * sym_low) / total;
        self.low = new_low;
        self.high = new_high;

        // Shift out bits that are now decided (either both top bits
        // agree → emit that bit; or we're in the middle-two-quarters
        // zone → defer with "underflow" tracking).
        while ((self.low ^ self.high) & TOP_MASK) == 0 {
            // Top bits agree — emit it
            let bit = ((self.low >> (STATE_SIZE - 1)) & 1) as u8;
            self.writer.write_bit(bit).map_err(Error::Io)?;
            // Emit any deferred underflow bits as the opposite
            while self.num_underflow > 0 {
                self.writer.write_bit(bit ^ 1).map_err(Error::Io)?;
                self.num_underflow -= 1;
            }
            self.low = (self.low << 1) & MASK;
            self.high = ((self.high << 1) & MASK) | 1;
        }

        // Underflow: low starts with 01, high starts with 10 →
        // we can't emit a bit yet, but we can strip the second-highest
        // bit from both and remember that we owe an opposite bit.
        while (self.low & !self.high & SECOND_MASK) != 0 {
            self.num_underflow += 1;
            self.low = (self.low << 1) & (MASK >> 1);
            self.high = ((self.high << 1) & (MASK >> 1)) | TOP_MASK | 1;
        }

        Ok(())
    }

    /// Finalize the stream. Must be called exactly once after the
    /// final symbol is encoded. Returns the underlying writer.
    pub fn finish(mut self) -> Result<W> {
        // Write one bit to disambiguate the final range, plus any
        // deferred underflow bits.
        self.writer.write_bit(1).map_err(Error::Io)?;
        // Flush the bit writer — this pads any trailing partial byte
        // with zeros on the right.
        let inner = self.writer.finish().map_err(Error::Io)?;
        Ok(inner)
    }
}

/// Decoder: reverses [`ArithmeticEncoder`] given the same sequence
/// of frequency tables.
///
/// The caller is responsible for knowing how many symbols to decode.
/// The decoder itself has no end-of-stream signal — it mirrors
/// Nayuki's reference implementation in treating past-EOF bits as
/// zeros.
pub struct ArithmeticDecoder<R: Read> {
    reader: BitReader<R>,
    low: u128,
    high: u128,
    code: u128,
}

impl<R: Read> ArithmeticDecoder<R> {
    /// Construct a new decoder reading from `source`.
    ///
    /// Reads the first `STATE_SIZE` bits of the stream into the
    /// initial code value.
    pub fn new(source: R) -> Result<Self> {
        let mut reader = BitReader::new(source);
        let mut code: u128 = 0;
        for _ in 0..STATE_SIZE {
            let bit = reader.read_bit().map_err(Error::Io)? as u128;
            code = (code << 1) | bit;
        }
        Ok(Self {
            reader,
            low: 0,
            high: MASK,
            code,
        })
    }

    /// Decode one symbol against a cumulative frequency table.
    ///
    /// The frequency table must be the same one the encoder used at
    /// the corresponding position. See
    /// [`ArithmeticEncoder::encode_symbol`] for the `cum_freqs`
    /// layout.
    pub fn decode_symbol(&mut self, cum_freqs: &[u64]) -> Result<u32> {
        let total = cum_freqs[cum_freqs.len() - 1] as u128;
        if total == 0 || total > MAX_TOTAL {
            return Err(Error::ZeroProbability { symbol: 0 });
        }
        let range = self.high - self.low + 1;
        // (code - low + 1) * total - 1) / range
        let offset = self.code - self.low;
        let value = ((offset + 1) * total - 1) / range;

        // Find the symbol whose cumulative range contains `value`.
        // cum_freqs is monotonically nondecreasing, so partition_point
        // gives the first index where cum_freqs[i] > value; the symbol
        // we want is one less. O(log V) instead of O(V) — the dominant
        // decompress cost at vocab=32K, where linear scan was ~16K
        // comparisons per token.
        let n = cum_freqs.len() - 1;
        let symbol = cum_freqs
            .partition_point(|&c| (c as u128) <= value)
            .saturating_sub(1) as u32;
        if symbol >= n as u32 {
            return Err(Error::ZeroProbability { symbol });
        }

        let sym_low = cum_freqs[symbol as usize] as u128;
        let sym_high = cum_freqs[symbol as usize + 1] as u128;

        // Update range the same way the encoder did.
        let new_high = self.low + (range * sym_high) / total - 1;
        let new_low = self.low + (range * sym_low) / total;
        self.low = new_low;
        self.high = new_high;

        // Shift matching high bits out of low/high/code.
        while ((self.low ^ self.high) & TOP_MASK) == 0 {
            self.low = (self.low << 1) & MASK;
            self.high = ((self.high << 1) & MASK) | 1;
            let bit = self.reader.read_bit().map_err(Error::Io)? as u128;
            self.code = ((self.code << 1) & MASK) | bit;
        }

        // Strip underflow bits.
        while (self.low & !self.high & SECOND_MASK) != 0 {
            self.low = (self.low << 1) & (MASK >> 1);
            self.high = ((self.high << 1) & (MASK >> 1)) | TOP_MASK | 1;
            let bit = self.reader.read_bit().map_err(Error::Io)? as u128;
            // After stripping the second-highest bit, the new top bit
            // gets the old high bit of `code` back with the new low
            // bit set from the stream.
            self.code = (self.code & TOP_MASK) | ((self.code << 1) & (MASK >> 1)) | bit;
        }

        Ok(symbol)
    }
}

// -------- tests -------- //

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a cumulative-frequency table from raw frequencies.
    fn cumulate(freqs: &[u64]) -> Vec<u64> {
        let mut cum = Vec::with_capacity(freqs.len() + 1);
        cum.push(0);
        let mut acc = 0u64;
        for &f in freqs {
            acc += f;
            cum.push(acc);
        }
        cum
    }

    #[test]
    fn uniform_binary_roundtrip() {
        // Two symbols with equal probability, 16 of them in a
        // deterministic pattern.
        let cum = cumulate(&[1, 1]);
        let input: Vec<u32> = vec![0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0];

        let mut encoded = Vec::new();
        {
            let mut enc = ArithmeticEncoder::new(&mut encoded);
            for &s in &input {
                enc.encode_symbol(&cum, s).unwrap();
            }
            enc.finish().unwrap();
        }

        let mut dec = ArithmeticDecoder::new(&encoded[..]).unwrap();
        let decoded: Vec<u32> = (0..input.len())
            .map(|_| dec.decode_symbol(&cum).unwrap())
            .collect();
        assert_eq!(decoded, input);
    }

    #[test]
    fn skewed_distribution_roundtrip() {
        // Symbol 0 with weight 95, symbol 1 with weight 5.
        // With 100 zeros and a few ones, the encoded stream should
        // be much shorter than 100 bits.
        let cum = cumulate(&[95, 5]);
        let input: Vec<u32> = (0..200).map(|i| if i % 17 == 0 { 1 } else { 0 }).collect();

        let mut encoded = Vec::new();
        {
            let mut enc = ArithmeticEncoder::new(&mut encoded);
            for &s in &input {
                enc.encode_symbol(&cum, s).unwrap();
            }
            enc.finish().unwrap();
        }
        // Sanity: the encoded stream is smaller than the raw bits.
        // 200 symbols at optimal rate ~0.3 bits each ≈ 60 bits ≈ 8 bytes
        assert!(
            encoded.len() < 30,
            "encoded is {} bytes, expected much smaller",
            encoded.len()
        );

        let mut dec = ArithmeticDecoder::new(&encoded[..]).unwrap();
        let decoded: Vec<u32> = (0..input.len())
            .map(|_| dec.decode_symbol(&cum).unwrap())
            .collect();
        assert_eq!(decoded, input);
    }

    #[test]
    fn larger_alphabet_roundtrip() {
        // 256-symbol alphabet with a non-trivial distribution.
        // This exercises the range-narrowing path harder.
        let mut freqs = vec![1u64; 256];
        // Make some symbols much more common
        for f in freqs.iter_mut().take(16) {
            *f = 100;
        }
        let cum = cumulate(&freqs);

        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);
        // Sample symbols proportionally to the frequency table
        let total: u64 = freqs.iter().sum();
        let sample_symbol = |rng: &mut StdRng| -> u32 {
            let r = rng.gen_range(0..total);
            let mut acc = 0u64;
            for (i, &f) in freqs.iter().enumerate() {
                acc += f;
                if r < acc {
                    return i as u32;
                }
            }
            unreachable!()
        };
        let input: Vec<u32> = (0..500).map(|_| sample_symbol(&mut rng)).collect();

        let mut encoded = Vec::new();
        {
            let mut enc = ArithmeticEncoder::new(&mut encoded);
            for &s in &input {
                enc.encode_symbol(&cum, s).unwrap();
            }
            enc.finish().unwrap();
        }

        let mut dec = ArithmeticDecoder::new(&encoded[..]).unwrap();
        let decoded: Vec<u32> = (0..input.len())
            .map(|_| dec.decode_symbol(&cum).unwrap())
            .collect();
        assert_eq!(decoded, input);
    }

    #[test]
    fn empty_stream_is_handled() {
        // Zero symbols encoded. finish() writes the terminal bit and
        // the decoder should have no symbols to decode (we don't
        // call decode_symbol at all).
        let mut encoded = Vec::new();
        {
            let enc = ArithmeticEncoder::new(&mut encoded);
            enc.finish().unwrap();
        }
        // Encoded should be at most a few bytes (just the terminal bit + padding).
        assert!(encoded.len() <= 9);
        // Decoder construction shouldn't panic.
        let _dec = ArithmeticDecoder::new(&encoded[..]).unwrap();
    }

    #[test]
    fn distribution_with_many_rare_symbols() {
        // Lots of symbols, one dominant, many very rare
        let mut freqs = vec![1u64; 100];
        freqs[50] = 10_000;
        let cum = cumulate(&freqs);

        let input: Vec<u32> = (0..1000)
            .map(|i| if i % 7 == 0 { (i % 100) as u32 } else { 50 })
            .collect();

        let mut encoded = Vec::new();
        {
            let mut enc = ArithmeticEncoder::new(&mut encoded);
            for &s in &input {
                enc.encode_symbol(&cum, s).unwrap();
            }
            enc.finish().unwrap();
        }

        let mut dec = ArithmeticDecoder::new(&encoded[..]).unwrap();
        let decoded: Vec<u32> = (0..input.len())
            .map(|_| dec.decode_symbol(&cum).unwrap())
            .collect();
        assert_eq!(decoded, input);
    }
}

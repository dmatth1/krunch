//! Bit-level input and output on top of byte streams.
//!
//! Arithmetic coding is naturally bit-level: each symbol emits a
//! variable number of bits that don't line up on byte boundaries.
//! This module provides `BitWriter` and `BitReader` that buffer
//! up 8 bits and flush as bytes, matching the behavior of
//! L3TC's Python reference implementation bit-for-bit.
//!
//! # Conventions
//!
//! - Bits are packed **MSB-first** within each byte. That is, the
//!   first bit written becomes bit 7 of the first byte, the second
//!   bit becomes bit 6, and so on. This matches Project Nayuki's
//!   reference arithmetic coder (which L3TC's Python uses).
//! - Writing is idempotent modulo `flush`: the writer holds a partial
//!   byte in its buffer until 8 bits arrive, then commits.
//! - `finish()` must be called to flush the trailing partial byte.
//!   The final byte is zero-padded on the right.
//! - Reading past the end of the stream returns 0 bits, which matches
//!   the Project Nayuki behavior of treating EOF as an infinite
//!   stream of zeros. This is a technique the arithmetic decoder
//!   uses to terminate cleanly without needing an explicit EOF flag.

use std::io::{self, Read, Write};

/// A writer that emits individual bits to an underlying byte sink.
///
/// Bits are buffered until a full byte is ready, then written. Call
/// [`BitWriter::finish`] (or drop the writer) to flush any trailing
/// partial byte.
pub struct BitWriter<W: Write> {
    inner: W,
    /// Current partial byte being built, in the low bits.
    buf: u8,
    /// Number of bits currently in `buf` (0..=7).
    n_bits: u8,
}

impl<W: Write> BitWriter<W> {
    /// Construct a new writer wrapping `inner`.
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            buf: 0,
            n_bits: 0,
        }
    }

    /// Write a single bit (`0` or `1`, encoded in the low bit of `bit`).
    ///
    /// Panics in debug builds if `bit > 1`; release builds silently
    /// mask with `& 1`.
    pub fn write_bit(&mut self, bit: u8) -> io::Result<()> {
        debug_assert!(bit <= 1, "bit must be 0 or 1, got {bit}");
        self.buf = (self.buf << 1) | (bit & 1);
        self.n_bits += 1;
        if self.n_bits == 8 {
            self.inner.write_all(&[self.buf])?;
            self.buf = 0;
            self.n_bits = 0;
        }
        Ok(())
    }

    /// Write the low `n_bits` bits of `value`, MSB-first.
    ///
    /// Convenience for writing small integers. `n_bits` must be in
    /// `0..=64`.
    pub fn write_bits(&mut self, value: u64, n_bits: u8) -> io::Result<()> {
        debug_assert!(n_bits <= 64, "n_bits must be 0..=64, got {n_bits}");
        for i in (0..n_bits).rev() {
            self.write_bit(((value >> i) & 1) as u8)?;
        }
        Ok(())
    }

    /// Flush any trailing partial byte, zero-padding on the right.
    ///
    /// After `finish`, the writer is done and should not be used
    /// again. The underlying writer is returned so the caller can
    /// use it for additional (byte-aligned) output if desired.
    pub fn finish(mut self) -> io::Result<W> {
        if self.n_bits > 0 {
            let padded = self.buf << (8 - self.n_bits);
            self.inner.write_all(&[padded])?;
            self.buf = 0;
            self.n_bits = 0;
        }
        self.inner.flush()?;
        Ok(self.inner)
    }

    /// Number of bits currently buffered (0..=7).
    pub fn buffered_bits(&self) -> u8 {
        self.n_bits
    }
}

/// A reader that pulls individual bits from an underlying byte source.
///
/// Reading past the end of the stream returns 0 bits (this matches
/// Project Nayuki's reference arithmetic coder convention).
pub struct BitReader<R: Read> {
    inner: R,
    /// Current byte being consumed, in the high bits.
    buf: u8,
    /// Number of bits remaining in `buf` (0..=8).
    n_bits: u8,
    /// Whether the underlying reader has returned EOF. Once true,
    /// all subsequent reads return zero bits.
    eof: bool,
}

impl<R: Read> BitReader<R> {
    /// Construct a new reader wrapping `inner`.
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            buf: 0,
            n_bits: 0,
            eof: false,
        }
    }

    /// Read a single bit as `0` or `1`.
    ///
    /// Returns `Ok(0)` if the underlying reader has reached EOF,
    /// matching the convention used by Nayuki's reference coder.
    pub fn read_bit(&mut self) -> io::Result<u8> {
        if self.n_bits == 0 {
            if self.eof {
                return Ok(0);
            }
            let mut byte = [0u8; 1];
            match self.inner.read(&mut byte)? {
                0 => {
                    self.eof = true;
                    return Ok(0);
                }
                1 => {
                    self.buf = byte[0];
                    self.n_bits = 8;
                }
                _ => unreachable!("read returned > 1 byte into a 1-byte buffer"),
            }
        }
        self.n_bits -= 1;
        let bit = (self.buf >> self.n_bits) & 1;
        Ok(bit)
    }

    /// Read the next `n_bits` bits and return them MSB-first in a
    /// `u64`. Returns zero bits past EOF.
    pub fn read_bits(&mut self, n_bits: u8) -> io::Result<u64> {
        debug_assert!(n_bits <= 64, "n_bits must be 0..=64, got {n_bits}");
        let mut out = 0u64;
        for _ in 0..n_bits {
            out = (out << 1) | (self.read_bit()? as u64);
        }
        Ok(out)
    }

    /// Whether the underlying reader has returned EOF. Bits may
    /// still be available in the internal buffer; use
    /// [`BitReader::buffered_bits`] to check.
    pub fn is_at_eof(&self) -> bool {
        self.eof && self.n_bits == 0
    }

    /// Number of bits remaining in the internal buffer (0..=8).
    pub fn buffered_bits(&self) -> u8 {
        self.n_bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_single_bits_msb_first() {
        let mut buf = Vec::new();
        {
            let mut w = BitWriter::new(&mut buf);
            // 10110010 -> 0xB2
            for bit in [1, 0, 1, 1, 0, 0, 1, 0] {
                w.write_bit(bit).unwrap();
            }
            w.finish().unwrap();
        }
        assert_eq!(buf, vec![0b1011_0010]);
    }

    #[test]
    fn writer_partial_byte_is_right_padded() {
        let mut buf = Vec::new();
        {
            let mut w = BitWriter::new(&mut buf);
            // 5 bits: 10110 -> 0xB0 (101100_00 padded on right)
            for bit in [1, 0, 1, 1, 0] {
                w.write_bit(bit).unwrap();
            }
            w.finish().unwrap();
        }
        assert_eq!(buf, vec![0b1011_0000]);
    }

    #[test]
    fn writer_multi_byte() {
        let mut buf = Vec::new();
        {
            let mut w = BitWriter::new(&mut buf);
            // 0xA5 then 0x3C (16 bits)
            w.write_bits(0xA5, 8).unwrap();
            w.write_bits(0x3C, 8).unwrap();
            w.finish().unwrap();
        }
        assert_eq!(buf, vec![0xA5, 0x3C]);
    }

    #[test]
    fn writer_mixed_bit_widths() {
        let mut buf = Vec::new();
        {
            let mut w = BitWriter::new(&mut buf);
            // 3 bits (101), 5 bits (10110), 2 bits (11), 6 bits (010101)
            // concatenated: 101 10110 11 010101 = 10110110 11010101 = 0xB6 0xD5
            w.write_bits(0b101, 3).unwrap();
            w.write_bits(0b10110, 5).unwrap();
            w.write_bits(0b11, 2).unwrap();
            w.write_bits(0b010101, 6).unwrap();
            w.finish().unwrap();
        }
        assert_eq!(buf, vec![0b10110110, 0b11010101]);
    }

    #[test]
    fn reader_single_bits_msb_first() {
        let data = [0b1011_0010u8];
        let mut r = BitReader::new(&data[..]);
        let bits: Vec<u8> = (0..8).map(|_| r.read_bit().unwrap()).collect();
        assert_eq!(bits, vec![1, 0, 1, 1, 0, 0, 1, 0]);
    }

    #[test]
    fn reader_past_eof_returns_zeros() {
        let data = [0b1111_1111u8];
        let mut r = BitReader::new(&data[..]);
        // Consume the 8 bits
        for _ in 0..8 {
            assert_eq!(r.read_bit().unwrap(), 1);
        }
        // Past EOF
        for _ in 0..10 {
            assert_eq!(r.read_bit().unwrap(), 0);
        }
        assert!(r.is_at_eof());
    }

    #[test]
    fn reader_multi_bit_read() {
        let data = [0xA5, 0x3C];
        let mut r = BitReader::new(&data[..]);
        assert_eq!(r.read_bits(8).unwrap(), 0xA5);
        assert_eq!(r.read_bits(8).unwrap(), 0x3C);
    }

    #[test]
    fn writer_reader_roundtrip_random() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        // Generate 1000 random (value, width) pairs where width in 1..=32
        let pairs: Vec<(u64, u8)> = (0..1000)
            .map(|_| {
                let width: u8 = rng.gen_range(1..=32);
                let value = rng.gen_range(0u64..(1u64 << width));
                (value, width)
            })
            .collect();

        let mut buf = Vec::new();
        {
            let mut w = BitWriter::new(&mut buf);
            for &(value, width) in &pairs {
                w.write_bits(value, width).unwrap();
            }
            w.finish().unwrap();
        }

        let mut r = BitReader::new(&buf[..]);
        for (i, &(expected, width)) in pairs.iter().enumerate() {
            let got = r.read_bits(width).unwrap();
            assert_eq!(
                got, expected,
                "mismatch at pair {i}: width={width} expected={expected} got={got}"
            );
        }
    }
}

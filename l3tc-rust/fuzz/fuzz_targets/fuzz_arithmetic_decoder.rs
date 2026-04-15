//! Fuzz the arithmetic decoder with arbitrary byte streams and
//! frequency tables.
//!
//! The AC decoder must never panic, OOM, or enter an infinite loop
//! regardless of what bytes it reads or what cumulative frequency
//! table it's given.

#![no_main]
use libfuzzer_sys::fuzz_target;

use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    if data.len() < 10 {
        return;
    }

    // Use first byte to determine vocab size (4..260)
    let vocab_size = (data[0] as usize).max(4).min(260);
    let rest = &data[1..];

    // Build a plausible cum_freqs table from the next `vocab_size` bytes
    let freq_bytes = vocab_size.min(rest.len());
    let mut cum_freqs = vec![0u64; vocab_size + 1];
    let mut total: u64 = 0;
    for i in 0..vocab_size {
        let freq = if i < freq_bytes {
            (rest[i] as u64).max(1)
        } else {
            1
        };
        total += freq;
        cum_freqs[i + 1] = total;
    }

    // Skip if total is 0 or exceeds AC limits
    if total == 0 || total > (1u64 << 62) {
        return;
    }

    let ac_data = &rest[freq_bytes..];
    if ac_data.is_empty() {
        return;
    }

    // Try to decode symbols — should never panic
    let cursor = Cursor::new(ac_data);
    let mut decoder = match l3tc::arithmetic::ArithmeticDecoder::new(cursor) {
        Ok(d) => d,
        Err(_) => return,
    };

    // Decode up to 1000 symbols (bounded to prevent timeouts)
    for _ in 0..1000 {
        match decoder.decode_symbol(&cum_freqs) {
            Ok(_) => {}
            Err(_) => break,
        }
    }
});

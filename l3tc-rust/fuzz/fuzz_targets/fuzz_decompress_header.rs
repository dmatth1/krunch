//! Fuzz the compressed-file header + segment metadata parsing.
//!
//! Feeds arbitrary bytes through the decompress_bytes path. The
//! decompressor will reject most inputs at the magic/CRC/version
//! check, but the fuzzer will discover inputs that pass those early
//! gates and exercise the segment metadata parser, varint decoder,
//! and allocation bounds checks.
//!
//! This harness does NOT require a model or tokenizer — it tests
//! the parsing layers before the RWKV forward pass is invoked.

#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // The simplest valuable fuzz: feed arbitrary bytes to the
    // checkpoint parser. This catches panics, OOM, and integer
    // overflow in the binary format reader.
    let _ = l3tc::Checkpoint::from_bytes(data);
});

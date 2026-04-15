//! Fuzz the checkpoint binary parser.
//!
//! The checkpoint format has a magic header, tensor metadata (names,
//! shapes, dtypes), and raw float data. Malformed checkpoints must
//! never cause panics, OOM, or undefined behavior — just clean
//! error returns.

#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Feed arbitrary bytes to the checkpoint parser.
    // Should return Err for malformed input, never panic.
    match l3tc::Checkpoint::from_bytes(data) {
        Ok(mut ckpt) => {
            // If parsing succeeded, try to load a model from it.
            // This exercises shape validation and tensor extraction.
            let _ = l3tc::Model::from_checkpoint(&mut ckpt);
        }
        Err(_) => {}
    }
});

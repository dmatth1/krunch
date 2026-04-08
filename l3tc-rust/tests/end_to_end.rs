//! End-to-end round trip test: compress and decompress real text
//! using the real L3TC-200K model, asserting byte-exact recovery.
//!
//! This is the single most important test in Phase 1. If it passes,
//! every piece of the Rust pipeline — checkpoint loading, tokenizer,
//! RWKV forward pass, arithmetic coder, file format — is wired up
//! correctly and agrees with itself.
//!
//! Run with:
//!     cargo test --release --test end_to_end -- --ignored --nocapture

use l3tc::{compress, decompress, Checkpoint, Model, Tokenizer, DEFAULT_SEGMENT_BYTES};
use std::path::PathBuf;
use std::time::Instant;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn checkpoint_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("checkpoints/l3tc_200k.bin")
}

fn spm_path() -> PathBuf {
    repo_root()
        .join("vendor/L3TC/dictionary/vocab_enwik8_bpe_16384_0.999/spm_enwik8_bpe_16384_0.999.model")
}

fn assert_exists(p: &PathBuf, what: &str) {
    assert!(p.exists(), "{what} not found at {p:?}; run Phase 0 setup");
}

fn load_model_and_tokenizer() -> (Model, Tokenizer) {
    let ckpt_path = checkpoint_path();
    let tok_path = spm_path();
    assert_exists(&ckpt_path, "converted checkpoint");
    assert_exists(&tok_path, "SPM tokenizer model");

    let mut ckpt = Checkpoint::load(&ckpt_path).expect("load checkpoint");
    let model = Model::from_checkpoint(&mut ckpt).expect("build model");
    let tokenizer = Tokenizer::load(&tok_path).expect("load tokenizer");
    (model, tokenizer)
}

#[test]
#[ignore = "integration — needs real checkpoint and SPM model"]
fn roundtrip_short_ascii() {
    let (model, tok) = load_model_and_tokenizer();
    let text = "The quick brown fox jumps over the lazy dog.\n";

    let compressed = compress(text, &tok, &model, DEFAULT_SEGMENT_BYTES).unwrap();
    let decompressed = decompress(&compressed, &tok, &model).unwrap();

    assert_eq!(decompressed, text, "short ASCII round trip failed");
    println!(
        "  short ASCII: {} bytes -> {} bytes (ratio {:.3})",
        text.len(),
        compressed.len(),
        compressed.len() as f64 / text.len() as f64
    );
}

#[test]
#[ignore = "integration — needs real checkpoint and SPM model"]
fn roundtrip_multi_segment_ascii() {
    let (model, tok) = load_model_and_tokenizer();
    // Force multiple segments by using a very small segment size
    let text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.\n".repeat(5);

    let compressed = compress(&text, &tok, &model, 128).unwrap();
    let decompressed = decompress(&compressed, &tok, &model).unwrap();

    assert_eq!(decompressed, text, "multi-segment ASCII round trip failed");
    println!(
        "  multi-segment: {} bytes -> {} bytes (ratio {:.3})",
        text.len(),
        compressed.len(),
        compressed.len() as f64 / text.len() as f64
    );
}

#[test]
#[ignore = "integration — needs real checkpoint and SPM model"]
fn roundtrip_enwik6_subset() {
    let (model, tok) = load_model_and_tokenizer();
    let enwik6_path = repo_root().join("bench/corpora/enwik6");
    assert_exists(&enwik6_path, "enwik6 corpus");

    // Use the first ~10 KB for a quick round-trip test (full enwik6
    // is 1 MB and would take a minute; this test is about
    // correctness, not throughput).
    let full = std::fs::read_to_string(&enwik6_path).expect("read enwik6");
    let text = &full[..full.len().min(10_000)];

    let start = Instant::now();
    let compressed = compress(text, &tok, &model, DEFAULT_SEGMENT_BYTES).unwrap();
    let compress_dt = start.elapsed();

    let start = Instant::now();
    let decompressed = decompress(&compressed, &tok, &model).unwrap();
    let decompress_dt = start.elapsed();

    assert_eq!(decompressed, text, "enwik6 subset round trip failed");

    let ratio = compressed.len() as f64 / text.len() as f64;
    let compress_kb_s = (text.len() as f64 / 1024.0) / compress_dt.as_secs_f64();
    let decompress_kb_s = (text.len() as f64 / 1024.0) / decompress_dt.as_secs_f64();

    println!(
        "  enwik6[:10K]: {} bytes -> {} bytes (ratio {:.4})",
        text.len(),
        compressed.len(),
        ratio,
    );
    println!("    compress:   {:?}  ({:.1} KB/s)", compress_dt, compress_kb_s);
    println!("    decompress: {:?}  ({:.1} KB/s)", decompress_dt, decompress_kb_s);
}

#[test]
#[ignore = "integration — long-running, full enwik6"]
fn roundtrip_full_enwik6_with_timing() {
    let (model, tok) = load_model_and_tokenizer();
    let enwik6_path = repo_root().join("bench/corpora/enwik6");
    assert_exists(&enwik6_path, "enwik6 corpus");

    let text = std::fs::read_to_string(&enwik6_path).expect("read enwik6");
    let text_len = text.len();
    println!("  enwik6: {} bytes", text_len);

    let start = Instant::now();
    let compressed = compress(&text, &tok, &model, DEFAULT_SEGMENT_BYTES).unwrap();
    let compress_dt = start.elapsed();

    let start = Instant::now();
    let decompressed = decompress(&compressed, &tok, &model).unwrap();
    let decompress_dt = start.elapsed();

    assert_eq!(decompressed.len(), text_len, "length mismatch");
    assert_eq!(decompressed, text, "round-trip failed");

    let ratio = compressed.len() as f64 / text_len as f64;
    let compress_kb_s = (text_len as f64 / 1024.0) / compress_dt.as_secs_f64();
    let decompress_kb_s = (text_len as f64 / 1024.0) / decompress_dt.as_secs_f64();

    println!("  compressed size: {} bytes", compressed.len());
    println!("  ratio: {:.4}", ratio);
    println!(
        "  compress wall: {:?}  = {:.2} KB/s",
        compress_dt, compress_kb_s
    );
    println!(
        "  decompress wall: {:?} = {:.2} KB/s",
        decompress_dt, decompress_kb_s
    );
    println!(
        "\n  Python reference on same corpus: L3TC-200K = 13.24 KB/s (ratio 0.1665)"
    );
    println!(
        "  Speedup factor: {:.2}x",
        compress_kb_s / 13.24
    );
}

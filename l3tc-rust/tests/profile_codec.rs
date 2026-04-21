//! Profile the codec to find where time goes on a small enwik6 chunk.

use l3tc::arithmetic::ArithmeticEncoder;
use l3tc::{Checkpoint, Model, Session, Tokenizer, DEFAULT_SEGMENT_BYTES};
use std::path::PathBuf;
use std::time::Instant;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

#[test]
#[ignore]
fn where_does_time_go() {
    let ckpt_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("checkpoints/l3tc_200k.bin");
    let tok_path = repo_root().join(
        "vendor/L3TC/dictionary/vocab_enwik8_bpe_16384_0.999/spm_enwik8_bpe_16384_0.999.model",
    );
    let enwik6 = repo_root().join("bench/corpora/enwik6");

    let mut ckpt = Checkpoint::load(&ckpt_path).unwrap();
    let model = Model::from_checkpoint(&mut ckpt).unwrap();
    let tok = Tokenizer::load(&tok_path).unwrap();
    let text = std::fs::read_to_string(&enwik6).unwrap();
    let text = &text[..10_000];

    // Phase 1: tokenization
    let start = Instant::now();
    let segments = tok.encode_file(text, DEFAULT_SEGMENT_BYTES).unwrap();
    let tok_dt = start.elapsed();
    let total_tokens: usize = segments.iter().map(|s| s.tokens.len()).sum();
    println!(
        "tokenize:    {:?}  ({} segments, {} tokens)",
        tok_dt,
        segments.len(),
        total_tokens
    );

    // Phase 2: pure forward passes through the model (no AC, no cum_freqs)
    let mut session = Session::new(&model);
    let start = Instant::now();
    for seg in &segments {
        session.reset();
        for &t in &seg.tokens {
            let _ = session.forward(t);
        }
    }
    let fwd_dt = start.elapsed();
    println!(
        "forward:     {:?}  ({:.1} us/token, {:.0} tok/s)",
        fwd_dt,
        fwd_dt.as_secs_f64() * 1e6 / total_tokens as f64,
        total_tokens as f64 / fwd_dt.as_secs_f64()
    );

    // Phase 3: forward + logits_to_cum_freqs (no AC)
    let mut session = Session::new(&model);
    let mut cum = vec![0u64; model.vocab_size + 1];
    let start = Instant::now();
    for seg in &segments {
        session.reset();
        for &t in &seg.tokens {
            let logits = session.forward(t);
            l3tc::codec::logits_to_cum_freqs_public(logits, &mut cum);
        }
    }
    let fwd_cum_dt = start.elapsed();
    println!(
        "fwd+cumfreq: {:?}  ({:.1} us/token)",
        fwd_cum_dt,
        fwd_cum_dt.as_secs_f64() * 1e6 / total_tokens as f64,
    );
    let overhead_cumfreq = fwd_cum_dt
        .checked_sub(fwd_dt)
        .unwrap_or_default()
        .as_secs_f64()
        * 1e6
        / total_tokens as f64;
    println!("  cum_freqs: {:.1} us/token", overhead_cumfreq);

    // Phase 4: full compress (forward + cum + AC)
    let mut session = Session::new(&model);
    let start = Instant::now();
    let mut total_ac_bytes = 0usize;
    for seg in &segments {
        session.reset();
        let mut ac_out = Vec::with_capacity(seg.tokens.len());
        {
            let mut enc = ArithmeticEncoder::new(&mut ac_out);
            session.forward(seg.tokens[0]);
            let mut cum_buf = vec![0u64; model.vocab_size + 1];
            let mut prev = seg.tokens[0];
            for &tok in &seg.tokens[1..] {
                let logits = session.forward(prev);
                l3tc::codec::logits_to_cum_freqs_public(logits, &mut cum_buf);
                enc.encode_symbol(&cum_buf, tok).unwrap();
                prev = tok;
            }
            enc.finish().unwrap();
        }
        total_ac_bytes += ac_out.len();
    }
    let full_dt = start.elapsed();
    println!(
        "full:        {:?}  ({:.1} us/token) = {:.1} KB/s",
        full_dt,
        full_dt.as_secs_f64() * 1e6 / total_tokens as f64,
        (text.len() as f64 / 1024.0) / full_dt.as_secs_f64()
    );
    let overhead_ac = full_dt
        .checked_sub(fwd_cum_dt)
        .unwrap_or_default()
        .as_secs_f64()
        * 1e6
        / total_tokens as f64;
    println!("  ac encode: {:.1} us/token", overhead_ac);
    println!("  ratio: {:.4}", total_ac_bytes as f64 / text.len() as f64);
}

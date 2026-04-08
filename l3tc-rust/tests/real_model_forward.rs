//! Integration test: load the real L3TC-200K model and run a forward pass.
//!
//! This test loads the converted binary and feeds a few token IDs
//! through the model, sanity-checking that the output logits are
//! finite, that they change between tokens, and that the same input
//! sequence produces the same output sequence (determinism).
//!
//! Run with:
//!
//!     cargo test --test real_model_forward -- --ignored --nocapture

use l3tc::{Checkpoint, Model, Session};
use std::path::PathBuf;

fn checkpoint_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("checkpoints/l3tc_200k.bin")
}

#[test]
#[ignore = "requires converted L3TC-200K checkpoint on disk"]
fn forward_pass_is_finite_and_deterministic() {
    let path = checkpoint_path();
    assert!(
        path.exists(),
        "expected converted checkpoint at {path:?}; run scripts/convert_checkpoint.py first"
    );

    let mut ckpt = Checkpoint::load(&path).expect("failed to load checkpoint");
    let model = Model::from_checkpoint(&mut ckpt).expect("failed to build model");

    println!("model: vocab={}, hidden={}, layers={}",
        model.vocab_size, model.hidden_size, model.num_layers());
    assert_eq!(model.vocab_size, 16384);
    assert_eq!(model.hidden_size, 96);
    assert_eq!(model.num_layers(), 2);

    let mut s1 = Session::new(&model);
    let mut s2 = Session::new(&model);

    // Feed the same input sequence through both sessions
    let tokens = [2u32, 100, 200, 500, 1000, 2000, 5000];

    let mut last_sum = 0.0f32;
    for (i, &t) in tokens.iter().enumerate() {
        let logits1 = s1.forward(t).to_vec();
        let logits2 = s2.forward(t).to_vec();

        // Determinism: both sessions give identical output
        assert_eq!(logits1, logits2, "non-deterministic at step {i}");

        // Finite
        for &v in &logits1 {
            assert!(v.is_finite(), "non-finite logit at step {i}: {v}");
        }

        // Not all the same value
        let min = logits1.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = logits1.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max > min, "logits at step {i} are all the same value");

        // The logits change meaningfully from step to step (state is flowing)
        let sum: f32 = logits1.iter().sum();
        if i > 0 {
            assert!(
                (sum - last_sum).abs() > 1e-9,
                "logits identical at step {i}: state not updating"
            );
        }
        last_sum = sum;

        println!(
            "  step {i} token={t}: min={min:.3} max={max:.3} sum={sum:.3}"
        );
    }
}

#[test]
#[ignore = "requires converted L3TC-200K checkpoint on disk; long-running"]
fn forward_pass_throughput_on_1k_tokens() {
    use std::time::Instant;

    let path = checkpoint_path();
    let mut ckpt = Checkpoint::load(&path).expect("failed to load checkpoint");
    let model = Model::from_checkpoint(&mut ckpt).expect("failed to build model");

    let mut s = Session::new(&model);

    // Run 1000 forward passes with a pseudo-random (but deterministic)
    // token sequence
    let n = 1000;
    let start = Instant::now();
    let mut last_max = f32::NEG_INFINITY;
    for i in 0..n {
        let token = (i * 2654435761u32) % model.vocab_size as u32;
        let logits = s.forward(token);
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        last_max = m;
    }
    let dt = start.elapsed();
    println!(
        "\n  {} forward passes in {:?} = {:.2} tokens/s = {:.2} us/token (last_max={last_max})",
        n,
        dt,
        n as f64 / dt.as_secs_f64(),
        dt.as_secs_f64() * 1e6 / n as f64,
    );
}

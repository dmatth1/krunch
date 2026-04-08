//! Micro-profile of the forward pass.
//!
//! Measures individual operations in isolation so we can pinpoint
//! the bottleneck. Run with:
//!
//!     cargo test --release --test profile_forward -- --ignored --nocapture

use l3tc::tensor;
use std::time::Instant;

fn time<F: FnMut()>(name: &str, n: usize, mut f: F) {
    // Warm up
    for _ in 0..10 {
        f();
    }
    let start = Instant::now();
    for _ in 0..n {
        f();
    }
    let dt = start.elapsed();
    let per = dt.as_secs_f64() * 1e9 / n as f64;
    println!(
        "  {:<40}  {:>10.1} ns/call  ({:>10.2} M ops/s)",
        name,
        per,
        1e9 / per / 1e6
    );
}

#[test]
#[ignore = "micro benchmark"]
fn profile_matvec_sizes() {
    println!("\n=== matvec at representative shapes ===");

    // Small: 96x96 (the per-block projections)
    let mat96 = vec![0.1f32; 96 * 96];
    let x96 = vec![0.5f32; 96];
    let mut out96 = vec![0.0f32; 96];
    time("matvec 96x96", 100_000, || {
        tensor::matvec(&mat96, &x96, &mut out96);
    });

    // Medium: 256x256 (if we had a 3.2M-style model)
    let mat256 = vec![0.1f32; 256 * 256];
    let x256 = vec![0.5f32; 256];
    let mut out256 = vec![0.0f32; 256];
    time("matvec 256x256", 20_000, || {
        tensor::matvec(&mat256, &x256, &mut out256);
    });

    // Large: 16384x96 (the head projection)
    let mat_head = vec![0.01f32; 16384 * 96];
    let x_head = vec![0.5f32; 96];
    let mut out_head = vec![0.0f32; 16384];
    time("matvec 16384x96 (head)", 2_000, || {
        tensor::matvec(&mat_head, &x_head, &mut out_head);
    });
}

#[test]
#[ignore = "micro benchmark"]
fn profile_layer_norm() {
    println!("\n=== layer_norm at h=96 ===");

    let x = vec![0.1f32; 96];
    let w = vec![1.0f32; 96];
    let b = vec![0.0f32; 96];
    let mut out = vec![0.0f32; 96];
    time("layer_norm(96)", 100_000, || {
        tensor::layer_norm(&x, &w, &b, 1e-5, &mut out);
    });
}

#[test]
#[ignore = "micro benchmark"]
fn profile_full_forward_breakdown() {
    use l3tc::{Checkpoint, Model, Session};
    use std::path::PathBuf;

    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("checkpoints/l3tc_200k.bin");
    if !path.exists() {
        eprintln!("skipping: no checkpoint at {path:?}");
        return;
    }
    let mut ckpt = Checkpoint::load(&path).expect("load");
    let model = Model::from_checkpoint(&mut ckpt).expect("build");
    let mut s = Session::new(&model);

    // How long does ONE forward pass take?
    let n = 10_000;

    // Warm up
    for i in 0..100 {
        s.forward((i * 97) as u32 % 16384);
    }

    let start = Instant::now();
    for i in 0..n {
        s.forward((i * 2654435761u32) % 16384);
    }
    let dt = start.elapsed();
    let per = dt.as_secs_f64() * 1e6 / n as f64;
    println!(
        "\nfull forward pass: {n} calls in {:?} = {:.1} us/call ({:.0} tokens/s)",
        dt,
        per,
        1.0 / (per * 1e-6)
    );

    // Now disable the scratch buffer session overhead by doing just
    // the head matmul in isolation at the same shape
    let head_mat = vec![0.01f32; model.vocab_size * model.hidden_size];
    let x = vec![0.5f32; model.hidden_size];
    let mut out = vec![0.0f32; model.vocab_size];
    let start = Instant::now();
    for _ in 0..n {
        tensor::matvec(&head_mat, &x, &mut out);
    }
    let dt = start.elapsed();
    println!(
        "head matvec alone: {n} calls in {:?} = {:.1} us/call",
        dt,
        dt.as_secs_f64() * 1e6 / n as f64
    );
}

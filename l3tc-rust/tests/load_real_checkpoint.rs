//! Integration test: load the real converted L3TC-200K checkpoint.
//!
//! This test is ignored by default because it depends on the
//! converted binary having been generated via
//! `scripts/convert_checkpoint.py` from an L3TC `.pth` file. To run:
//!
//!     cargo test --test load_real_checkpoint -- --ignored
//!
//! Fails gracefully with a clear message if the checkpoint file
//! hasn't been generated.

use l3tc::Checkpoint;
use std::path::PathBuf;

fn checkpoint_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("checkpoints/l3tc_200k.bin")
}

#[test]
#[ignore = "requires converted L3TC-200K checkpoint on disk"]
fn load_200k_checkpoint() {
    let path = checkpoint_path();
    assert!(
        path.exists(),
        "expected converted checkpoint at {path:?}; run scripts/convert_checkpoint.py first"
    );

    let mut ckpt = Checkpoint::load(&path).expect("failed to load checkpoint");

    println!("loaded {} tensors", ckpt.len());

    // The converter produces 46 tensors for the 200K model:
    // 2 embedding tables (emb, head) + 2 top-level LN pairs
    // (ln0, ln_out) + 20 per-block tensors × 2 blocks.
    // = 2 + 4 + 40 = 46. See docs/phase_0_findings.md for the
    // full tensor inventory.
    assert_eq!(ckpt.len(), 46);

    // Spot-check a few important tensors by shape. These shapes
    // are determined by the config and are how we validate that
    // the HiRA merge and squeeze steps produced the expected output.
    let emb = ckpt.take_shape("emb.weight", &[16384, 96]).unwrap();
    assert_eq!(emb.data.len(), 16384 * 96);

    let head = ckpt.take_shape("head.weight", &[16384, 96]).unwrap();
    assert_eq!(head.data.len(), 16384 * 96);

    let ln0 = ckpt.take_shape("ln0.weight", &[96]).unwrap();
    assert_eq!(ln0.data.len(), 96);

    let att_key_0 = ckpt
        .take_shape("blocks.0.att.key.weight", &[96, 96])
        .unwrap();
    assert_eq!(att_key_0.data.len(), 96 * 96);

    let tmk = ckpt.take_shape("blocks.0.att.time_mix_k", &[96]).unwrap();
    // time_mix was (1, 1, 96) in the .pth, squeezed to (96,) here.
    assert_eq!(tmk.data.len(), 96);

    // Sanity: the tensor values are real (not all zeros, not all NaN)
    let sum: f32 = emb.data.iter().sum();
    assert!(sum.is_finite(), "emb has NaN or inf");
    assert!(
        sum.abs() > 0.0,
        "emb is all zeros — HiRA merge broke something?"
    );

    println!("✓ all spot checks passed");
}

//! Build script.
//!
//! When `--features=cuda` is active, compile `src/cuda/wkv.cu` to PTX
//! with nvcc and drop it in `$OUT_DIR/wkv.ptx`. The backend module
//! `include_bytes!`'s this file into the binary and loads it via
//! cudarc's NVRTC at runtime.
//!
//! When `--features=cuda` is NOT active, this script is effectively
//! a no-op — we still emit a zero-byte `wkv.ptx` so the macOS-default
//! build (which doesn't enable `cuda`) still produces the same
//! `include_bytes!` target path, avoiding compile errors from a
//! missing file. The CUDA backend module itself is `#[cfg(feature =
//! "cuda")]`-gated so the empty PTX is never actually loaded.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/cuda/wkv.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDACXX");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let ptx_path = out_dir.join("wkv.ptx");

    let cuda_feature = env::var("CARGO_FEATURE_CUDA").is_ok();

    if !cuda_feature {
        // No CUDA feature — emit an empty placeholder so the
        // include_bytes! target exists. The Rust code that would
        // use this is cfg-gated and never runs.
        std::fs::write(&ptx_path, b"").expect("write placeholder wkv.ptx");
        return;
    }

    // Find nvcc. Priority: $CUDACXX > $CUDA_HOME/bin/nvcc > PATH.
    let nvcc = if let Ok(p) = env::var("CUDACXX") {
        PathBuf::from(p)
    } else if let Ok(home) = env::var("CUDA_HOME") {
        PathBuf::from(home).join("bin/nvcc")
    } else {
        PathBuf::from("nvcc")  // rely on PATH
    };

    // SM target: Ampere (sm_80) covers A10G / A100; Ada (sm_89)
    // covers RTX 4090. PTX-only output means the driver JIT-compiles
    // to the exact target at load time, so a single PTX works on any
    // compute-capability ≥ 7.0.
    //
    // `--maxrregcount 60` is from BlinkDL's original — prevents
    // register spilling on older GPUs for the WKV kernel. Harmless
    // on A10G.
    let nvcc_output = Command::new(&nvcc)
        .args([
            "-ptx",
            "-O3",
            "--use_fast_math",
            "-Xptxas", "-O3",
            "--maxrregcount=60",
            "-arch=compute_80",
            "src/cuda/wkv.cu",
            "-o",
        ])
        .arg(&ptx_path)
        .output();

    match nvcc_output {
        Ok(out) if out.status.success() => {
            println!("cargo:warning=wkv.ptx compiled to {}", ptx_path.display());
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            panic!(
                "nvcc failed (cuda feature enabled). Is CUDA toolkit installed and nvcc on PATH?\n\
                 Set CUDA_HOME or CUDACXX to point at your install.\n\
                 nvcc stderr:\n{}",
                stderr
            );
        }
        Err(e) => {
            panic!(
                "Could not invoke nvcc ({}). The `cuda` feature requires the CUDA toolkit.\n\
                 Install cuda-toolkit (Ubuntu: apt install nvidia-cuda-toolkit; or from NVIDIA) \
                 and make sure nvcc is on PATH, or set CUDA_HOME=/usr/local/cuda, or \
                 CUDACXX=/path/to/nvcc.\n\
                 Underlying error: {}",
                nvcc.display(),
                e
            );
        }
    }
}

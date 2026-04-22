//! l3tc — command-line interface.
//!
//! Wires the library's `compress` / `decompress` together with a
//! file-based CLI. Uses clap for argument parsing and anyhow for
//! error reporting.
//!
//! The CLI is deliberately minimal for Phase 1: it reads the full
//! input into memory, runs compress or decompress end-to-end, and
//! writes the output. Streaming and stdin/stdout support come in
//! Phase 3.
//!
//! Defaults are tuned for fast iteration during development:
//!
//! - `--model` defaults to `checkpoints/l3tc_200k.bin` relative to
//!   the binary. The converter script
//!   (`scripts/convert_checkpoint.py`) drops the file here.
//! - `--tokenizer` defaults to the SPM model that lives inside
//!   `vendor/L3TC/dictionary/` (i.e. the one `scripts/setup.sh`
//!   produced).
//! - `--segment-bytes` defaults to 2048 to match L3TC's segment
//!   boundaries.
//!
//! With the defaults in place, the common iteration loop is:
//!
//! ```text
//! cargo build --release
//! ./target/release/l3tc compress bench/corpora/enwik6 -o /tmp/a.l3tc --time
//! ./target/release/l3tc decompress /tmp/a.l3tc -o /tmp/b.txt --time
//! diff bench/corpora/enwik6 /tmp/b.txt
//! ```

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use l3tc::{
    audit_compress, decode_writer, decompress_bytes, dump_teacher, encode_reader, hybrid_decode,
    hybrid_encode, profile_compress, AuditStats, Bzip3Codec, Checkpoint, ClpStub, Codec,
    DispatchStats, Lz4Codec, Model, NeuralCodec, PassthroughCodec, ProfileStats, Tokenizer,
    Zstd22Codec, ZstdDictCodec, DEFAULT_CHUNK_SIZE, DEFAULT_SEGMENT_BYTES,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

// Phase 14 modules: live next to the CLI because their semantics
// (which model file maps to which input domain) are a UI concern.
// The library only knows about a `model_id: u8` byte in the header.
mod install_models;
mod registry;
mod specialist;

use registry::ResolvedSpecialist;
use specialist::{detect, Specialist};

/// l3tc — learned lossless text compression.
#[derive(Parser, Debug)]
#[command(name = "l3tc", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Compress a text file.
    Compress {
        /// Input file to compress.
        input: PathBuf,
        /// Output path. Defaults to `<input>.l3tc`.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Explicit path to a converted L3TC model binary. When
        /// omitted, the CLI auto-detects the right specialist from
        /// the input content and loads its registered model.
        #[arg(long)]
        model: Option<PathBuf>,
        /// Explicit path to a SentencePiece tokenizer model. Paired
        /// with `--model`; both should be from the same specialist.
        /// When omitted, resolved from the registry.
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Specialist to use: `auto` (default) runs content
        /// detection; `prose`, `code`, `structured`, `logs`,
        /// `tabular`, `markup`, `fallback` force a specific model.
        #[arg(long, default_value = "auto")]
        specialist: String,
        /// Print a one-line note about which specialist was picked
        /// and why. Useful when a file gets routed unexpectedly.
        #[arg(short, long)]
        verbose: bool,
        /// Segment length in bytes. Matches L3TC's segment boundaries
        /// and resets model state at each boundary.
        #[arg(long, default_value_t = DEFAULT_SEGMENT_BYTES)]
        segment_bytes: usize,
        /// Print per-phase timing information.
        #[arg(long)]
        time: bool,
        /// Round-trip verify after compressing: decompress the
        /// output and compare to the original. Useful for local
        /// testing. Slows the run by ~2x because it does the work
        /// twice.
        #[arg(long)]
        verify: bool,
        /// Backend for the forward pass. `cpu` (default) uses Phase
        /// 12 NEON; `metal` uses the Phase 13 GPU backend (only on
        /// builds with `--features=metal`). Files compressed with
        /// `metal` MUST be decompressed with `metal` (the FP
        /// arithmetic differs by a few ULPs across backends and
        /// would desync the AC).
        #[arg(long, default_value = "cpu")]
        backend: String,
        /// Metal-only: number of segments processed in GPU lockstep
        /// per BatchedSession chunk. Higher = more amortization of
        /// per-token dispatch overhead, until GPU occupancy saturates
        /// or you run out of segments. Ignored for `--backend=cpu`.
        /// Default 256 is the post-Phase-13n amortization knee on a
        /// 1 MB enwik6 reference (~250 segments populate 256 lanes
        /// well; higher values leave idle lanes wasting GPU time).
        /// Small inputs with fewer segments should sweep downward
        /// (e.g. 32 for 50 KB / ~13 segments).
        #[arg(long, default_value_t = 256)]
        metal_batch: usize,
        /// Metal-only: number of parallel BatchedSession workers.
        /// Each worker has its own Metal command queue; the segment
        /// list is split across workers. On Apple M-series with
        /// multiple GPU cores this lets the scheduler run independent
        /// compute streams in parallel. Default 1 (serial).
        #[arg(long, default_value_t = 1)]
        metal_workers: usize,
    },
    /// Decompress a file produced by `compress`.
    Decompress {
        /// Input file to decompress.
        input: PathBuf,
        /// Output path. Defaults to `<input>.out`.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Explicit path to a converted L3TC model binary. When
        /// omitted, the CLI reads the specialist id from the file
        /// header and loads the matching registered model.
        #[arg(long)]
        model: Option<PathBuf>,
        /// Explicit path to a SentencePiece tokenizer model. Paired
        /// with `--model`. When omitted, resolved from the registry.
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Print a one-line note about which specialist was loaded.
        #[arg(short, long)]
        verbose: bool,
        /// Print per-phase timing information.
        #[arg(long)]
        time: bool,
        /// Backend hint. Default `auto` reads the file header's
        /// FLAG_GPU_ENCODED bit and picks the matching backend.
        /// Use `cpu` or `metal` to force a specific backend.
        #[arg(long, default_value = "auto")]
        backend: String,
    },
    /// Compress a file with the hybrid codec dispatcher.
    ///
    /// Splits the input into fixed-size chunks, probes every enabled
    /// codec on each, and writes a tagged blob with per-chunk codec
    /// selection. Reads `--model` + `--tokenizer` if supplied (neural
    /// path); reads `--zstd-dict` if supplied (dictionary-backed zstd
    /// path). With neither, runs classical-only (zstd, bzip3, lz4,
    /// passthrough).
    ///
    /// Prints per-run stats (bytes in/out, per-codec breakdown,
    /// savings vs zstd shadow, throughput, safety-net count) to
    /// stderr on completion. With `--stats <path>`, also writes the
    /// same stats as JSON to that path — consumed by the service-
    /// side compression_worker.py to emit CloudWatch EMF metrics.
    HybridCompress {
        /// Input file to compress.
        input: PathBuf,
        /// Output path for the hybrid blob. Defaults to `<input>.l3h`.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Optional path to a per-dataset RWKV model binary. Pair
        /// with `--tokenizer`. When supplied the dispatcher includes
        /// the neural codec in the probe menu.
        #[arg(long)]
        model: Option<PathBuf>,
        /// Optional path to the SentencePiece tokenizer that matches
        /// `--model`. Must be supplied iff `--model` is.
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Optional path to a zstd dictionary trained on the target
        /// dataset (e.g. the `.zstd_dict` artifact emitted by the
        /// training job). Enables the `ZstdDict` codec in the probe
        /// menu.
        #[arg(long)]
        zstd_dict: Option<PathBuf>,
        /// Uncompressed bytes per chunk. Smaller = finer-grained
        /// per-chunk codec selection; larger = more room for zstd's
        /// sliding window. Default matches `DEFAULT_CHUNK_SIZE`.
        #[arg(long, default_value_t = DEFAULT_CHUNK_SIZE)]
        chunk_size: usize,
        /// Write per-run stats as JSON to this path. When omitted,
        /// stats are printed to stderr only (still available via
        /// tee / redirect).
        #[arg(long)]
        stats: Option<PathBuf>,
    },

    /// Decompress a hybrid blob produced by `hybrid-compress`.
    ///
    /// The blob carries a codec tag per chunk; the decoder dispatches
    /// each chunk to its tag's codec. `--model` / `--tokenizer` /
    /// `--zstd-dict` must be supplied iff the blob used that codec
    /// (e.g. skip `--model` on a classical-only blob). Decoding a
    /// chunk whose codec isn't in the registry returns a clear
    /// error rather than silently producing bad output.
    HybridDecompress {
        /// Input hybrid blob.
        input: PathBuf,
        /// Output path. Defaults to `<input>` with `.l3h` stripped.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Optional path to the RWKV model used during
        /// hybrid-compress. Required if the blob includes any
        /// neural-tagged chunks.
        #[arg(long)]
        model: Option<PathBuf>,
        /// Optional path to the tokenizer paired with `--model`.
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Optional path to the zstd dictionary used during
        /// hybrid-compress. Required if the blob includes any
        /// `ZstdDict`-tagged chunks.
        #[arg(long)]
        zstd_dict: Option<PathBuf>,
    },

    /// Phase 4a debug: dump per-token logits to a binary file for
    /// numerical diff against the Python L3TC reference. Reads the
    /// first `--max-tokens` of `--input` (after tokenizing the
    /// first `--segment-bytes`) and writes a flat binary in the
    /// same format as `scripts/dump_python_logits.py`.
    DumpLogits {
        /// Input text file to tokenize.
        #[arg(long)]
        input: PathBuf,
        /// Path to the converted L3TC model binary.
        #[arg(long, default_value = "checkpoints/l3tc_200k.bin")]
        model: PathBuf,
        /// Path to the SentencePiece tokenizer model.
        #[arg(
            long,
            default_value = "../vendor/L3TC/dictionary/vocab_enwik8_bpe_16384_0.999/spm_enwik8_bpe_16384_0.999.model"
        )]
        tokenizer: PathBuf,
        /// Bytes of input to tokenize before truncating to
        /// `--max-tokens`. Should match the Python harness so
        /// both implementations see the same input.
        #[arg(long, default_value_t = 4096)]
        segment_bytes: usize,
        /// Maximum number of tokens (including BOS) to forward.
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        /// Output directory for `tokens.bin` and `logits.bin`.
        #[arg(long, default_value = "/tmp/l3tc_rust_dump")]
        out_dir: PathBuf,
    },
    /// Phase 4a debug: compute the *theoretical entropy bound* for
    /// a file under our forward pass — for direct comparison with
    /// Python L3TC's reported `entropy_sum / 8` ratio. Tokenizes
    /// `input` at the given segment size, runs the model on every
    /// segment, accumulates `-log2(softmax_p[next_token])` across
    /// all positions, and prints the total bits + byte ratio. No
    /// arithmetic coding, no segment headers, no file framing —
    /// just the entropy lower bound.
    EntropyBound {
        /// Input text file.
        #[arg(long)]
        input: PathBuf,
        /// Path to the converted L3TC model binary.
        #[arg(long, default_value = "checkpoints/l3tc_200k.bin")]
        model: PathBuf,
        /// Path to the SentencePiece tokenizer model.
        #[arg(
            long,
            default_value = "../vendor/L3TC/dictionary/vocab_enwik8_bpe_16384_0.999/spm_enwik8_bpe_16384_0.999.model"
        )]
        tokenizer: PathBuf,
        /// Segment length in bytes. Defaults to 2048 to match the
        /// Python L3TC reference's compressor.py default for
        /// L3TC-200K, so the entropy numbers are directly
        /// comparable.
        #[arg(long, default_value_t = 2048)]
        segment_bytes: usize,
    },
    /// Phase 4b debug: compress a file and print a per-source byte
    /// breakdown so we can localize where the gap between actual
    /// coded bytes and the entropy bound goes. Reports input
    /// bytes, entropy bound, AC body bytes, segment header bytes,
    /// unk payload bytes, file framing, and the difference
    /// accounting.
    Audit {
        /// Input text file.
        #[arg(long)]
        input: PathBuf,
        /// Path to the converted L3TC model binary.
        #[arg(long, default_value = "checkpoints/l3tc_200k.bin")]
        model: PathBuf,
        /// Path to the SentencePiece tokenizer model.
        #[arg(
            long,
            default_value = "../vendor/L3TC/dictionary/vocab_enwik8_bpe_16384_0.999/spm_enwik8_bpe_16384_0.999.model"
        )]
        tokenizer: PathBuf,
        /// Segment length in bytes.
        #[arg(long, default_value_t = DEFAULT_SEGMENT_BYTES)]
        segment_bytes: usize,
    },
    /// Phase 4c debug: per-phase timing breakdown of the hot
    /// loop. Sequentially compresses the input (no rayon) and
    /// reports how much wall-clock time is spent in forward
    /// pass vs cum_freqs vs AC encode vs everything else.
    Profile {
        /// Input text file.
        #[arg(long)]
        input: PathBuf,
        /// Path to the converted L3TC model binary.
        #[arg(long, default_value = "checkpoints/l3tc_200k.bin")]
        model: PathBuf,
        /// Path to the SentencePiece tokenizer model.
        #[arg(
            long,
            default_value = "../vendor/L3TC/dictionary/vocab_enwik8_bpe_16384_0.999/spm_enwik8_bpe_16384_0.999.model"
        )]
        tokenizer: PathBuf,
        /// Segment length in bytes.
        #[arg(long, default_value_t = DEFAULT_SEGMENT_BYTES)]
        segment_bytes: usize,
    },
    /// Phase 4e: dump top-K softmax distributions from a model
    /// running over an input file. Used to generate training
    /// data for knowledge distillation — the student model
    /// minimizes KL divergence against these distributions.
    ///
    /// Format documented at `codec::dump_teacher`. For L3TC-3.2M
    /// on enwik8 with top_k=64, the output is ~140 MB.
    DumpTeacher {
        /// Input text file (typically enwik8 for training).
        #[arg(long)]
        input: PathBuf,
        /// Path to the converted L3TC model binary (typically
        /// the teacher, e.g. checkpoints/l3tc_3m2.bin).
        #[arg(long)]
        model: PathBuf,
        /// Path to the SentencePiece tokenizer model.
        #[arg(
            long,
            default_value = "../vendor/L3TC/dictionary/vocab_enwik8_bpe_16384_0.999/spm_enwik8_bpe_16384_0.999.model"
        )]
        tokenizer: PathBuf,
        /// Output file for the dumped distributions.
        #[arg(short, long)]
        output: PathBuf,
        /// Segment length in bytes. Match the training
        /// configuration (L3TC reference uses 2048).
        #[arg(long, default_value_t = 2048)]
        segment_bytes: usize,
        /// Number of top probabilities to keep per step. The
        /// long tail is reconstructed at training time as a
        /// uniform floor. 64 is typical; use 128 for a
        /// higher-fidelity dump at 2× the storage cost.
        #[arg(long, default_value_t = 64)]
        top_k: usize,
    },
    /// Run content-based specialist detection on a file and print
    /// the result. Reads up to the first 4 KB of the input, runs
    /// `specialist::detect()`, and prints the chosen specialist plus
    /// the heuristic's confidence in `[0.0, 1.0]`.
    ///
    /// Exit code is always 0 on successful detection (even for
    /// `fallback` / low-confidence), nonzero only on I/O errors.
    /// The `--json` flag emits a machine-readable record suitable
    /// for scripts and the detection-accuracy eval harness.
    Detect {
        /// Input file to classify.
        input: PathBuf,
        /// Emit a JSON record instead of human-readable text.
        /// Schema: `{"file":"<path>","specialist":"<name>","confidence":<f32>,"bytes":<u64>}`
        #[arg(long)]
        json: bool,
    },

    /// Print version information.
    Version,

    /// Download, verify, and install the specialist model bundle.
    ///
    /// The release binary ships without models to keep the install
    /// fast and the Homebrew bottle small. This subcommand fetches
    /// the ~100 MB bundle, verifies every artifact's SHA-256
    /// against the manifest inside the tarball, and extracts it to
    /// the per-user models directory (default `~/.l3tc/models`).
    InstallModels {
        /// URL of the `.tar.zst` bundle to install. Defaults to the
        /// v0.1.0 GitHub Release asset.
        #[arg(long)]
        url: Option<String>,
        /// Install destination. Defaults to `$L3TC_MODEL_DIR` if set,
        /// else `~/.l3tc/models`.
        #[arg(long)]
        dest: Option<PathBuf>,
        /// Reinstall even if a bundle is already present at `--dest`.
        #[arg(long)]
        force: bool,
        /// Re-hash every installed artifact against the local
        /// `manifest.json`. Does not download anything.
        #[arg(long)]
        verify: bool,
        /// List installed specialists and their sizes. Does not
        /// hash-verify; use `--verify` for that.
        #[arg(long)]
        list: bool,
    },

    /// Phase 13b: run a self-contained Metal smoke test (compile a
    /// trivial element-wise add kernel, dispatch it, verify output).
    /// Available only when built with `--features=metal`.
    #[cfg(feature = "metal")]
    MetalSmoke {
        /// Number of f32 elements to operate on.
        #[arg(long, default_value_t = 1024)]
        n: usize,
    },

    /// Phase 13c benchmark: time the INT8 head matvec on CPU NEON vs
    /// Metal at the L3TC-200K shape (16384 × 96), single-call latency.
    /// Available only when built with `--features=metal`.
    #[cfg(feature = "metal")]
    MetalBenchHead {
        /// Iterations per backend (excluding warm-up).
        #[arg(long, default_value_t = 200)]
        iters: usize,
    },

    /// Phase 13m: isolate the Metal forward_batched per-token cost
    /// with no CPU AC work between calls. Runs N forward passes on
    /// a BatchedSession and reports mean time per token.
    #[cfg(feature = "metal")]
    MetalBenchForward {
        /// Number of tokens to run through forward_batched.
        #[arg(long, default_value_t = 200)]
        iters: usize,
        /// Batch size (lockstep lanes).
        #[arg(long, default_value_t = 32)]
        batch: usize,
        /// Path to the converted L3TC model binary.
        #[arg(long, default_value = "checkpoints/l3tc_200k.bin")]
        model: PathBuf,
    },

    /// Phase 13p: isolate per-stage cost inside a single forward
    /// pass. Times head matvec alone, cum_freqs alone, etc. so we
    /// know which stage to attack next.
    #[cfg(feature = "metal")]
    MetalBenchStages {
        /// Iterations per stage (warm-up is 10% of this).
        #[arg(long, default_value_t = 200)]
        iters: usize,
        /// Batch size.
        #[arg(long, default_value_t = 256)]
        batch: usize,
        /// Path to the converted L3TC model binary.
        #[arg(long, default_value = "checkpoints/l3tc_200k.bin")]
        model: PathBuf,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Version => {
            println!("l3tc {}", env!("CARGO_PKG_VERSION"));
            println!("phase 1 — research inference runtime");
            #[cfg(feature = "metal")]
            println!("backends: cpu, metal");
            #[cfg(not(feature = "metal"))]
            println!("backends: cpu");
            Ok(())
        }
        Command::Detect { input, json } => run_detect(&input, json),
        Command::InstallModels {
            url,
            dest,
            force,
            verify,
            list,
        } => {
            if list {
                install_models::run_list(dest.as_deref())
            } else if verify {
                install_models::run_verify(dest.as_deref())
            } else {
                install_models::run_install(url.as_deref(), dest.as_deref(), force)
            }
        }
        #[cfg(feature = "metal")]
        Command::MetalSmoke { n } => {
            use l3tc::backend::mtl;
            let t0 = Instant::now();
            match mtl::smoke_test(n) {
                Ok(processed) => {
                    let elapsed = t0.elapsed();
                    println!(
                        "Metal smoke test: OK — processed {} f32 elements in {:?} \
                         (compile + dispatch + verify roundtrip)",
                        processed, elapsed
                    );
                    Ok(())
                }
                Err(e) => Err(anyhow::anyhow!("Metal smoke test failed: {}", e)),
            }
        }
        #[cfg(feature = "metal")]
        Command::MetalBenchHead { iters } => run_metal_bench_head(iters),
        #[cfg(feature = "metal")]
        Command::MetalBenchForward {
            iters,
            batch,
            model,
        } => run_metal_bench_forward(iters, batch, &model),
        #[cfg(feature = "metal")]
        Command::MetalBenchStages {
            iters,
            batch,
            model,
        } => run_metal_bench_stages(iters, batch, &model),
        Command::Compress {
            input,
            output,
            model,
            tokenizer,
            specialist,
            verbose,
            segment_bytes,
            time,
            verify,
            backend,
            metal_batch,
            metal_workers,
        } => run_compress(
            &input,
            output.as_deref(),
            model.as_deref(),
            tokenizer.as_deref(),
            &specialist,
            verbose,
            segment_bytes,
            time,
            verify,
            &backend,
            metal_batch,
            metal_workers,
        ),
        Command::Decompress {
            input,
            output,
            model,
            tokenizer,
            verbose,
            time,
            backend,
        } => run_decompress(
            &input,
            output.as_deref(),
            model.as_deref(),
            tokenizer.as_deref(),
            verbose,
            time,
            &backend,
        ),
        Command::HybridCompress {
            input,
            output,
            model,
            tokenizer,
            zstd_dict,
            chunk_size,
            stats,
        } => run_hybrid_compress(
            &input,
            output.as_deref(),
            model.as_deref(),
            tokenizer.as_deref(),
            zstd_dict.as_deref(),
            chunk_size,
            stats.as_deref(),
        ),
        Command::HybridDecompress {
            input,
            output,
            model,
            tokenizer,
            zstd_dict,
        } => run_hybrid_decompress(
            &input,
            output.as_deref(),
            model.as_deref(),
            tokenizer.as_deref(),
            zstd_dict.as_deref(),
        ),
        Command::DumpLogits {
            input,
            model,
            tokenizer,
            segment_bytes,
            max_tokens,
            out_dir,
        } => run_dump_logits(
            &input,
            &model,
            &tokenizer,
            segment_bytes,
            max_tokens,
            &out_dir,
        ),
        Command::EntropyBound {
            input,
            model,
            tokenizer,
            segment_bytes,
        } => run_entropy_bound(&input, &model, &tokenizer, segment_bytes),
        Command::Audit {
            input,
            model,
            tokenizer,
            segment_bytes,
        } => run_audit(&input, &model, &tokenizer, segment_bytes),
        Command::Profile {
            input,
            model,
            tokenizer,
            segment_bytes,
        } => run_profile(&input, &model, &tokenizer, segment_bytes),
        Command::DumpTeacher {
            input,
            model,
            tokenizer,
            output,
            segment_bytes,
            top_k,
        } => run_dump_teacher(&input, &model, &tokenizer, &output, segment_bytes, top_k),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("l3tc: {e:#}");
            ExitCode::from(1)
        }
    }
}

#[cfg(feature = "metal")]
#[cfg(feature = "metal")]
fn run_metal_bench_stages(iters: usize, batch: usize, model_path: &Path) -> anyhow::Result<()> {
    use l3tc::backend::mtl::{CumFreqsKernelMetal, HeadKernelMetal, MetalError};
    use l3tc::checkpoint::Checkpoint;
    use l3tc::rwkv::Model;

    let mut ckpt = Checkpoint::load(model_path)
        .map_err(|e| anyhow::anyhow!("loading checkpoint {model_path:?}: {e}"))?;
    let model =
        Model::from_checkpoint(&mut ckpt).map_err(|e| anyhow::anyhow!("building model: {e}"))?;

    let h = model.hidden_size;
    let vocab = model.vocab_size;

    // Head matvec: (h → vocab) per lane, batch lanes.
    let head = match HeadKernelMetal::new(&model.head_q, &model.head_scales, vocab, h) {
        Ok(k) => k,
        Err(MetalError::NoDevice) => return Err(anyhow::anyhow!("no Metal device")),
        Err(e) => return Err(anyhow::anyhow!("head init: {e}")),
    };

    // Cum_freqs: (vocab logits → u32 freqs) per lane.
    let cum = match CumFreqsKernelMetal::new(vocab) {
        Ok(k) => k,
        Err(e) => return Err(anyhow::anyhow!("cum init: {e}")),
    };

    let warmup = (iters / 10).max(5);
    let x_in = vec![0.0f32; batch * h];
    let mut logits_out = vec![0.0f32; batch * vocab];
    let logits_in = vec![0.0f32; batch * vocab];
    let mut freqs_out = vec![0u32; batch * vocab];

    for _ in 0..warmup {
        head.forward_batched(&x_in, &mut logits_out, batch);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        head.forward_batched(&x_in, &mut logits_out, batch);
    }
    let head_elapsed = t0.elapsed();
    let head_per = head_elapsed / iters as u32;

    for _ in 0..warmup {
        cum.forward_batched(&logits_in, &mut freqs_out, batch, 10_000_000);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        cum.forward_batched(&logits_in, &mut freqs_out, batch, 10_000_000);
    }
    let cum_elapsed = t0.elapsed();
    let cum_per = cum_elapsed / iters as u32;

    println!("Metal per-stage bench");
    println!("  h={h}, vocab={vocab}, batch={batch}, iters={iters}");
    println!("  head.forward_batched (in-isolation, incl. upload/commit/wait/readback):");
    println!(
        "    per-call: {head_per:?}  ({:.1} µs)",
        head_per.as_secs_f64() * 1e6
    );
    println!("  cum_freqs.forward_batched (in-isolation, incl. upload/commit/wait/readback):");
    println!(
        "    per-call: {cum_per:?}  ({:.1} µs)",
        cum_per.as_secs_f64() * 1e6
    );
    Ok(())
}

#[cfg(feature = "metal")]
fn run_metal_bench_forward(iters: usize, batch: usize, model_path: &Path) -> anyhow::Result<()> {
    use l3tc::backend::batched::BatchedSession;
    use l3tc::backend::mtl::MetalError;
    use l3tc::checkpoint::Checkpoint;
    use l3tc::rwkv::Model;

    let mut ckpt = Checkpoint::load(model_path)
        .map_err(|e| anyhow::anyhow!("loading checkpoint {model_path:?}: {e}"))?;
    let model =
        Model::from_checkpoint(&mut ckpt).map_err(|e| anyhow::anyhow!("building model: {e}"))?;

    let mut bs = match BatchedSession::new(&model, batch) {
        Ok(s) => s,
        Err(MetalError::NoDevice) => {
            return Err(anyhow::anyhow!("no Metal device available"));
        }
        Err(e) => return Err(anyhow::anyhow!("BatchedSession::new: {e}")),
    };

    let prev: Vec<u32> = (0..batch).map(|b| (b as u32) % 16u32).collect();

    // Warm up (pipeline JIT, GPU buffers).
    let warmup = 20.min(iters / 5).max(5);
    for _ in 0..warmup {
        bs.forward_batched(&prev);
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        bs.forward_batched(&prev);
    }
    let total = t0.elapsed();
    let per_tok = total / iters as u32;
    let per_lane = per_tok / batch as u32;

    println!("Metal forward_batched bench");
    println!("  model       : {model_path:?}");
    println!("  batch       : {batch}");
    println!("  iters       : {iters} (after {warmup} warm-up)");
    println!("  total       : {total:?}");
    println!("  per-token   : {per_tok:?}");
    println!(
        "  per-lane-tok: {per_lane:?}  ({:.0} µs)",
        per_lane.as_secs_f64() * 1e6
    );
    println!(
        "  aggregate   : ~{:.1} KB/s (assuming 2 B/token across {batch} lanes)",
        (batch * 2) as f64 / per_tok.as_secs_f64() / 1000.0
    );
    Ok(())
}

#[cfg(feature = "metal")]
fn run_metal_bench_head(iters: usize) -> anyhow::Result<()> {
    use l3tc::backend::mtl::{HeadKernelMetal, MetalError};
    use l3tc::tensor;

    // L3TC-200K head shape.
    let rows: usize = 16384;
    let cols: usize = 96;
    let warmup: usize = 20;

    // Synthetic weights mirroring the unit test.
    let mut qmat = vec![0i8; rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            qmat[j * rows + i] = (((i as i32 * 31 + j as i32 * 7) % 251) - 125) as i8;
        }
    }
    let scales: Vec<f32> = (0..cols).map(|j| 1e-4 + (j as f32) * 1e-5).collect();
    let x: Vec<f32> = (0..cols).map(|j| ((j as f32) - 48.0) * 0.05).collect();

    println!(
        "head matvec INT8 bench — shape {}×{}, {} warm-up + {} iters",
        rows, cols, warmup, iters
    );

    // CPU NEON (Phase 12d).
    let mut cpu_out = vec![0.0f32; rows];
    for _ in 0..warmup {
        tensor::matvec_col_major_int8(&qmat, &scales, &x, &mut cpu_out, rows, cols);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        tensor::matvec_col_major_int8(&qmat, &scales, &x, &mut cpu_out, rows, cols);
    }
    let cpu_total = t0.elapsed();
    let cpu_per_call = cpu_total / iters as u32;

    // Metal (Phase 13c) — batch=1.
    let kernel = HeadKernelMetal::new(&qmat, &scales, rows, cols).map_err(|e| {
        if matches!(e, MetalError::NoDevice) {
            anyhow::anyhow!("no Metal device available on this machine")
        } else {
            anyhow::anyhow!("Metal kernel init failed: {}", e)
        }
    })?;
    let mut gpu_out = vec![0.0f32; rows];
    for _ in 0..warmup {
        kernel.forward(&x, &mut gpu_out);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        kernel.forward(&x, &mut gpu_out);
    }
    let gpu_total = t0.elapsed();
    let gpu_per_call = gpu_total / iters as u32;

    let speedup = cpu_per_call.as_secs_f64() / gpu_per_call.as_secs_f64();

    println!(
        "  CPU NEON               : {:?} per call ({:?} total)",
        cpu_per_call, cpu_total
    );
    println!(
        "  Metal GPU batch=1      : {:?} per call ({:?} total) [{:.2}× CPU]",
        gpu_per_call, gpu_total, speedup
    );

    // Sweep batched throughput for representative batch sizes.
    println!("\nBatched dispatch (per-token amortized):");
    for &batch in &[8usize, 32, 64, 128, 256, 512] {
        let xb: Vec<f32> = (0..batch).flat_map(|_| x.iter().copied()).collect();
        let mut outb = vec![0.0f32; batch * rows];
        // Warm up
        for _ in 0..(warmup / 4).max(2) {
            kernel.forward_batched(&xb, &mut outb, batch);
        }
        let n_calls = (iters / batch).max(4);
        let t0 = Instant::now();
        for _ in 0..n_calls {
            kernel.forward_batched(&xb, &mut outb, batch);
        }
        let elapsed = t0.elapsed();
        let per_call = elapsed / n_calls as u32;
        let per_token = per_call / batch as u32;
        let cpu_speedup = cpu_per_call.as_secs_f64() / per_token.as_secs_f64();
        println!(
            "  batch={:4} : {:>9?} per call, {:>9?} per token  [{:.1}× CPU per-token]",
            batch, per_call, per_token, cpu_speedup
        );
    }

    println!(
        "\nNote: per-token amortized cost is what matters for throughput. \n      A compress workload runs ~60K predict steps per 200KB of text."
    );
    Ok(())
}

/// Read the first 4 KB of `input` and run the specialist detector.
/// Prints a single human-readable line or a JSON record, depending
/// on `json`. Returns non-zero only for I/O errors — low-confidence
/// or fallback results still exit 0 because "we couldn't classify
/// confidently" is a valid outcome, not a program failure.
fn run_detect(input: &Path, json: bool) -> Result<()> {
    use std::io::Read;

    let mut f =
        std::fs::File::open(input).with_context(|| format!("open input: {}", input.display()))?;
    // Detection only looks at the first 4 KB (see specialist::detect);
    // reading more would be wasted work.
    let mut buf = [0u8; 4096];
    let n = f.read(&mut buf).context("read input")?;
    let total = f.metadata().map(|m| m.len()).unwrap_or(n as u64);

    let d = detect(&buf[..n]);
    if json {
        // Stable schema — used by bench/detection_eval.py. Keep
        // field names aligned with the public spec.
        println!(
            "{{\"file\":\"{}\",\"specialist\":\"{}\",\"confidence\":{:.4},\"bytes\":{}}}",
            input
                .display()
                .to_string()
                .replace('\\', "\\\\")
                .replace('"', "\\\""),
            d.specialist.name(),
            d.confidence,
            total,
        );
    } else {
        println!(
            "{}\tspecialist={}\tconfidence={:.2}\tbytes={}",
            input.display(),
            d.specialist.name(),
            d.confidence,
            total,
        );
    }
    Ok(())
}

#[cfg_attr(not(feature = "metal"), allow(unused_variables))]
// CLI entry point: flags map 1:1 to function args so clap can call
// this from the Command::Compress match arm without pre-bundling.
#[allow(clippy::too_many_arguments)]
fn run_compress(
    input: &Path,
    output: Option<&Path>,
    model_override: Option<&Path>,
    tokenizer_override: Option<&Path>,
    specialist_arg: &str,
    verbose: bool,
    segment_bytes: usize,
    print_time: bool,
    verify: bool,
    backend: &str,
    metal_batch: usize,
    metal_workers: usize,
) -> Result<()> {
    let default_out = input.with_extension({
        let ext = input.extension().and_then(|s| s.to_str()).unwrap_or("");
        if ext.is_empty() {
            "l3tc".to_string()
        } else {
            format!("{ext}.l3tc")
        }
    });
    let output = output.unwrap_or(&default_out);

    let input_bytes = std::fs::metadata(input)
        .with_context(|| format!("stat input {input:?}"))?
        .len() as usize;
    let read_dt = Instant::now().elapsed();

    // Resolve specialist + model/tokenizer paths.
    let (chosen_specialist, model_path, tokenizer_path, resolved) =
        resolve_compress_assets(input, model_override, tokenizer_override, specialist_arg)?;
    if verbose {
        print_specialist_note("compress", chosen_specialist, resolved.as_ref());
    }

    let t0 = Instant::now();
    let model = load_model(&model_path)?;
    let model_dt = t0.elapsed();

    let t0 = Instant::now();
    let tokenizer = Tokenizer::load(&tokenizer_path)
        .with_context(|| format!("loading tokenizer {tokenizer_path:?}"))?;
    let tok_load_dt = t0.elapsed();

    let model_id = chosen_specialist.to_byte();

    let t0 = Instant::now();
    let _ = segment_bytes; // segment size honoured per-backend below
    match backend {
        "cpu" => {
            let in_file =
                std::fs::File::open(input).with_context(|| format!("opening input {input:?}"))?;
            let src = std::io::BufReader::new(in_file);
            let out_file = std::fs::File::create(output)
                .with_context(|| format!("creating output {output:?}"))?;
            let mut out_buf = std::io::BufWriter::new(out_file);
            encode_reader(
                src,
                &mut out_buf,
                &tokenizer,
                &model,
                segment_bytes,
                model_id,
            )
            .with_context(|| "compression failed")?;
            use std::io::Write;
            out_buf
                .flush()
                .with_context(|| format!("flushing output {output:?}"))?;
        }
        #[cfg(feature = "metal")]
        "metal" => {
            let text = std::fs::read_to_string(input)
                .with_context(|| format!("reading input {input:?}"))?;
            let bytes = l3tc::codec::compress_with_metal_workers(
                &text,
                &tokenizer,
                &model,
                segment_bytes,
                metal_batch.max(1),
                metal_workers.max(1),
                model_id,
            )
            .with_context(|| "metal compression failed")?;
            std::fs::write(output, &bytes).with_context(|| format!("writing output {output:?}"))?;
        }
        #[cfg(not(feature = "metal"))]
        "metal" => {
            return Err(anyhow::anyhow!(
                "this build does not include Metal support — \
                 rebuild with `--features=metal` to use --backend=metal"
            ));
        }
        other => {
            return Err(anyhow::anyhow!(
                "unknown backend {other:?}; expected `cpu` or `metal`"
            ));
        }
    }
    let compress_dt = t0.elapsed();
    let write_dt = std::time::Duration::from_secs(0);

    let output_bytes = std::fs::metadata(output)
        .with_context(|| format!("stat output {output:?}"))?
        .len() as usize;
    let ratio = output_bytes as f64 / input_bytes.max(1) as f64;
    let compress_kb_s = (input_bytes as f64 / 1024.0) / compress_dt.as_secs_f64();

    println!(
        "{} -> {}  {} bytes -> {} bytes  ratio {:.4}",
        input.display(),
        output.display(),
        input_bytes,
        output_bytes,
        ratio,
    );

    if print_time {
        println!("  read:     {:?}", read_dt);
        println!("  model:    {:?}", model_dt);
        println!("  tok load: {:?}", tok_load_dt);
        println!("  compress: {:?}  ({:.2} KB/s)", compress_dt, compress_kb_s);
        println!("  write:    {:?}", write_dt);
    }

    if verify {
        let vt0 = Instant::now();
        let compressed =
            std::fs::read(output).with_context(|| format!("verify: reading {output:?}"))?;
        let original =
            std::fs::read(input).with_context(|| format!("verify: reading {input:?}"))?;
        // verify uses the same backend the file was compressed with
        let decompressed = match backend {
            #[cfg(feature = "metal")]
            "metal" => l3tc::codec::decompress_with_metal(&compressed, &tokenizer, &model)
                .with_context(|| "verify: metal decompression failed")?
                .into_bytes(),
            _ => decompress_bytes(&compressed, &tokenizer, &model)
                .with_context(|| "verify: decompression failed")?,
        };
        let verify_dt = vt0.elapsed();
        if decompressed != original {
            return Err(anyhow::anyhow!(
                "verify: decompressed output differs from input"
            ));
        }
        let dec_kb_s = (input_bytes as f64 / 1024.0) / verify_dt.as_secs_f64();
        println!(
            "  verify:   {:?}  ({:.2} KB/s decompress)",
            verify_dt, dec_kb_s
        );
        println!("  round-trip: OK");
    }

    Ok(())
}

fn run_decompress(
    input: &Path,
    output: Option<&Path>,
    model_override: Option<&Path>,
    tokenizer_override: Option<&Path>,
    verbose: bool,
    print_time: bool,
    backend: &str,
) -> Result<()> {
    // Default output strips the .l3tc extension and appends .out
    let default_out = {
        let mut p = input.with_extension("");
        if p.extension().is_none() {
            p.set_extension("out");
        }
        p
    };
    let output = output.unwrap_or(&default_out);

    let read_dt = std::time::Duration::from_secs(0);

    // Resolve model/tokenizer from the header's specialist id, unless
    // the user overrode. This has to happen before model load so we
    // know which checkpoint to read.
    let (file_specialist, model_path, tokenizer_path, resolved) =
        resolve_decompress_assets(input, model_override, tokenizer_override)?;
    if verbose {
        print_specialist_note("decompress", file_specialist, resolved.as_ref());
    }

    let t0 = Instant::now();
    let model = load_model(&model_path)?;
    let model_dt = t0.elapsed();

    let t0 = Instant::now();
    let tokenizer = Tokenizer::load(&tokenizer_path)
        .with_context(|| format!("loading tokenizer {tokenizer_path:?}"))?;
    let tok_load_dt = t0.elapsed();

    // Auto-detect backend by reading the file's flag byte if
    // requested. Allows GPU-encoded files to round-trip on a
    // GPU-equipped machine without the user passing --backend=metal.
    let chosen_backend = if backend == "auto" {
        let buf = std::fs::read(input).with_context(|| format!("opening input {input:?}"))?;
        // Header layout: magic(4) + version(1) + flags(1) + ...
        let flags_byte = if buf.len() > 5 { buf[5] } else { 0 };
        let gpu_flag = flags_byte & 0x02 != 0; // FLAG_GPU_ENCODED
        if gpu_flag {
            "metal"
        } else {
            "cpu"
        }
    } else {
        backend
    };
    if backend == "auto" {
        eprintln!("auto-detected backend: {chosen_backend}");
    }

    let t0 = Instant::now();
    let written = match chosen_backend {
        "cpu" => {
            let in_file =
                std::fs::File::open(input).with_context(|| format!("opening input {input:?}"))?;
            let src = std::io::BufReader::new(in_file);
            let out_file = std::fs::File::create(output)
                .with_context(|| format!("creating output {output:?}"))?;
            let mut out_buf = std::io::BufWriter::new(out_file);
            let n = decode_writer(src, &mut out_buf, &tokenizer, &model)
                .with_context(|| "decompression failed")?;
            use std::io::Write;
            out_buf
                .flush()
                .with_context(|| format!("flushing output {output:?}"))?;
            n
        }
        #[cfg(feature = "metal")]
        "metal" => {
            let bytes = std::fs::read(input).with_context(|| format!("reading input {input:?}"))?;
            let text = l3tc::codec::decompress_with_metal(&bytes, &tokenizer, &model)
                .with_context(|| "metal decompression failed")?;
            std::fs::write(output, text.as_bytes())
                .with_context(|| format!("writing output {output:?}"))?;
            text.len() as u64
        }
        #[cfg(not(feature = "metal"))]
        "metal" => {
            return Err(anyhow::anyhow!(
                "this build does not include Metal support — \
                 rebuild with `--features=metal` to decompress GPU-encoded files"
            ));
        }
        other => {
            return Err(anyhow::anyhow!(
                "unknown backend {other:?}; expected `cpu`, `metal`, or `auto`"
            ));
        }
    };
    let decompress_dt = t0.elapsed();
    let write_dt = std::time::Duration::from_secs(0);

    let output_bytes = written as usize;
    let decompress_kb_s = (output_bytes as f64 / 1024.0) / decompress_dt.as_secs_f64();

    let compressed_bytes = std::fs::metadata(input)
        .with_context(|| format!("stat input {input:?}"))?
        .len() as usize;
    println!(
        "{} -> {}  {} bytes -> {} bytes",
        input.display(),
        output.display(),
        compressed_bytes,
        output_bytes,
    );

    if print_time {
        println!("  read:        {:?}", read_dt);
        println!("  model:       {:?}", model_dt);
        println!("  tok load:    {:?}", tok_load_dt);
        println!(
            "  decompress:  {:?}  ({:.2} KB/s)",
            decompress_dt, decompress_kb_s
        );
        println!("  write:       {:?}", write_dt);
    }

    Ok(())
}

/// Phase 4a debug entry: compute the entropy lower bound on a full
/// file. Mirrors what Python L3TC's `compressor.py` reports as
/// `total_bin_size_min = math.ceil(entropy_sum / 8)`. Each segment
/// gets a fresh model state; for each (current_token, next_token)
/// pair we accumulate `-log2(softmax(logits)[next_token])`.
fn run_entropy_bound(
    input: &Path,
    model_path: &Path,
    tokenizer_path: &Path,
    segment_bytes: usize,
) -> Result<()> {
    use l3tc::Session;

    let model = load_model(model_path)?;
    let tokenizer = Tokenizer::load(tokenizer_path)
        .with_context(|| format!("loading tokenizer {tokenizer_path:?}"))?;

    let raw = std::fs::read(input).with_context(|| format!("reading input {input:?}"))?;
    let total_input_bytes = raw.len();
    let text = std::str::from_utf8(&raw).with_context(|| "input is not valid UTF-8")?;

    let segments = tokenizer
        .encode_file(text, segment_bytes)
        .with_context(|| "tokenizing input")?;

    let mut total_bits: f64 = 0.0;
    let mut total_steps: u64 = 0;
    let mut session = Session::new(&model);

    let n_segs = segments.len();
    for (si, seg) in segments.iter().enumerate() {
        if seg.needs_raw_fallback {
            // Skip raw-fallback segments — Python reference uses
            // SPM round-trippable text only and falls back to a
            // separate path. For entropy comparison we ignore
            // these (they don't contribute model entropy).
            continue;
        }
        session.reset();
        // Forward through every (prev, next) pair in this segment.
        for i in 1..seg.tokens.len() {
            let prev = seg.tokens[i - 1];
            let next = seg.tokens[i] as usize;
            let logits = session.forward(prev);
            // Numerically stable softmax + -log2(p[next]).
            let mut max = f32::NEG_INFINITY;
            for &l in logits {
                if l > max {
                    max = l;
                }
            }
            let mut sum = 0.0f64;
            for &l in logits {
                sum += ((l - max) as f64).exp();
            }
            let target_logit = (logits[next] - max) as f64;
            let log_p = target_logit - sum.ln();
            // bits = -log2(p) = -log_p / ln(2)
            total_bits += -log_p / std::f64::consts::LN_2;
            total_steps += 1;
        }

        if (si + 1) % 32 == 0 || si == n_segs - 1 {
            let bytes = total_bits / 8.0;
            let ratio = bytes / total_input_bytes as f64;
            eprintln!(
                "  seg {}/{}: bits={:.0}, bytes={:.0}, running ratio={:.4}",
                si + 1,
                n_segs,
                total_bits,
                bytes,
                ratio
            );
        }
    }

    let total_bytes_min = (total_bits / 8.0).ceil();
    let ratio = total_bytes_min / total_input_bytes as f64;
    println!();
    println!("=== entropy bound ===");
    println!("input bytes:        {}", total_input_bytes);
    println!("segments:           {}", segments.len());
    println!("predict steps:      {}", total_steps);
    println!("total bits:         {:.2}", total_bits);
    println!("total_bin_size_min: {}", total_bytes_min as u64);
    println!("ratio:              {:.6}", ratio);
    Ok(())
}

/// Phase 4b debug entry: print a per-source byte breakdown of the
/// gap between actual coded bytes and the entropy bound. This is
/// the measurement that drives the Phase 4b optimization work.
fn run_audit(
    input: &Path,
    model_path: &Path,
    tokenizer_path: &Path,
    segment_bytes: usize,
) -> Result<()> {
    let model = load_model(model_path)?;
    let tokenizer = Tokenizer::load(tokenizer_path)
        .with_context(|| format!("loading tokenizer {tokenizer_path:?}"))?;

    let text =
        std::fs::read_to_string(input).with_context(|| format!("reading input {input:?}"))?;

    let stats: AuditStats = audit_compress(&text, &tokenizer, &model, segment_bytes)
        .with_context(|| "audit_compress failed")?;

    let total = stats.total_compressed_bytes();
    let bound = stats.entropy_bound_bytes();
    let overhead = stats.overhead_bytes();
    let input_bytes = stats.input_bytes;

    fn pct_of(n: u64, denom: u64) -> f64 {
        if denom == 0 {
            0.0
        } else {
            100.0 * n as f64 / denom as f64
        }
    }

    println!("=== audit ===");
    println!("input file:                  {}", input.display());
    println!("input bytes:                 {input_bytes}");
    println!("segment_bytes:               {segment_bytes}");
    println!("segments:                    {}", stats.n_segments);
    println!(
        "  raw-fallback segments:     {}",
        stats.n_raw_fallback_segments
    );
    println!("tokens (incl. BOS):          {}", stats.n_tokens);
    println!("predict steps (encoded):     {}", stats.n_predict_steps);
    println!();

    println!("entropy bound:");
    println!("  bits:                      {:.2}", stats.entropy_bits);
    println!(
        "  bytes (ceil bits/8):       {bound}     ({:.4}% of input)",
        pct_of(bound, input_bytes)
    );
    println!(
        "  avg bits/predict step:     {:.4}",
        if stats.n_predict_steps == 0 {
            0.0
        } else {
            stats.entropy_bits / stats.n_predict_steps as f64
        }
    );
    println!();

    println!("actual coded bytes:");
    println!(
        "  AC body bytes:             {:>10}     ({:.4}% of input)",
        stats.ac_body_bytes,
        pct_of(stats.ac_body_bytes, input_bytes)
    );
    println!(
        "  segment header bytes:      {:>10}     ({:.4}% of input, {:.2} bytes/segment avg)",
        stats.segment_header_bytes,
        pct_of(stats.segment_header_bytes, input_bytes),
        if stats.n_segments == 0 {
            0.0
        } else {
            stats.segment_header_bytes as f64 / stats.n_segments as f64
        }
    );
    println!(
        "  unk payload bytes:         {:>10}     ({:.4}% of input)",
        stats.unk_payload_bytes,
        pct_of(stats.unk_payload_bytes, input_bytes)
    );
    println!(
        "  raw-fallback bytes:        {:>10}     ({:.4}% of input)",
        stats.raw_fallback_bytes,
        pct_of(stats.raw_fallback_bytes, input_bytes)
    );
    println!(
        "  file header:               {:>10}",
        stats.file_header_bytes
    );
    println!(
        "  file trailer:              {:>10}",
        stats.file_trailer_bytes
    );
    println!("  file CRC32:                {:>10}", stats.file_crc_bytes);
    println!(
        "  TOTAL:                     {total}     ({:.4}% of input)",
        pct_of(total, input_bytes)
    );
    println!();

    println!("overhead vs entropy bound:");
    println!(
        "  AC tail bytes (body - bound): {} bytes  (avg {:.2} bytes/segment)",
        stats.ac_body_bytes as i64 - bound as i64,
        if stats.n_segments == 0 {
            0.0
        } else {
            (stats.ac_body_bytes as f64 - bound as f64) / stats.n_segments as f64
        }
    );
    println!("  framing overhead:           {} bytes", overhead);
    println!(
        "  ratio actual / input:       {:.6}",
        total as f64 / input_bytes.max(1) as f64
    );
    println!(
        "  ratio entropy / input:      {:.6}",
        bound as f64 / input_bytes.max(1) as f64
    );
    println!(
        "  gap to entropy:             {:.4} pp ({} bytes)",
        100.0 * (total as f64 - bound as f64) / input_bytes.max(1) as f64,
        overhead
    );

    Ok(())
}

/// Phase 4c debug: run a sequential compress with per-phase
/// timing and print the breakdown.
fn run_profile(
    input: &Path,
    model_path: &Path,
    tokenizer_path: &Path,
    segment_bytes: usize,
) -> Result<()> {
    let model = load_model(model_path)?;
    let tokenizer = Tokenizer::load(tokenizer_path)
        .with_context(|| format!("loading tokenizer {tokenizer_path:?}"))?;
    let text =
        std::fs::read_to_string(input).with_context(|| format!("reading input {input:?}"))?;

    let stats: ProfileStats = profile_compress(&text, &tokenizer, &model, segment_bytes)
        .with_context(|| "profile_compress failed")?;

    let total = stats.total_ns.max(1);
    let pct = |x: u128| -> f64 { 100.0 * x as f64 / total as f64 };
    let per_step = |x: u128| -> f64 {
        if stats.n_predict_steps == 0 {
            0.0
        } else {
            x as f64 / stats.n_predict_steps as f64 / 1000.0
        }
    };

    println!("=== profile ===");
    println!("input file:          {}", input.display());
    println!("input bytes:         {}", text.len());
    println!("segments:            {}", stats.n_segments);
    println!("tokens (incl. BOS):  {}", stats.n_tokens);
    println!("predict steps:       {}", stats.n_predict_steps);
    println!();
    println!(
        "total wall-clock:    {:>10.3} ms",
        stats.total_ns as f64 / 1_000_000.0
    );
    println!(
        "forward pass:        {:>10.3} ms  ({:5.2}%)  {:6.2} us/step",
        stats.forward_ns as f64 / 1_000_000.0,
        pct(stats.forward_ns),
        per_step(stats.forward_ns),
    );
    println!(
        "cum_freqs:           {:>10.3} ms  ({:5.2}%)  {:6.2} us/step",
        stats.cum_freqs_ns as f64 / 1_000_000.0,
        pct(stats.cum_freqs_ns),
        per_step(stats.cum_freqs_ns),
    );
    println!(
        "AC encode:           {:>10.3} ms  ({:5.2}%)  {:6.2} us/step",
        stats.ac_encode_ns as f64 / 1_000_000.0,
        pct(stats.ac_encode_ns),
        per_step(stats.ac_encode_ns),
    );
    println!(
        "other (bookkeeping): {:>10.3} ms  ({:5.2}%)",
        stats.other_ns as f64 / 1_000_000.0,
        pct(stats.other_ns),
    );
    println!();
    let kb_s = if stats.total_ns > 0 {
        (text.len() as f64 / 1024.0) / (stats.total_ns as f64 / 1e9)
    } else {
        0.0
    };
    println!(
        "throughput:          {:.2} KB/s (single-thread, no rayon)",
        kb_s
    );

    Ok(())
}

/// Phase 4e entry: run a model over an input file and dump
/// top-K teacher softmax distributions for distillation
/// training.
fn run_dump_teacher(
    input: &Path,
    model_path: &Path,
    tokenizer_path: &Path,
    output: &Path,
    segment_bytes: usize,
    top_k: usize,
) -> Result<()> {
    let model = load_model(model_path)?;
    let tokenizer = Tokenizer::load(tokenizer_path)
        .with_context(|| format!("loading tokenizer {tokenizer_path:?}"))?;

    let text =
        std::fs::read_to_string(input).with_context(|| format!("reading input {input:?}"))?;

    println!(
        "dumping top-{top_k} teacher distributions: {} -> {}",
        input.display(),
        output.display()
    );
    println!("  model:       {}", model_path.display());
    println!(
        "  hidden_size: {}   intermediate: {}   layers: {}",
        model.hidden_size,
        model.intermediate_size,
        model.num_layers()
    );
    println!("  input bytes: {}", text.len());
    println!("  segment_bytes: {segment_bytes}");

    let out_file = std::fs::File::create(output).with_context(|| format!("creating {output:?}"))?;
    let mut buf = std::io::BufWriter::new(out_file);

    let t0 = Instant::now();
    let n_steps = dump_teacher(&text, &tokenizer, &model, segment_bytes, top_k, &mut buf)
        .with_context(|| "dump_teacher failed")?;
    use std::io::Write;
    buf.flush().with_context(|| "flushing output")?;
    drop(buf);
    let elapsed = t0.elapsed();

    let out_bytes = std::fs::metadata(output)
        .with_context(|| "stat output")?
        .len();
    let kb_s = (text.len() as f64 / 1024.0) / elapsed.as_secs_f64();

    println!();
    println!("  predict steps dumped: {n_steps}");
    println!("  output bytes:         {out_bytes}");
    println!(
        "  per-step cost:        {:.1} bytes",
        out_bytes as f64 / n_steps.max(1) as f64
    );
    println!("  wall-clock:           {:?}", elapsed);
    println!("  throughput:           {kb_s:.2} KB/s");
    Ok(())
}

fn load_model(path: &Path) -> Result<Model> {
    let mut ckpt =
        Checkpoint::load(path).with_context(|| format!("loading checkpoint {path:?}"))?;
    Model::from_checkpoint(&mut ckpt).with_context(|| "building model from checkpoint")
}

/// Assemble the dispatcher codec menu based on which optional assets
/// the caller supplied. `Zstd22` is unconditional (it's the
/// safety-net reference). Neural joins the menu iff both model and
/// tokenizer loaded; `ZstdDict` joins iff a dictionary was loaded;
/// `Clp` is always present as the stub (it can never actually win
/// the shortest-pick race — see `ClpStub` docs).
fn build_hybrid_codecs(
    model: Option<Arc<Model>>,
    tokenizer: Option<Arc<Tokenizer>>,
    zstd_dict: Option<Vec<u8>>,
) -> Vec<Box<dyn Codec>> {
    let mut codecs: Vec<Box<dyn Codec>> = vec![
        Box::new(PassthroughCodec),
        Box::new(Lz4Codec),
        Box::new(Zstd22Codec),
        Box::new(Bzip3Codec),
        Box::new(ClpStub),
    ];
    if let Some(dict) = zstd_dict {
        codecs.push(Box::new(ZstdDictCodec::new(dict)));
    }
    if let (Some(m), Some(t)) = (model, tokenizer) {
        codecs.push(Box::new(NeuralCodec::new(t, m)));
    }
    codecs
}

/// Load `model_path` + `tokenizer_path` into Arc-wrapped references
/// so multiple Codec instances can share them. Both paths are
/// validated together — supplying one without the other is an
/// argument error.
fn load_hybrid_neural(
    model_path: Option<&Path>,
    tokenizer_path: Option<&Path>,
) -> Result<(Option<Arc<Model>>, Option<Arc<Tokenizer>>)> {
    match (model_path, tokenizer_path) {
        (None, None) => Ok((None, None)),
        (Some(m), Some(t)) => {
            let model = load_model(m)?;
            let tok = Tokenizer::load(t)
                .with_context(|| format!("loading tokenizer {t:?}"))?;
            Ok((Some(Arc::new(model)), Some(Arc::new(tok))))
        }
        _ => Err(anyhow::anyhow!(
            "--model and --tokenizer must be supplied together"
        )),
    }
}

/// Serialize `DispatchStats` to the JSON shape the service-side
/// `compression_worker.py` expects when emitting CloudWatch EMF
/// metrics. Kept hand-rolled (no serde) since the crate already
/// avoids serde in the library path for binary-size reasons, and
/// the shape is small + stable.
fn dispatch_stats_json(stats: &DispatchStats) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(512);
    s.push_str("{\n");
    writeln!(s, "  \"bytes_in\": {},", stats.bytes_in).unwrap();
    writeln!(s, "  \"bytes_out\": {},", stats.bytes_out).unwrap();
    writeln!(s, "  \"chunks_total\": {},", stats.chunks_total).unwrap();
    writeln!(s, "  \"zstd_shadow_bytes\": {},", stats.zstd_shadow_bytes).unwrap();
    writeln!(s, "  \"ratio\": {:.6},", stats.ratio()).unwrap();
    writeln!(
        s,
        "  \"zstd_shadow_ratio\": {:.6},",
        stats.zstd_shadow_ratio()
    )
    .unwrap();
    writeln!(
        s,
        "  \"savings_vs_zstd_pct\": {:.4},",
        stats.savings_vs_zstd_pct()
    )
    .unwrap();
    writeln!(
        s,
        "  \"throughput_mb_per_sec\": {:.4},",
        stats.throughput_mb_per_sec()
    )
    .unwrap();
    writeln!(
        s,
        "  \"safety_net_substitutions\": {},",
        stats.safety_net_substitutions
    )
    .unwrap();
    writeln!(s, "  \"encode_seconds\": {:.6},", stats.encode_seconds).unwrap();
    // Per-codec byte counts. Emit as a nested object keyed by codec
    // name. We sort keys so the emitted JSON is deterministic between
    // runs — makes EMF dimension values stable for the metric query.
    let mut tags: Vec<&&'static str> = stats.per_codec_bytes.keys().collect();
    tags.sort();
    s.push_str("  \"per_codec_bytes\": {");
    for (i, tag) in tags.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        write!(s, "\"{}\": {}", tag, stats.per_codec_bytes[*tag]).unwrap();
    }
    s.push_str("},\n");
    let mut ctags: Vec<&&'static str> = stats.per_codec_chunks.keys().collect();
    ctags.sort();
    s.push_str("  \"per_codec_chunks\": {");
    for (i, tag) in ctags.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        write!(s, "\"{}\": {}", tag, stats.per_codec_chunks[*tag]).unwrap();
    }
    s.push_str("}\n");
    s.push('}');
    s
}

fn run_hybrid_compress(
    input: &Path,
    output: Option<&Path>,
    model_path: Option<&Path>,
    tokenizer_path: Option<&Path>,
    zstd_dict_path: Option<&Path>,
    chunk_size: usize,
    stats_path: Option<&Path>,
) -> Result<()> {
    let input_bytes = fs::read(input)
        .with_context(|| format!("reading input {input:?}"))?;

    let (model, tokenizer) = load_hybrid_neural(model_path, tokenizer_path)?;
    let dict_bytes: Option<Vec<u8>> = match zstd_dict_path {
        Some(p) => Some(fs::read(p).with_context(|| format!("reading zstd dict {p:?}"))?),
        None => None,
    };

    let codecs = build_hybrid_codecs(model, tokenizer, dict_bytes);

    let (blob, stats) = hybrid_encode(&input_bytes, &codecs, chunk_size)
        .with_context(|| "hybrid_encode failed")?;

    let out_path = output
        .map(PathBuf::from)
        .unwrap_or_else(|| input.with_extension("l3h"));
    fs::write(&out_path, &blob)
        .with_context(|| format!("writing output {out_path:?}"))?;

    // Human-readable stats on stderr so stdout stays clean for pipe-
    // friendly machine consumption if we ever add it.
    eprintln!(
        "hybrid-compress: {} B in -> {} B out (ratio {:.4}, zstd-shadow {:.4}, \
         savings vs zstd {:+.2}%, {} chunks, safety-net x{}, {:.2} MB/s)",
        stats.bytes_in,
        stats.bytes_out,
        stats.ratio(),
        stats.zstd_shadow_ratio(),
        stats.savings_vs_zstd_pct(),
        stats.chunks_total,
        stats.safety_net_substitutions,
        stats.throughput_mb_per_sec(),
    );
    let mut per_codec: Vec<(&&'static str, &u64)> = stats.per_codec_chunks.iter().collect();
    per_codec.sort_by_key(|(k, _)| *k);
    for (k, v) in per_codec {
        let bytes = stats.per_codec_bytes.get(*k).copied().unwrap_or(0);
        eprintln!("  codec {k:<12} chunks={v:>5}  bytes={bytes}");
    }

    if let Some(p) = stats_path {
        fs::write(p, dispatch_stats_json(&stats))
            .with_context(|| format!("writing stats json {p:?}"))?;
    }

    Ok(())
}

fn run_hybrid_decompress(
    input: &Path,
    output: Option<&Path>,
    model_path: Option<&Path>,
    tokenizer_path: Option<&Path>,
    zstd_dict_path: Option<&Path>,
) -> Result<()> {
    let blob = fs::read(input).with_context(|| format!("reading input {input:?}"))?;

    let (model, tokenizer) = load_hybrid_neural(model_path, tokenizer_path)?;
    let dict_bytes: Option<Vec<u8>> = match zstd_dict_path {
        Some(p) => Some(fs::read(p).with_context(|| format!("reading zstd dict {p:?}"))?),
        None => None,
    };

    let codecs = build_hybrid_codecs(model, tokenizer, dict_bytes);

    let decoded = hybrid_decode(&blob, &codecs).with_context(|| "hybrid_decode failed")?;

    let out_path = output.map(PathBuf::from).unwrap_or_else(|| {
        // Strip `.l3h` suffix if present; else append `.out`.
        let s = input.to_string_lossy();
        if let Some(stripped) = s.strip_suffix(".l3h") {
            PathBuf::from(stripped)
        } else {
            input.with_extension("out")
        }
    });
    fs::write(&out_path, &decoded)
        .with_context(|| format!("writing output {out_path:?}"))?;
    eprintln!(
        "hybrid-decompress: {} B -> {} B written to {:?}",
        blob.len(),
        decoded.len(),
        out_path,
    );
    Ok(())
}

/// Compress-side asset resolution.
///
/// When the user supplies `--model` (and implicitly `--tokenizer`),
/// honor them and stamp the header with whatever `--specialist` says
/// (defaulting to `Unspecified` = 0). When the user omits the paths,
/// run `detect()` on the first 4 KB of input (or honor `--specialist`
/// if set to something other than `auto`) and ask the registry for
/// the matching files.
///
/// Returns `(chosen_specialist, model_path, tokenizer_path, resolved)`.
/// `resolved` is `Some` when the path came from the registry, `None`
/// when the user overrode manually — used by the verbose banner.
fn resolve_compress_assets(
    input: &Path,
    model_override: Option<&Path>,
    tokenizer_override: Option<&Path>,
    specialist_arg: &str,
) -> Result<(Specialist, PathBuf, PathBuf, Option<ResolvedSpecialist>)> {
    // Parse the --specialist flag.
    let requested = Specialist::from_cli_name(specialist_arg).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown --specialist {specialist_arg:?}; expected one of: auto, prose, \
             code, structured, logs, tabular, markup, fallback"
        )
    })?;

    // User-supplied model+tokenizer: trust them, only the specialist
    // tag written to the header comes from --specialist.
    if let (Some(m), Some(t)) = (model_override, tokenizer_override) {
        return Ok((requested, m.to_path_buf(), t.to_path_buf(), None));
    }
    if model_override.is_some() ^ tokenizer_override.is_some() {
        return Err(anyhow::anyhow!(
            "--model and --tokenizer must be supplied together (or neither, for auto-detect)"
        ));
    }

    // No overrides: pick a specialist.
    let chosen = if requested == Specialist::Unspecified {
        // "auto": sample the input and detect.
        let sample = read_detection_sample(input)?;
        let det = detect(&sample);
        det.specialist
    } else {
        requested
    };

    let resolved = registry::resolve(chosen).ok_or_else(|| {
        anyhow::anyhow!(
            "no model installed for specialist {chosen}; {}",
            install_models::missing_models_hint(),
        )
    })?;

    // Header records what we asked for, even if we fell back to the
    // legacy generalist — the tag is a hint for the decoder, and if
    // future-us installs that specialist the files already in the
    // wild will pick it up on decompress.
    Ok((
        chosen,
        resolved.assets.model.clone(),
        resolved.assets.tokenizer.clone(),
        Some(resolved),
    ))
}

/// Decompress-side asset resolution: peek the header to learn which
/// specialist encoded the file, then resolve via the registry (unless
/// the user overrode both paths).
fn resolve_decompress_assets(
    input: &Path,
    model_override: Option<&Path>,
    tokenizer_override: Option<&Path>,
) -> Result<(Specialist, PathBuf, PathBuf, Option<ResolvedSpecialist>)> {
    let peek = {
        let f = std::fs::File::open(input).with_context(|| format!("opening input {input:?}"))?;
        let mut r = std::io::BufReader::new(f);
        l3tc::peek_header(&mut r).with_context(|| format!("reading header of {input:?}"))?
    };
    let file_specialist = Specialist::from_byte(peek.model_id);

    if let (Some(m), Some(t)) = (model_override, tokenizer_override) {
        return Ok((file_specialist, m.to_path_buf(), t.to_path_buf(), None));
    }
    if model_override.is_some() ^ tokenizer_override.is_some() {
        return Err(anyhow::anyhow!(
            "--model and --tokenizer must be supplied together (or neither, to load from registry)"
        ));
    }

    let resolved = registry::resolve(file_specialist).ok_or_else(|| {
        anyhow::anyhow!(
            "file was encoded with specialist {file_specialist}, but no matching model \
             is installed. {}",
            install_models::missing_models_hint(),
        )
    })?;

    Ok((
        file_specialist,
        resolved.assets.model.clone(),
        resolved.assets.tokenizer.clone(),
        Some(resolved),
    ))
}

/// Read up to 4 KB of the input for the detector. Must not consume
/// the file handle used later by the encoder.
fn read_detection_sample(input: &Path) -> Result<Vec<u8>> {
    use std::io::Read;
    let f = std::fs::File::open(input)
        .with_context(|| format!("opening input for detection {input:?}"))?;
    let mut r = std::io::BufReader::new(f);
    let mut buf = vec![0u8; 4096];
    let mut n = 0;
    while n < buf.len() {
        match r.read(&mut buf[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) => return Err(anyhow::Error::new(e).context("reading detection sample")),
        }
    }
    buf.truncate(n);
    Ok(buf)
}

fn print_specialist_note(phase: &str, chosen: Specialist, resolved: Option<&ResolvedSpecialist>) {
    match resolved {
        Some(r) if r.is_fallback() => {
            eprintln!(
                "l3tc {phase}: specialist={chosen}, model not installed — falling back to \
                 {} ({})",
                r.resolved,
                r.assets.model.display()
            );
        }
        Some(r) => {
            eprintln!(
                "l3tc {phase}: specialist={chosen} ({})",
                r.assets.model.display()
            );
        }
        None => {
            eprintln!("l3tc {phase}: specialist={chosen} (user-supplied --model/--tokenizer)");
        }
    }
}

/// Phase 4a debug entry: tokenize a fixed slice of the input, forward
/// each token through the model, and dump per-token logits to a binary
/// file in the same format as `scripts/dump_python_logits.py`. The
/// resulting `tokens.bin` and `logits.bin` can be diffed against the
/// Python reference dump to localize where the forward pass diverges.
fn run_dump_logits(
    input: &Path,
    model_path: &Path,
    tokenizer_path: &Path,
    segment_bytes: usize,
    max_tokens: usize,
    out_dir: &Path,
) -> Result<()> {
    use l3tc::Session;
    use std::io::Write;

    let model = load_model(model_path)?;
    let tokenizer = Tokenizer::load(tokenizer_path)
        .with_context(|| format!("loading tokenizer {tokenizer_path:?}"))?;

    // Tokenize exactly the same prefix the Python harness sees:
    // first `segment_bytes` of the input file as one segment.
    let raw = std::fs::read(input).with_context(|| format!("reading input {input:?}"))?;
    let take = raw.len().min(segment_bytes);
    let text =
        std::str::from_utf8(&raw[..take]).with_context(|| "input prefix is not valid UTF-8")?;
    let seg = tokenizer
        .encode_segment(text)
        .with_context(|| "tokenizing input prefix")?;
    let mut tokens: Vec<u32> = seg.tokens.clone();
    if max_tokens > 0 && tokens.len() > max_tokens {
        tokens.truncate(max_tokens);
    }
    println!(
        "  tokens: {} (first 10: {:?})",
        tokens.len(),
        &tokens[..tokens.len().min(10)]
    );

    std::fs::create_dir_all(out_dir).with_context(|| format!("mkdir {out_dir:?}"))?;

    // Write tokens.bin
    let tokens_path = out_dir.join("tokens.bin");
    let mut tf = std::io::BufWriter::new(
        std::fs::File::create(&tokens_path).with_context(|| format!("creating {tokens_path:?}"))?,
    );
    tf.write_all(&(tokens.len() as u32).to_le_bytes())
        .with_context(|| "write n_tokens")?;
    for &t in &tokens {
        tf.write_all(&t.to_le_bytes())
            .with_context(|| "write token")?;
    }
    tf.flush().with_context(|| "flush tokens.bin")?;
    drop(tf);

    // Forward each token, dump logits.
    let mut session = Session::new(&model);
    let logits_path = out_dir.join("logits.bin");
    let mut lf = std::io::BufWriter::new(
        std::fs::File::create(&logits_path).with_context(|| format!("creating {logits_path:?}"))?,
    );
    lf.write_all(&(tokens.len() as u32).to_le_bytes())
        .with_context(|| "write n_tokens")?;
    lf.write_all(&(model.vocab_size as u32).to_le_bytes())
        .with_context(|| "write vocab")?;

    for (i, &tok) in tokens.iter().enumerate() {
        let logits = session.forward(tok);
        // Write all logits as f32 LE.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(logits.as_ptr() as *const u8, std::mem::size_of_val(logits))
        };
        lf.write_all(bytes).with_context(|| "write logits")?;
        if (i + 1) % 32 == 0 || i == tokens.len() - 1 {
            println!("  forward {}/{}", i + 1, tokens.len());
        }
    }
    lf.flush().with_context(|| "flush logits.bin")?;
    println!("wrote: {}", logits_path.display());
    Ok(())
}

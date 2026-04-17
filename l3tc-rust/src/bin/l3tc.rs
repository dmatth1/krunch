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
    audit_compress, decode_writer, decompress_bytes, dump_teacher, encode_reader,
    profile_compress, AuditStats, Checkpoint, Model, ProfileStats, Tokenizer,
    DEFAULT_SEGMENT_BYTES,
};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

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
        /// Path to the converted L3TC model binary
        /// (produced by scripts/convert_checkpoint.py).
        #[arg(long, default_value = "checkpoints/l3tc_200k.bin")]
        model: PathBuf,
        /// Path to the SentencePiece tokenizer model.
        #[arg(
            long,
            default_value = "../vendor/L3TC/dictionary/vocab_enwik8_bpe_16384_0.999/spm_enwik8_bpe_16384_0.999.model"
        )]
        tokenizer: PathBuf,
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
    },
    /// Decompress a file produced by `compress`.
    Decompress {
        /// Input file to decompress.
        input: PathBuf,
        /// Output path. Defaults to `<input>.out`.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Path to the converted L3TC model binary.
        #[arg(long, default_value = "checkpoints/l3tc_200k.bin")]
        model: PathBuf,
        /// Path to the SentencePiece tokenizer model.
        #[arg(
            long,
            default_value = "../vendor/L3TC/dictionary/vocab_enwik8_bpe_16384_0.999/spm_enwik8_bpe_16384_0.999.model"
        )]
        tokenizer: PathBuf,
        /// Print per-phase timing information.
        #[arg(long)]
        time: bool,
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
    /// Print version information.
    Version,

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
        Command::Compress {
            input,
            output,
            model,
            tokenizer,
            segment_bytes,
            time,
            verify,
        } => run_compress(&input, output.as_deref(), &model, &tokenizer, segment_bytes, time, verify),
        Command::Decompress {
            input,
            output,
            model,
            tokenizer,
            time,
        } => run_decompress(&input, output.as_deref(), &model, &tokenizer, time),
        Command::DumpLogits {
            input,
            model,
            tokenizer,
            segment_bytes,
            max_tokens,
            out_dir,
        } => run_dump_logits(&input, &model, &tokenizer, segment_bytes, max_tokens, &out_dir),
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
        } => run_dump_teacher(
            &input,
            &model,
            &tokenizer,
            &output,
            segment_bytes,
            top_k,
        ),
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

fn run_compress(
    input: &Path,
    output: Option<&Path>,
    model_path: &Path,
    tokenizer_path: &Path,
    segment_bytes: usize,
    print_time: bool,
    verify: bool,
) -> Result<()> {
    let default_out = input.with_extension({
        let ext = input.extension().and_then(|s| s.to_str()).unwrap_or("");
        if ext.is_empty() { "l3tc".to_string() } else { format!("{ext}.l3tc") }
    });
    let output = output.unwrap_or(&default_out);

    // Stat the input up front so we can report the ratio without
    // having to hold the whole thing in memory.
    let input_bytes = std::fs::metadata(input)
        .with_context(|| format!("stat input {input:?}"))?
        .len() as usize;
    let read_dt = Instant::now().elapsed(); // placeholder; streaming reads during compress

    let t0 = Instant::now();
    let model = load_model(model_path)?;
    let model_dt = t0.elapsed();

    let t0 = Instant::now();
    let tokenizer = Tokenizer::load(tokenizer_path)
        .with_context(|| format!("loading tokenizer {tokenizer_path:?}"))?;
    let tok_load_dt = t0.elapsed();

    // Streaming encode: read input in bounded batches, write each
    // batch's compressed segments immediately. Peak RSS is the
    // model + tokenizer + ~4 MB batch buffer, independent of
    // input size.
    let t0 = Instant::now();
    let in_file = std::fs::File::open(input)
        .with_context(|| format!("opening input {input:?}"))?;
    let src = std::io::BufReader::new(in_file);
    let out_file = std::fs::File::create(output)
        .with_context(|| format!("creating output {output:?}"))?;
    let mut out_buf = std::io::BufWriter::new(out_file);
    encode_reader(src, &mut out_buf, &tokenizer, &model, segment_bytes)
        .with_context(|| "compression failed")?;
    use std::io::Write;
    out_buf
        .flush()
        .with_context(|| format!("flushing output {output:?}"))?;
    drop(out_buf);
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
        let compressed = std::fs::read(output)
            .with_context(|| format!("verify: reading {output:?}"))?;
        let original = std::fs::read(input)
            .with_context(|| format!("verify: reading {input:?}"))?;
        let decompressed = decompress_bytes(&compressed, &tokenizer, &model)
            .with_context(|| "verify: decompression failed")?;
        let verify_dt = vt0.elapsed();
        if decompressed != original {
            return Err(anyhow::anyhow!(
                "verify: decompressed output differs from input"
            ));
        }
        let dec_kb_s = (input_bytes as f64 / 1024.0) / verify_dt.as_secs_f64();
        println!("  verify:   {:?}  ({:.2} KB/s decompress)", verify_dt, dec_kb_s);
        println!("  round-trip: OK");
    }

    Ok(())
}

fn run_decompress(
    input: &Path,
    output: Option<&Path>,
    model_path: &Path,
    tokenizer_path: &Path,
    print_time: bool,
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

    let t0 = Instant::now();
    let model = load_model(model_path)?;
    let model_dt = t0.elapsed();

    let t0 = Instant::now();
    let tokenizer = Tokenizer::load(tokenizer_path)
        .with_context(|| format!("loading tokenizer {tokenizer_path:?}"))?;
    let tok_load_dt = t0.elapsed();

    // Streaming decode: pipe src → dst with bounded RSS for
    // raw-store files. Tokenized files still slurp the compressed
    // body internally (it's small) but the output write stays
    // streaming.
    let t0 = Instant::now();
    let in_file = std::fs::File::open(input)
        .with_context(|| format!("opening input {input:?}"))?;
    let src = std::io::BufReader::new(in_file);
    let out_file = std::fs::File::create(output)
        .with_context(|| format!("creating output {output:?}"))?;
    let mut out_buf = std::io::BufWriter::new(out_file);
    let written = decode_writer(src, &mut out_buf, &tokenizer, &model)
        .with_context(|| "decompression failed")?;
    use std::io::Write;
    out_buf
        .flush()
        .with_context(|| format!("flushing output {output:?}"))?;
    drop(out_buf);
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
        println!("  decompress:  {:?}  ({:.2} KB/s)", decompress_dt, decompress_kb_s);
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

    let raw = std::fs::read(input)
        .with_context(|| format!("reading input {input:?}"))?;
    let total_input_bytes = raw.len();
    let text = std::str::from_utf8(&raw)
        .with_context(|| "input is not valid UTF-8")?;

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
                si + 1, n_segs, total_bits, bytes, ratio
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

    let text = std::fs::read_to_string(input)
        .with_context(|| format!("reading input {input:?}"))?;

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
        stats.ac_body_bytes, pct_of(stats.ac_body_bytes, input_bytes)
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
        stats.unk_payload_bytes, pct_of(stats.unk_payload_bytes, input_bytes)
    );
    println!(
        "  raw-fallback bytes:        {:>10}     ({:.4}% of input)",
        stats.raw_fallback_bytes, pct_of(stats.raw_fallback_bytes, input_bytes)
    );
    println!(
        "  file header:               {:>10}",
        stats.file_header_bytes
    );
    println!(
        "  file trailer:              {:>10}",
        stats.file_trailer_bytes
    );
    println!(
        "  file CRC32:                {:>10}",
        stats.file_crc_bytes
    );
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
    let text = std::fs::read_to_string(input)
        .with_context(|| format!("reading input {input:?}"))?;

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
    println!("throughput:          {:.2} KB/s (single-thread, no rayon)", kb_s);

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

    let text = std::fs::read_to_string(input)
        .with_context(|| format!("reading input {input:?}"))?;

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

    let out_file = std::fs::File::create(output)
        .with_context(|| format!("creating {output:?}"))?;
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
    let mut ckpt = Checkpoint::load(path)
        .with_context(|| format!("loading checkpoint {path:?}"))?;
    Model::from_checkpoint(&mut ckpt).with_context(|| "building model from checkpoint")
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
    let raw = std::fs::read(input)
        .with_context(|| format!("reading input {input:?}"))?;
    let take = raw.len().min(segment_bytes);
    let text = std::str::from_utf8(&raw[..take])
        .with_context(|| "input prefix is not valid UTF-8")?;
    let seg = tokenizer
        .encode_segment(text)
        .with_context(|| "tokenizing input prefix")?;
    let mut tokens: Vec<u32> = seg.tokens.clone();
    if max_tokens > 0 && tokens.len() > max_tokens {
        tokens.truncate(max_tokens);
    }
    println!("  tokens: {} (first 10: {:?})", tokens.len(), &tokens[..tokens.len().min(10)]);

    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("mkdir {out_dir:?}"))?;

    // Write tokens.bin
    let tokens_path = out_dir.join("tokens.bin");
    let mut tf = std::io::BufWriter::new(
        std::fs::File::create(&tokens_path)
            .with_context(|| format!("creating {tokens_path:?}"))?,
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
        std::fs::File::create(&logits_path)
            .with_context(|| format!("creating {logits_path:?}"))?,
    );
    lf.write_all(&(tokens.len() as u32).to_le_bytes())
        .with_context(|| "write n_tokens")?;
    lf.write_all(&(model.vocab_size as u32).to_le_bytes())
        .with_context(|| "write vocab")?;

    for (i, &tok) in tokens.iter().enumerate() {
        let logits = session.forward(tok);
        // Write all logits as f32 LE.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                logits.as_ptr() as *const u8,
                logits.len() * std::mem::size_of::<f32>(),
            )
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

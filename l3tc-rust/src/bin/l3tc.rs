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
    decode_writer, decompress_bytes, encode_reader, Checkpoint, Model, Tokenizer,
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
    /// Print version information.
    Version,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Version => {
            println!("l3tc {}", env!("CARGO_PKG_VERSION"));
            println!("phase 1 — research inference runtime");
            Ok(())
        }
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
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("l3tc: {e:#}");
            ExitCode::from(1)
        }
    }
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

fn load_model(path: &Path) -> Result<Model> {
    let mut ckpt = Checkpoint::load(path)
        .with_context(|| format!("loading checkpoint {path:?}"))?;
    Model::from_checkpoint(&mut ckpt).with_context(|| "building model from checkpoint")
}

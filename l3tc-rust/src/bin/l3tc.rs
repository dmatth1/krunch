//! l3tc — command-line interface.
//!
//! Phase 1 stub: the compress/decompress subcommands will be wired
//! up once the model and codec modules land. For now this binary
//! just proves the crate builds and the CLI parser works.

use clap::{Parser, Subcommand};
use std::process::ExitCode;

/// l3tc — learned lossless text compression.
#[derive(Parser, Debug)]
#[command(name = "l3tc", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Compress a file.
    Compress {
        /// Input file path. If omitted, reads from stdin.
        input: Option<std::path::PathBuf>,
        /// Output file path. If omitted, writes to stdout.
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,
    },
    /// Decompress a file.
    Decompress {
        /// Input file path. If omitted, reads from stdin.
        input: Option<std::path::PathBuf>,
        /// Output file path. If omitted, writes to stdout.
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,
    },
    /// Print version and build information.
    Version,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Version => {
            println!("l3tc {}", env!("CARGO_PKG_VERSION"));
            println!("build: phase 1 (scaffolding)");
            println!("status: compress/decompress not yet wired up");
            ExitCode::SUCCESS
        }
        Command::Compress { input: _, output: _ } => {
            eprintln!("l3tc: compress not yet implemented (phase 1 in progress)");
            ExitCode::from(2)
        }
        Command::Decompress { input: _, output: _ } => {
            eprintln!("l3tc: decompress not yet implemented (phase 1 in progress)");
            ExitCode::from(2)
        }
    }
}

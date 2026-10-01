//! Single entry point for Confucius-R2T2 speech recognition.
//!
//! Three modes, one binary:
//!
//! ```text
//! r2t2 transcribe -i audio.wav
//! r2t2 serve      --port 8272
//! r2t2 subtitle   -i movie.mp4 -o movie.srt
//! ```
//!
//! The modes share the model, the engine and the shared flags; only the work
//! around them differs.

use std::process::ExitCode;

use anyhow::Result;
use clap::Parser;

use r2t2::cli::{Cli, Command};

fn main() -> ExitCode {
    let cli = Cli::parse();

    // `serve` is async and installs its own tracing subscriber; the file modes
    // are synchronous and report through stderr directly.
    let result: Result<()> = match cli.command {
        Command::Transcribe(args) => r2t2::cli::transcribe::run(&args),
        Command::Subtitle(args) => r2t2::cli::subtitle::run(&args),
        Command::Serve(args) => {
            tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| "info".into()),
                )
                .init();
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("failed to start the async runtime")
                .block_on(r2t2::cli::serve::run(args))
        }
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

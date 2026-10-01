//! Command-line surface.
//!
//! One binary, one subcommand per mode. Everything the three modes agree on --
//! where the model lives, which language to expect, how much context to give
//! llama.cpp -- is defined once in [`CommonArgs`] and flattened into each
//! subcommand, so the flags cannot drift apart between modes.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

pub mod serve;
pub mod subtitle;
pub mod transcribe;

/// Transcribe speech, serve it live, or generate subtitles.
#[derive(Debug, Parser)]
#[command(
    name = "r2t2",
    version,
    about = "Speech recognition with Confucius-R2T2, via llama.cpp",
    long_about = "Speech recognition with Confucius-R2T2, via llama.cpp.\n\n\
                  Runs entirely in process: no Python, no conda, only the NVIDIA\n\
                  driver is required at runtime."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Transcribe an audio file, streaming or in one shot.
    Transcribe(transcribe::TranscribeArgs),

    /// Serve live audio over WebSocket.
    Serve(serve::ServeArgs),

    /// Generate an .srt subtitle file from a video or audio file.
    Subtitle(subtitle::SubtitleArgs),
}

/// Settings every mode shares.
#[derive(Debug, Args)]
pub struct CommonArgs {
    /// Directory holding exactly one `mmproj*.gguf` and one other `*.gguf`.
    #[arg(long = "gguf-dir", value_name = "DIR", default_value = "checkpoints/gguf")]
    pub gguf_dir: PathBuf,

    /// Language hint, e.g. `Chinese` or `English`.
    #[arg(short = 'l', long = "language", value_name = "LANG", default_value = "Chinese")]
    pub language: String,

    /// Let the model detect the language instead of forcing one.
    #[arg(long = "auto-language", conflicts_with = "language")]
    pub auto_language: bool,

    /// Context or hotword hint, e.g. names and terminology.
    #[arg(short = 'c', long = "context", value_name = "TEXT", default_value = "")]
    pub context: String,

    /// Context size for llama.cpp.
    #[arg(long = "n-ctx", value_name = "N", default_value_t = 8192)]
    pub n_ctx: u32,

    /// Batch size for llama.cpp.
    #[arg(long = "n-batch", value_name = "N", default_value_t = 2048)]
    pub n_batch: u32,

    /// CPU threads for llama.cpp.
    #[arg(long = "n-threads", value_name = "N", default_value_t = 16)]
    pub n_threads: i32,

    /// Run on CPU only.
    #[arg(long = "cpu-only")]
    pub cpu_only: bool,

    /// Print progress information to stderr.
    #[arg(short = 'v', long = "verbose")]
    pub verbose: bool,
}

impl CommonArgs {
    /// The language to force, or `None` to let the model decide.
    pub fn forced_language(&self) -> Option<&str> {
        (!self.auto_language).then(|| self.language.as_str())
    }

    /// Build engine settings from the shared flags plus the resolved model.
    pub fn engine_config(
        &self,
        model: impl Into<PathBuf>,
        mmproj: impl Into<PathBuf>,
    ) -> crate::engine::EngineConfig {
        let mut cfg = crate::engine::EngineConfig::new(model.into(), mmproj.into());
        cfg.n_ctx = self.n_ctx;
        cfg.n_batch = self.n_batch;
        cfg.n_threads = self.n_threads;
        cfg.use_gpu = !self.cpu_only;
        cfg.n_gpu_layers = if self.cpu_only { 0 } else { -1 };
        cfg
    }

    /// Resolve the GGUF pair, reporting it when `--verbose`.
    pub fn resolve_model(&self) -> anyhow::Result<(PathBuf, PathBuf)> {
        let (model, mmproj) = crate::model::resolve_gguf(&self.gguf_dir)?;
        if self.verbose {
            eprintln!("model  : {}", model.display());
            eprintln!("mmproj : {}", mmproj.display());
        }
        Ok((model, mmproj))
    }
}

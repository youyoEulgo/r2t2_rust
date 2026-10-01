//! Command-line surface.
//!
//! One binary, one subcommand per mode. Everything the three modes agree on --
//! where the model lives, which language to expect, how much context to give
//! llama.cpp -- is defined once in [`CommonArgs`] and flattened into each
//! subcommand, so the flags cannot drift apart between modes.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

pub mod mux;
pub mod serve;
pub mod transcribe;

/// Transcribe speech, serve it live, or generate subtitles.
///
/// Results go to stdout so they can be piped; diagnostics go to stderr and only
/// with `--verbose`. Nothing is written to a file unless asked for with `-o`,
/// which keeps a shell redirect and `-o` interchangeable.
#[derive(Debug, Parser)]
#[command(
    name = "r2t2",
    version,
    // `-v` prints the version rather than toggling verbosity: that is the
    // convention for a CLI, and verbosity is one flag nobody types often
    // enough to need a short form.
    disable_version_flag = true,
    about = "Speech recognition with Confucius-R2T2, via llama.cpp",
    long_about = "Speech recognition with Confucius-R2T2, via llama.cpp.\n\n\
                  Runs entirely in process: no Python, no conda, only the NVIDIA\n\
                  driver is required at runtime.\n\n\
                  Results are written to stdout, so redirect or pipe them to save:\n\
                  r2t2 transcribe -i audio.wav > out.txt"
)]
pub struct Cli {
    /// Print version information.
    #[arg(short = 'v', long = "version", action = clap::ArgAction::Version)]
    pub version: (),

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Transcribe an audio file, streaming or in one shot.
    Transcribe(transcribe::TranscribeArgs),

    /// Serve live audio over WebSocket.
    Serve(serve::ServeArgs),

    /// Combine a video and a subtitle file into one Matroska file.
    Mux(mux::MuxArgs),
}

/// Settings every mode shares.
#[derive(Debug, Args)]
pub struct CommonArgs {
    /// Directory holding exactly one `mmproj*.gguf` and one other `*.gguf`.
    ///
    /// Defaults to `~/.local/share/r2t2/models`, which is where the program
    /// offers to download the model on first use.
    #[arg(long = "gguf-dir", value_name = "DIR")]
    pub gguf_dir: Option<PathBuf>,

    /// Download the model without asking, if it is missing.
    ///
    /// The prompt is skipped automatically when stdin is not a terminal, so
    /// unattended runs fail with instructions rather than hanging.
    #[arg(long = "yes", short = 'y')]
    pub assume_yes: bool,

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

    /// Print progress and timing information to stderr.
    ///
    /// Diagnostics never go to stdout, so they cannot contaminate a piped
    /// result.
    #[arg(long = "verbose")]
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
        // llama.cpp narrates everything it does. Without --verbose that output
        // buries this program's own messages, so it is switched off before any
        // model is loaded.
        if !self.verbose {
            crate::engine::silence_llama_logging();
        }

        // Resolves the default directory, offering a download when it is
        // empty; an explicit --gguf-dir is trusted as-is.
        let dir = crate::paths::resolve_models(self.gguf_dir.as_deref(), self.assume_yes)?;
        let (model, mmproj) = crate::model::resolve_gguf(&dir)?;
        if self.verbose {
            eprintln!("model  : {}", model.display());
            eprintln!("mmproj : {}", mmproj.display());
        }
        Ok((model, mmproj))
    }
}

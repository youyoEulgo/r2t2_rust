// Copyright 2026 youyoEulgo
// SPDX-License-Identifier: Apache-2.0

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
                  Runs entirely in process: no Python and no conda. The GPU is\n\
                  used through Metal on macOS and CUDA on Linux; without either,\n\
                  it falls back to the CPU.\n\n\
                  Results are written to stdout, so redirect or pipe them to save:\n\
                  r2t2 transcribe -i audio.wav > out.txt\n\n\
                  Run with no arguments to start the live caption server."
)]
pub struct Cli {
    /// Print version information.
    #[arg(short = 'v', long = "version", action = clap::ArgAction::Version)]
    pub version: (),

    /// Server settings, accepted at the top level as well as after `serve`.
    ///
    /// Flattening these here means a bare `r2t2 --port 9000` works, which is
    /// what someone who double-clicked the program and then opened a terminal
    /// will try. They are ignored unless the effective command is `serve`.
    #[command(flatten)]
    pub serve: serve::ServeArgs,

    #[command(subcommand)]
    pub command: Option<Command>,
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

impl Default for Command {
    /// What a bare `r2t2` does.
    ///
    /// Running with no arguments starts the server rather than printing usage.
    /// The usual way to start a program on Windows is to double-click it, and a
    /// console window that flashes a help screen and vanishes is neither useful
    /// nor self-explanatory; starting the interface the user was evidently
    /// after is.
    fn default() -> Self {
        Command::Serve(serve::ServeArgs::default())
    }
}

/// Settings every mode shares.
///
/// `Default` mirrors the clap defaults, and exists so that a bare `r2t2` can
/// build a `ServeArgs` without going through the parser.
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

impl Default for CommonArgs {
    fn default() -> Self {
        Self {
            gguf_dir: None,
            assume_yes: false,
            language: "Chinese".to_string(),
            auto_language: false,
            context: String::new(),
            n_ctx: 8192,
            n_batch: 2048,
            n_threads: 16,
            cpu_only: false,
            verbose: false,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser as _;

    /// A bare `r2t2` must mean `serve`.
    #[test]
    fn no_subcommand_defaults_to_serve() {
        let cli = Cli::try_parse_from(["r2t2"]).expect("a bare invocation should parse");
        assert!(cli.command.is_none(), "the subcommand should be optional");
        assert!(matches!(Command::default(), Command::Serve(_)));
    }

    /// A bare run must accept the server's flags, not just `r2t2 serve`.
    #[test]
    fn a_bare_run_accepts_server_flags() {
        let cli = Cli::try_parse_from(["r2t2", "--port", "9000", "--no-rtmp"])
            .expect("a bare run should accept server flags");
        assert!(cli.command.is_none());
        assert_eq!(cli.serve.port, 9000);
        assert!(cli.serve.no_rtmp);
    }

    /// `Command::default` is built by hand, so it can drift from the clap
    /// defaults it is supposed to mirror. This is what catches that.
    #[test]
    fn defaults_match_the_clap_defaults() {
        let parsed = Cli::try_parse_from(["r2t2", "serve"]).expect("serve should parse");
        let Some(Command::Serve(from_clap)) = parsed.command else {
            panic!("expected the serve subcommand");
        };
        let built = match Command::default() {
            Command::Serve(args) => args,
            _ => panic!("the default command should be serve"),
        };

        assert_eq!(built.bind, from_clap.bind);
        assert_eq!(built.port, from_clap.port);
        assert_eq!(built.rtmp_port, from_clap.rtmp_port);
        assert_eq!(built.vad_sensitivity, from_clap.vad_sensitivity);
        assert_eq!(built.vad_min_silence_ms, from_clap.vad_min_silence_ms);
        assert_eq!(built.vad_min_speech_ms, from_clap.vad_min_speech_ms);
        assert_eq!(built.no_web, from_clap.no_web);
        assert_eq!(built.no_rtmp, from_clap.no_rtmp);
        assert_eq!(built.no_subtitles, from_clap.no_subtitles);
        assert_eq!(built.open, from_clap.open);

        let c = |a: &CommonArgs| {
            (
                a.gguf_dir.clone(),
                a.assume_yes,
                a.language.clone(),
                a.auto_language,
                a.context.clone(),
                a.n_ctx,
                a.n_batch,
                a.n_threads,
                a.cpu_only,
                a.verbose,
            )
        };
        assert_eq!(c(&built.common), c(&from_clap.common));
    }
}

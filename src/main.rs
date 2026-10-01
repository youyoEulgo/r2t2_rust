//! File-oriented CLI for Confucius4-R2T2 speech recognition.
//!
//! Supports one-shot and streaming transcription through llama.cpp. No Python,
//! no conda: the binary links `libllama`/`libmtmd` from `vendor/lib` (or
//! `$R2T2_LIB_DIR`) and needs only the NVIDIA driver at runtime.
//!
//! ```text
//! r2t2 -i audio.wav -o transcript.txt --gguf-dir checkpoints/gguf
//! r2t2 -i audio.wav --stream --chunk-ms 160
//! ```
//!
//! For the WebSocket service, see the `r2t2-server` binary.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{bail, Context as _, Result};
use clap::Parser;

use r2t2::engine::{Engine, EngineConfig};
use r2t2::stream::StreamEngine;
use r2t2::{audio, prompt};

/// Transcribe speech to text with Confucius4-R2T2 (llama.cpp backend).
#[derive(Debug, Parser)]
#[command(name = "r2t2", version, about, long_about = None)]
struct Cli {
    /// Input audio file (WAV; any sample rate or channel count).
    #[arg(short = 'i', long = "input", value_name = "FILE")]
    input: PathBuf,

    /// Write the transcript here instead of stdout.
    #[arg(short = 'o', long = "output", value_name = "FILE")]
    output: Option<PathBuf>,

    /// Directory holding exactly one `mmproj*.gguf` and one other `*.gguf`.
    #[arg(long = "gguf-dir", value_name = "DIR", default_value = "checkpoints/gguf")]
    gguf_dir: PathBuf,

    /// Language hint, e.g. `Chinese` or `English`.
    #[arg(short = 'l', long = "language", value_name = "LANG", default_value = "Chinese")]
    language: String,

    /// Let the model detect the language instead of forcing one.
    #[arg(long = "auto-language", conflicts_with = "language")]
    auto_language: bool,

    /// Optional context / hotword hint.
    #[arg(short = 'c', long = "context", value_name = "TEXT", default_value = "")]
    context: String,

    /// Maximum tokens to generate.
    #[arg(long = "max-tokens", value_name = "N", default_value_t = 256)]
    max_tokens: i32,

    /// Context size for llama.cpp.
    #[arg(long = "n-ctx", value_name = "N", default_value_t = 8192)]
    n_ctx: u32,

    /// Batch size for llama.cpp.
    #[arg(long = "n-batch", value_name = "N", default_value_t = 2048)]
    n_batch: u32,

    /// CPU threads for llama.cpp.
    #[arg(long = "n-threads", value_name = "N", default_value_t = 16)]
    n_threads: i32,

    /// Run on CPU only.
    #[arg(long = "cpu-only")]
    cpu_only: bool,

    /// Transcribe in streaming mode instead of one shot.
    #[arg(long = "stream")]
    stream: bool,

    /// Streaming chunk size in milliseconds.
    #[arg(long = "chunk-ms", value_name = "MS", default_value_t = 160)]
    chunk_ms: u32,

    /// Streaming lookahead for the first chunk, in milliseconds.
    #[arg(long = "lookahead-ms", value_name = "MS", default_value_t = 160)]
    lookahead_ms: u32,

    /// Trailing tokens left unfixed when prompting (rollback window).
    #[arg(long = "unfixed-token-num", value_name = "N", default_value_t = 1)]
    unfixed_token_num: usize,

    /// Print each incremental update to stderr as it is produced.
    #[arg(long = "show-updates")]
    show_updates: bool,

    /// Print progress information to stderr.
    #[arg(short = 'v', long = "verbose")]
    verbose: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: &Cli) -> Result<()> {
    let (model, mmproj) = resolve_gguf(&cli.gguf_dir)?;
    if cli.verbose {
        eprintln!("model  : {}", model.display());
        eprintln!("mmproj : {}", mmproj.display());
    }

    let samples = audio::load_wav_16k_mono(&cli.input)?;
    if samples.is_empty() {
        bail!("no audio samples found in {}", cli.input.display());
    }
    if cli.verbose {
        let secs = samples.len() as f64 / audio::TARGET_SAMPLE_RATE as f64;
        eprintln!("audio  : {} samples ({secs:.2}s)", samples.len());
    }

    let language = (!cli.auto_language).then(|| cli.language.as_str());

    let mut cfg = EngineConfig::new(&model, &mmproj);
    cfg.n_ctx = cli.n_ctx;
    cfg.n_batch = cli.n_batch;
    cfg.n_threads = cli.n_threads;
    cfg.use_gpu = !cli.cpu_only;
    cfg.n_gpu_layers = if cli.cpu_only { 0 } else { -1 };

    if cli.verbose {
        eprintln!("loading model...");
    }

    let text = if cli.stream {
        run_streaming(cli, &cfg, language, &samples)?
    } else {
        run_onetime(cli, &cfg, language, &samples)?
    };

    let text = text.trim();
    match &cli.output {
        Some(path) => {
            std::fs::write(path, format!("{text}\n"))
                .with_context(|| format!("could not write {}", path.display()))?;
            if cli.verbose {
                eprintln!("wrote  : {}", path.display());
            }
        }
        None => println!("{text}"),
    }
    Ok(())
}

/// One-shot transcription: the whole file in a single decode.
fn run_onetime(
    cli: &Cli,
    cfg: &EngineConfig,
    language: Option<&str>,
    samples: &[f32],
) -> Result<String> {
    let prompt = prompt::build(&cli.context, language, "");
    let engine = Engine::load(cfg).context("failed to initialise the llama.cpp engine")?;

    if cli.verbose {
        eprintln!("transcribing...");
    }
    let result = engine
        .transcribe(samples, &prompt, cli.max_tokens)
        .context("transcription failed")?;

    if cli.verbose {
        eprintln!("finish : {}", result.finish_reason);
        eprintln!("tokens : {}", result.token_ids.len());
    }
    Ok(result.text)
}

/// Streaming transcription, feeding the audio in fixed-size steps.
///
/// The first step carries an extra `lookahead_ms` of audio and the chunk size
/// is widened to match, so the opening decode has more context than subsequent
/// ones. The Python driver (`example.py::run_streaming`) does the same.
fn run_streaming(
    cli: &Cli,
    cfg: &EngineConfig,
    language: Option<&str>,
    samples: &[f32],
) -> Result<String> {
    let engine = StreamEngine::load(cfg, cli.max_tokens)
        .context("failed to initialise the streaming engine")?;

    let chunk_size_sec = cli.chunk_ms as f32 / 1000.0;
    let mut state = engine.init_state(
        &cli.context,
        language,
        // 0 means the prefix is used from the very first chunk.
        0,
        cli.unfixed_token_num,
        chunk_size_sec,
    );

    let sr = audio::TARGET_SAMPLE_RATE;
    let step = ((cli.chunk_ms as f32 / 1000.0) * sr as f32).round() as usize;
    let lookahead = ((cli.lookahead_ms as f32 / 1000.0) * sr as f32).round() as usize;

    let mut pos = 0usize;
    let mut first = true;
    let mut updates = 0usize;

    while pos < samples.len() {
        // The opening step reads ahead and decodes a larger chunk.
        let (seg, new_chunk_sec) = if first {
            let end = (pos + step + lookahead).min(samples.len());
            (&samples[pos..end], (step + lookahead) as f32 / sr as f32)
        } else {
            let end = (pos + step).min(samples.len());
            (&samples[pos..end], chunk_size_sec)
        };
        pos += seg.len();
        first = false;

        state.chunk_size_sec = new_chunk_sec;
        state.chunk_size_samples = ((new_chunk_sec * sr as f32).round() as usize).max(1);

        match engine.push(seg, &mut state)? {
            Some((_text, fixed)) => {
                updates += 1;
                if cli.show_updates {
                    eprintln!("text={fixed}");
                }
            }
            None => {
                if cli.show_updates {
                    eprintln!("text=");
                }
            }
        }
    }

    let final_text = engine.finish(&mut state)?;
    if cli.verbose {
        eprintln!("chunks : {}", state.chunk_id);
        eprintln!("updates: {updates}");
    }
    Ok(final_text)
}

/// Find the paired GGUF files in `dir`.
///
/// The projector is the file whose name starts with `mmproj`; the language
/// model is the other one. Requiring exactly one of each catches the common
/// mistake of dropping several quantisations into the same directory, which
/// would otherwise silently pick whichever sorts first.
fn resolve_gguf(dir: &Path) -> Result<(PathBuf, PathBuf)> {
    if !dir.is_dir() {
        bail!("gguf directory not found: {}", dir.display());
    }

    let mut models = Vec::new();
    let mut projectors = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("could not read {}", dir.display()))? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if name.starts_with("mmproj") {
                projectors.push(path);
            } else {
                models.push(path);
            }
        }
    }
    models.sort();
    projectors.sort();

    match (models.len(), projectors.len()) {
        (1, 1) => Ok((models.remove(0), projectors.remove(0))),
        (0, _) => bail!(
            "no language-model GGUF found in {} (expected one non-'mmproj' *.gguf)",
            dir.display()
        ),
        (_, 0) => bail!(
            "no projector GGUF found in {} (expected one 'mmproj*.gguf')",
            dir.display()
        ),
        (m, p) => bail!(
            "expected exactly one mmproj*.gguf and one other *.gguf in {}, found {m} model(s) and {p} projector(s)",
            dir.display()
        ),
    }
}

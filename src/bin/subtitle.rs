//! Subtitle generator: transcribe a media file into an `.srt` file.
//!
//! ```text
//! r2t2-sub   -i movie.mp4 -o movie.srt
//! r2t2-sub   -i movie.mkv -o movie.srt --language Chinese
//! r2t2-sub   -i clip.mp4 --print          # cues to stdout, no file
//! ```
//!
//! The audio is segmented by voice activity detection first, so each cue gets a
//! real timestamp; see [`r2t2::subtitle`] for why that works and what the
//! accuracy limits are.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{bail, Context as _, Result};
use clap::Parser;

use r2t2::engine::EngineConfig;
use r2t2::subtitle::{self, QualityConfig, SplitConfig};
use r2t2::vad::{Sensitivity, VadConfig};

#[derive(Debug, Parser)]
#[command(
    name = "r2t2-sub",
    version,
    about = "Generate subtitles from a video/audio file with Confucius4-R2T2"
)]
struct Cli {
    /// Input media file (video or audio; anything ffmpeg can read).
    #[arg(short = 'i', long = "input", value_name = "FILE")]
    input: PathBuf,

    /// Where to write the subtitle file. Defaults to the input name with `.srt`.
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

    /// Optional context / hotword hint (film title, jargon, names).
    #[arg(short = 'c', long = "context", value_name = "TEXT", default_value = "")]
    context: String,

    /// VAD sensitivity: quality | lowbitrate | aggressive | veryaggressive.
    #[arg(long = "vad-sensitivity", value_name = "LEVEL", default_value = "aggressive")]
    vad_sensitivity: String,

    /// Milliseconds of silence that end a segment.
    #[arg(long = "vad-min-silence-ms", value_name = "MS", default_value_t = 300)]
    vad_min_silence_ms: u64,

    /// Milliseconds of speech that start one.
    #[arg(long = "vad-min-speech-ms", value_name = "MS", default_value_t = 120)]
    vad_min_speech_ms: u64,

    /// Maximum characters per subtitle cue.
    #[arg(long = "max-chars", value_name = "N", default_value_t = 24)]
    max_chars: usize,

    /// Maximum seconds per subtitle cue.
    #[arg(long = "max-seconds", value_name = "SECS", default_value_t = 8.0)]
    max_seconds: f64,

    /// Maximum tokens per segment decode.
    #[arg(long = "max-tokens", value_name = "N", default_value_t = 512)]
    max_tokens: i32,

    /// Context size for llama.cpp.
    #[arg(long = "n-ctx", value_name = "N", default_value_t = 8192)]
    n_ctx: u32,

    /// Run on CPU only.
    #[arg(long = "cpu-only")]
    cpu_only: bool,

    /// Print the cues to stdout instead of writing a file.
    #[arg(long = "print")]
    print: bool,

    /// Repeats needed before a run of identical output counts as stuck.
    #[arg(long = "repeat-threshold", value_name = "N", default_value_t = 5)]
    repeat_threshold: usize,

    /// Keep segments flagged as hallucinated instead of dropping them.
    #[arg(long = "keep-hallucinations")]
    keep_hallucinations: bool,

    /// Print progress to stderr.
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
    if !cli.input.is_file() {
        bail!("input file not found: {}", cli.input.display());
    }

    let (model, mmproj) = resolve_gguf(&cli.gguf_dir)?;
    if cli.verbose {
        eprintln!("model   : {}", model.display());
        eprintln!("mmproj  : {}", mmproj.display());
        eprintln!("input   : {}", cli.input.display());
    }

    let language = (!cli.auto_language).then(|| cli.language.as_str());

    let vad = VadConfig {
        sensitivity: Sensitivity::parse(&cli.vad_sensitivity)?,
        min_silence_frames: (cli.vad_min_silence_ms / 20).max(1) as usize,
        min_speech_frames: (cli.vad_min_speech_ms / 20).max(1) as usize,
        ..Default::default()
    };
    let split = SplitConfig {
        max_chars: cli.max_chars,
        max_seconds: cli.max_seconds,
        ..Default::default()
    };
    let quality = QualityConfig {
        repeat_threshold: cli.repeat_threshold,
        keep_hallucinations: cli.keep_hallucinations,
        ..Default::default()
    };

    let mut cfg = EngineConfig::new(&model, &mmproj);
    cfg.n_ctx = cli.n_ctx;
    cfg.use_gpu = !cli.cpu_only;
    cfg.n_gpu_layers = if cli.cpu_only { 0 } else { -1 };

    if cli.verbose {
        eprintln!("loading : model...");
    }
    let engine = subtitle::load_engine(&cfg, cli.max_tokens)?;

    if cli.verbose {
        eprintln!("decoding: audio...");
    }
    let samples = r2t2::media::load_media_16k_mono(&cli.input)?;
    let duration = samples.len() as f64 / r2t2::audio::TARGET_SAMPLE_RATE as f64;
    if samples.is_empty() {
        bail!("no audio could be decoded from {}", cli.input.display());
    }
    if cli.verbose {
        eprintln!("audio   : {duration:.1}s");
    }

    if cli.verbose {
        eprintln!("segmenting and transcribing...");
    }
    let result = subtitle::build(
        &engine,
        &samples,
        language,
        &cli.context,
        vad,
        &split,
        &quality,
        cli.max_tokens,
    )?;

    if cli.verbose {
        eprintln!(
            "segments: {} ({} produced no text)",
            result.segments, result.empty_segments
        );
        eprintln!("cues    : {}", result.cues.len());
    }

    let srt = subtitle::to_srt(&result.cues);
    if cli.print {
        print!("{srt}");
        return Ok(());
    }

    let out_path = cli.output.clone().unwrap_or_else(|| {
        let mut p = cli.input.clone();
        p.set_extension("srt");
        p
    });
    std::fs::write(&out_path, &srt)
        .with_context(|| format!("could not write {}", out_path.display()))?;

    // Brief summary so a batch run shows something useful.
    let first = result
        .cues
        .first()
        .map(|c| subtitle::srt_timestamp(c.start))
        .unwrap_or_else(|| "-".into());
    let last = result
        .cues
        .last()
        .map(|c| subtitle::srt_timestamp(c.end))
        .unwrap_or_else(|| "-".into());
    println!(
        "{} -> {} ({} cues, {} segments, span {} - {})",
        cli.input.display(),
        out_path.display(),
        result.cues.len(),
        result.segments,
        first,
        last
    );
    Ok(())
}

/// Find the paired GGUF files, same rule as the other binaries.
fn resolve_gguf(dir: &std::path::Path) -> Result<(PathBuf, PathBuf)> {
    if !dir.is_dir() {
        bail!("gguf directory not found: {}", dir.display());
    }
    let mut models = Vec::new();
    let mut projectors = Vec::new();
    for entry in std::fs::read_dir(dir)? {
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
        (0, _) => bail!("no language-model GGUF found in {}", dir.display()),
        (_, 0) => bail!("no projector GGUF found in {}", dir.display()),
        (m, p) => bail!(
            "expected exactly one mmproj*.gguf and one other *.gguf in {}, found {m} model(s) and {p} projector(s)",
            dir.display()
        ),
    }
}

//! `r2t2 subtitle` — turn a video or audio file into an `.srt`.

use std::path::PathBuf;

use anyhow::{bail, Context as _, Result};
use clap::Args;

use crate::audio::TARGET_SAMPLE_RATE;
use crate::cli::CommonArgs;
use crate::media;
use crate::subtitle::{self, QualityConfig, SplitConfig};
use crate::vad::{Sensitivity, VadConfig};

#[derive(Debug, Args)]
pub struct SubtitleArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    /// Input media file (video or audio; anything ffmpeg can read).
    #[arg(short = 'i', long = "input", value_name = "FILE")]
    pub input: PathBuf,

    /// Where to write the subtitle file. Defaults to the input name with `.srt`.
    #[arg(short = 'o', long = "output", value_name = "FILE")]
    pub output: Option<PathBuf>,

    /// Maximum tokens per segment decode.
    #[arg(long = "max-tokens", value_name = "N", default_value_t = 512)]
    pub max_tokens: i32,

    // ---- segmentation -----------------------------------------------------
    /// VAD sensitivity: quality | lowbitrate | aggressive | veryaggressive.
    #[arg(long = "vad-sensitivity", value_name = "LEVEL", default_value = "aggressive")]
    pub vad_sensitivity: String,

    /// Milliseconds of silence that end a segment.
    #[arg(long = "vad-min-silence-ms", value_name = "MS", default_value_t = 300)]
    pub vad_min_silence_ms: u64,

    /// Milliseconds of speech that start one.
    #[arg(long = "vad-min-speech-ms", value_name = "MS", default_value_t = 120)]
    pub vad_min_speech_ms: u64,

    // ---- cue shape --------------------------------------------------------
    /// Maximum characters per subtitle cue.
    #[arg(long = "max-chars", value_name = "N", default_value_t = 24)]
    pub max_chars: usize,

    /// Maximum seconds per subtitle cue.
    #[arg(long = "max-seconds", value_name = "SECS", default_value_t = 8.0)]
    pub max_seconds: f64,

    // ---- quality guards ---------------------------------------------------
    /// Repeats before a run of identical output counts as a stuck decoder.
    #[arg(long = "repeat-threshold", value_name = "N", default_value_t = 5)]
    pub repeat_threshold: usize,

    /// Keep segments flagged as hallucinated instead of dropping them.
    #[arg(long = "keep-hallucinations")]
    pub keep_hallucinations: bool,

    /// Print the cues to stdout instead of writing a file.
    #[arg(long = "print")]
    pub print: bool,
}

pub fn run(args: &SubtitleArgs) -> Result<()> {
    if !args.input.is_file() {
        bail!("input file not found: {}", args.input.display());
    }

    let (model, mmproj) = args.common.resolve_model()?;
    if args.common.verbose {
        eprintln!("input  : {}", args.input.display());
    }

    let vad = VadConfig {
        sensitivity: Sensitivity::parse(&args.vad_sensitivity)?,
        min_silence_frames: (args.vad_min_silence_ms / 20).max(1) as usize,
        min_speech_frames: (args.vad_min_speech_ms / 20).max(1) as usize,
        ..Default::default()
    };
    let split = SplitConfig {
        max_chars: args.max_chars,
        max_seconds: args.max_seconds,
        ..Default::default()
    };
    let quality = QualityConfig {
        repeat_threshold: args.repeat_threshold,
        keep_hallucinations: args.keep_hallucinations,
        ..Default::default()
    };

    let cfg = args.common.engine_config(model, mmproj);
    if args.common.verbose {
        eprintln!("loading model...");
    }
    let engine = subtitle::load_engine(&cfg, args.max_tokens)?;

    if args.common.verbose {
        eprintln!("decoding audio...");
    }
    let samples = media::load_media_16k_mono(&args.input)?;
    if samples.is_empty() {
        bail!("no audio could be decoded from {}", args.input.display());
    }
    if args.common.verbose {
        let secs = samples.len() as f64 / TARGET_SAMPLE_RATE as f64;
        eprintln!("audio  : {secs:.1}s");
    }

    if args.common.verbose {
        eprintln!("segmenting and transcribing...");
    }
    let result = subtitle::build(
        &engine,
        &samples,
        args.common.forced_language(),
        &args.common.context,
        vad,
        &split,
        &quality,
        args.max_tokens,
    )?;

    if args.common.verbose {
        eprintln!(
            "segments: {} ({} dropped)",
            result.segments, result.empty_segments
        );
        eprintln!("cues    : {}", result.cues.len());
    }

    let srt = subtitle::to_srt(&result.cues);
    if args.print {
        print!("{srt}");
        return Ok(());
    }

    let out_path = args.output.clone().unwrap_or_else(|| {
        let mut p = args.input.clone();
        p.set_extension("srt");
        p
    });
    std::fs::write(&out_path, &srt)
        .with_context(|| format!("could not write {}", out_path.display()))?;

    // A one-line summary, so a batch run shows something useful.
    let stamp = |c: Option<&subtitle::Cue>| {
        c.map(|c| subtitle::srt_timestamp(c.start))
            .unwrap_or_else(|| "-".into())
    };
    let last = result
        .cues
        .last()
        .map(|c| subtitle::srt_timestamp(c.end))
        .unwrap_or_else(|| "-".into());
    println!(
        "{} -> {} ({} cues, {} segments, span {} - {})",
        args.input.display(),
        out_path.display(),
        result.cues.len(),
        result.segments,
        stamp(result.cues.first()),
        last
    );
    Ok(())
}

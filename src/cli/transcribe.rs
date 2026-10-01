// Copyright 2026 youyoEulgo
// SPDX-License-Identifier: Apache-2.0

//! `r2t2 transcribe` — turn an audio or video file into text or subtitles.

use std::path::PathBuf;

use anyhow::{Context as _, Result, bail};
use clap::{Args, ValueEnum};

use crate::audio::TARGET_SAMPLE_RATE;
use crate::cli::CommonArgs;
use crate::engine::Engine;
use crate::media;
use crate::prompt;
use crate::subtitle::{self, QualityConfig, SplitConfig};
use crate::vad::{Sensitivity, VadConfig};

/// What to write to stdout or the output file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Format {
    /// SubRip subtitles, with timings taken from voice activity detection.
    Srt,
    /// Plain text, with segment boundaries joined into readable lines.
    Txt,
}

#[derive(Debug, Args)]
pub struct TranscribeArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    /// Input media file. WAV is read in process; anything else goes through
    /// ffmpeg, so video containers work directly.
    #[arg(short = 'i', long = "input", value_name = "FILE")]
    pub input: PathBuf,

    /// Write the result here instead of stdout.
    #[arg(short = 'o', long = "output", value_name = "FILE")]
    pub output: Option<PathBuf>,

    /// What to produce. Subtitles are the default because they carry the
    /// timings, which plain text cannot recover.
    #[arg(long = "format", value_enum, default_value_t = Format::Srt)]
    pub format: Format,

    /// Maximum tokens per segment decode.
    #[arg(long = "max-tokens", value_name = "N", default_value_t = 512)]
    pub max_tokens: i32,

    /// Stream the audio in chunks rather than decoding it in one pass.
    ///
    /// The text is the same; this exists to exercise the streaming path, and
    /// only applies to `--format txt`.
    #[arg(long = "stream")]
    pub stream: bool,

    /// Streaming chunk size in milliseconds.
    #[arg(long = "chunk-ms", value_name = "MS", default_value_t = 160)]
    pub chunk_ms: u32,

    /// Streaming lookahead for the first chunk, in milliseconds.
    #[arg(long = "lookahead-ms", value_name = "MS", default_value_t = 160)]
    pub lookahead_ms: u32,

    /// Trailing tokens left unfixed when prompting (rollback window).
    #[arg(long = "unfixed-token-num", value_name = "N", default_value_t = 1)]
    pub unfixed_token_num: usize,

    /// Print each incremental update to stderr as it is produced.
    #[arg(long = "show-updates")]
    pub show_updates: bool,

    // ---- segmentation -----------------------------------------------------
    /// VAD sensitivity: quality | lowbitrate | aggressive | veryaggressive.
    #[arg(
        long = "vad-sensitivity",
        value_name = "LEVEL",
        default_value = "aggressive"
    )]
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
}

pub fn run(args: &TranscribeArgs) -> Result<()> {
    if !args.input.is_file() {
        bail!("input file not found: {}", args.input.display());
    }

    let text = match args.format {
        Format::Txt => transcribe_text(args)?,
        Format::Srt => transcribe_subtitles(args)?,
    };

    match &args.output {
        Some(path) => {
            std::fs::write(path, &text)
                .with_context(|| format!("could not write {}", path.display()))?;
            if args.common.verbose {
                eprintln!("wrote  : {}", path.display());
            }
        }
        None => print!("{text}"),
    }
    Ok(())
}

/// Plain text: decode each detected segment and join the results.
fn transcribe_text(args: &TranscribeArgs) -> Result<String> {
    if args.stream {
        return stream_text(args);
    }

    let samples = load_audio(args)?;
    let engine = load_engine(args)?;

    let segments = subtitle::detect_segments(&samples, vad_config(args)?)?;
    if args.common.verbose {
        eprintln!("segments: {}", segments.len());
    }

    let mut pieces = Vec::with_capacity(segments.len());
    for (i, seg) in segments.iter().enumerate() {
        pieces.push(speech_of_segment(&engine, &samples, seg, args)?);
        if args.common.verbose {
            eprint!("\r  {}/{}", i + 1, segments.len());
        }
    }
    if args.common.verbose {
        eprintln!();
    }

    Ok(format!("{}\n", join_segments(&pieces)))
}

/// Streaming text, for exercising the chunked path.
fn stream_text(args: &TranscribeArgs) -> Result<String> {
    use crate::audio::TARGET_SAMPLE_RATE as SR;
    use crate::stream::StreamEngine;

    let (model, mmproj) = args.common.resolve_model()?;
    let cfg = args.common.engine_config(model, mmproj);
    let samples = load_audio(args)?;

    let engine = StreamEngine::load(&cfg, args.max_tokens)
        .context("failed to initialise the streaming engine")?;

    let chunk_sec = args.chunk_ms as f32 / 1000.0;
    let mut state = engine.init_state(
        &args.common.context,
        args.common.forced_language(),
        0,
        args.unfixed_token_num,
        chunk_sec,
    );

    let step = (chunk_sec * SR as f32).round() as usize;
    let lookahead = ((args.lookahead_ms as f32 / 1000.0) * SR as f32).round() as usize;

    let mut pos = 0usize;
    let mut first = true;
    let mut updates = 0usize;

    while pos < samples.len() {
        let (seg, secs) = if first {
            let end = (pos + step + lookahead).min(samples.len());
            (&samples[pos..end], (step + lookahead) as f32 / SR as f32)
        } else {
            let end = (pos + step).min(samples.len());
            (&samples[pos..end], chunk_sec)
        };
        pos += seg.len();
        first = false;

        state.chunk_size_sec = secs;
        state.chunk_size_samples = ((secs * SR as f32).round() as usize).max(1);

        match engine.push(seg, &mut state)? {
            Some((_text, fixed)) => {
                updates += 1;
                if args.show_updates {
                    eprintln!("text={fixed}");
                }
            }
            None => {
                if args.show_updates {
                    eprintln!("text=");
                }
            }
        }
    }

    let final_text = engine.finish(&mut state)?;
    if args.common.verbose {
        eprintln!("chunks : {}", state.chunk_id);
        eprintln!("updates: {updates}");
    }
    Ok(format!("{}\n", join_segments(&[final_text])))
}

/// Subtitles: decode each detected segment into cues.
fn transcribe_subtitles(args: &TranscribeArgs) -> Result<String> {
    let samples = load_audio(args)?;
    let engine = load_engine(args)?;
    let vad = vad_config(args)?;
    let split = split_config(args);
    let quality = quality_config(args);

    let segments = subtitle::detect_segments(&samples, vad)?;
    if args.common.verbose {
        eprintln!("segments: {}", segments.len());
    }

    let mut cues = Vec::new();
    let mut dropped = 0usize;

    for (i, seg) in segments.iter().enumerate() {
        let text = speech_of_segment(&engine, &samples, seg, args)?;
        if crate::quality::is_degenerate(&text) {
            dropped += 1;
            continue;
        }
        if !quality.keep_hallucinations
            && crate::quality::detect_hallucination(
                &text,
                quality.repeat_threshold,
                quality.max_pattern_len,
                quality.tail_check_len,
            )
            .is_some()
        {
            dropped += 1;
            continue;
        }
        let (t0, t1) = subtitle::segment_times(seg);
        cues.extend(subtitle::split_cue(t0, t1, &text, &split));

        if args.common.verbose {
            eprint!("\r  {}/{}", i + 1, segments.len());
        }
    }
    if args.common.verbose {
        eprintln!();
        if dropped > 0 {
            eprintln!("dropped : {dropped} segment(s) with no usable speech");
        }
    }

    let cues = subtitle::merge_adjacent(cues, &split);
    if args.common.verbose {
        eprintln!("cues    : {}", cues.len());
    }
    Ok(subtitle::to_srt(&cues))
}

/// Load the input as mono 16 kHz, via ffmpeg for anything but WAV.
fn load_audio(args: &TranscribeArgs) -> Result<Vec<f32>> {
    if args.common.verbose {
        eprintln!("decoding: {}", args.input.display());
    }
    let samples = media::load_media_16k_mono(&args.input)?;
    if samples.is_empty() {
        bail!("no audio could be decoded from {}", args.input.display());
    }
    if args.common.verbose {
        let secs = samples.len() as f64 / TARGET_SAMPLE_RATE as f64;
        eprintln!("audio   : {secs:.1}s");
    }
    Ok(samples)
}

fn load_engine(args: &TranscribeArgs) -> Result<Engine> {
    let (model, mmproj) = args.common.resolve_model()?;
    let cfg = args.common.engine_config(model, mmproj);
    if args.common.verbose {
        eprintln!("loading : model...");
    }
    Engine::load(&cfg)
}

fn vad_config(args: &TranscribeArgs) -> Result<VadConfig> {
    Ok(VadConfig {
        sensitivity: Sensitivity::parse(&args.vad_sensitivity)?,
        min_silence_frames: (args.vad_min_silence_ms / 20).max(1) as usize,
        min_speech_frames: (args.vad_min_speech_ms / 20).max(1) as usize,
        ..Default::default()
    })
}

fn split_config(args: &TranscribeArgs) -> SplitConfig {
    SplitConfig {
        max_chars: args.max_chars,
        max_seconds: args.max_seconds,
        ..Default::default()
    }
}

fn quality_config(args: &TranscribeArgs) -> QualityConfig {
    QualityConfig {
        repeat_threshold: args.repeat_threshold,
        keep_hallucinations: args.keep_hallucinations,
        ..Default::default()
    }
}

/// Transcribe one detected segment, with the repetition guard applied.
pub(crate) fn speech_of_segment(
    engine: &Engine,
    samples: &[f32],
    seg: &crate::vad::Segment,
    args: &TranscribeArgs,
) -> Result<String> {
    use crate::vad::FRAME_SAMPLES;

    let start = (seg.start_frame as usize * FRAME_SAMPLES).min(samples.len());
    let end = (start + seg.frames as usize * FRAME_SAMPLES).min(samples.len());
    if end <= start {
        return Ok(String::new());
    }

    let p = prompt::build(&args.common.context, args.common.forced_language(), "");
    let result = engine.transcribe(&samples[start..end], &p, args.max_tokens)?;
    let (_, text) = crate::stream::parse_asr_output(&result.text, args.common.forced_language());

    Ok(crate::quality::fix_repetitions(
        text.trim(),
        args.repeat_threshold,
    ))
}

/// Join segment texts into a readable transcript.
///
/// Segments arrive without trailing punctuation when the speaker runs on, so
/// bare concatenation reads as one long word. A space is inserted between two
/// Latin runs; CJK is left alone, since Chinese does not space between
/// characters.
pub fn join_segments(parts: &[String]) -> String {
    let mut out = String::new();
    for part in parts {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if !out.is_empty() {
            let prev_latin = out
                .chars()
                .last()
                .is_some_and(|c| c.is_ascii_alphanumeric());
            let next_latin = part
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphanumeric());
            if prev_latin && next_latin {
                out.push(' ');
            }
        }
        out.push_str(part);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_latin_segments_with_a_space() {
        let parts = vec!["hello".to_string(), "world".to_string()];
        assert_eq!(join_segments(&parts), "hello world");
    }

    #[test]
    fn does_not_space_between_cjk() {
        let parts = vec!["你好".to_string(), "世界".to_string()];
        assert_eq!(join_segments(&parts), "你好世界");
    }

    #[test]
    fn skips_empty_segments() {
        let parts = vec!["第一句".to_string(), String::new(), "第二句".to_string()];
        assert_eq!(join_segments(&parts), "第一句第二句");
    }
}

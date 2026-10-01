//! `r2t2 mux` — combine a video and a subtitle file into one Matroska file.
//!
//! A separate subcommand rather than a flag on `transcribe`, because it does no
//! recognition at all: it takes an existing subtitle file, which may have been
//! corrected elsewhere, and packages it with the video.
//!
//! Video and audio are **stream-copied**, never re-encoded, so this is a
//! remuxing job measured in seconds regardless of length or codec.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context as _, Result};
use clap::Args;

use crate::media;

#[derive(Debug, Args)]
pub struct MuxArgs {
    /// The video to carry the subtitles.
    #[arg(long = "video", value_name = "FILE")]
    pub video: PathBuf,

    /// The subtitle file to embed (SRT).
    #[arg(long = "subtitle", value_name = "FILE")]
    pub subtitle: PathBuf,

    /// Where to write the result. Defaults to the video's name with `.mkv`.
    #[arg(long = "output", value_name = "FILE")]
    pub output: Option<PathBuf>,

    /// Language tag for the subtitle track, e.g. `Chinese` or `eng`.
    ///
    /// Players select tracks by this, so an untagged subtitle shows up as
    /// "unknown".
    #[arg(short = 'l', long = "language", value_name = "LANG", default_value = "Chinese")]
    pub language: String,

    /// Print progress information to stderr.
    #[arg(long = "verbose")]
    pub verbose: bool,
}

pub fn run(args: &MuxArgs) -> Result<()> {
    if !args.video.is_file() {
        bail!("video not found: {}", args.video.display());
    }
    if !args.subtitle.is_file() {
        bail!("subtitle file not found: {}", args.subtitle.display());
    }
    if !media::ffmpeg_available() {
        bail!("ffmpeg is required to mux but was not found on PATH");
    }

    // Refuse to package subtitles with something that has no video track: the
    // result would be an audio file with a pointless subtitle stream, which is
    // almost certainly a mistake.
    match media::has_video_stream(&args.video) {
        Ok(true) => {}
        Ok(false) => bail!(
            "{} has no video track; mux is for combining subtitles with video",
            args.video.display()
        ),
        Err(err) => {
            // ffprobe missing or the file is unreadable; let ffmpeg report it.
            if args.verbose {
                eprintln!("warning: could not inspect the input ({err}); continuing");
            }
        }
    }

    let output = args.output.clone().unwrap_or_else(|| {
        let mut p = args.video.clone();
        p.set_extension("mkv");
        p
    });
    if output == args.video {
        bail!("output would overwrite the input video; pass --output");
    }

    if args.verbose {
        eprintln!("video   : {}", args.video.display());
        eprintln!("subtitle: {}", args.subtitle.display());
        eprintln!("output  : {}", output.display());
        eprintln!("language: {}", args.language);
    }

    media::mux_subtitles(&args.video, &args.subtitle, &output, &args.language)?;

    // Report what went in, so a silent success is still verifiable.
    let size = std::fs::metadata(&output)
        .with_context(|| format!("wrote {} but cannot read it back", output.display()))?
        .len();
    let tracks = count_tracks(&output);
    println!(
        "{} -> {} ({}, {})",
        args.video.display(),
        output.display(),
        human_size(size),
        tracks
    );
    Ok(())
}

/// Describe the streams in the result, for the summary line.
fn count_tracks(path: &Path) -> String {
    match media::stream_summary(path) {
        Ok(s) if !s.is_empty() => s,
        _ => "tracks unknown".to_string(),
    }
}

fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

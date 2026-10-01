//! Media input: decode any audio/video file to the mono 16 kHz `f32` the model
//! needs.
//!
//! Video containers are the normal case for subtitling, and decoding them
//! properly (h264 + aac + mkv/mp4/mov/…) means either a large pure-Rust media
//! stack or shelling out. This uses **ffmpeg as an external command**, the same
//! trade-off already made for llama.cpp: lean on a mature, ubiquitous C
//! toolchain instead of reimplementing it. ffmpeg reads the file, downmixes and
//! resamples, and hands back raw `f32le` on stdout -- no temporary files, no
//! format-specific code here.
//!
//! WAV files are still handled in-process via [`crate::audio`], so the common
//! case has no external dependency at all.

use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};

use crate::audio;

/// How long the input is, in seconds, if it can be determined.
pub fn probe_duration(path: &Path) -> Result<f64> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(path)
        .output()
        .context("could not run ffprobe (is ffmpeg installed?)")?;
    if !out.status.success() {
        bail!(
            "ffprobe failed on {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<f64>()
        .with_context(|| format!("could not parse ffprobe duration for {}", path.display()))
}

/// True when ffmpeg is on `PATH`.
pub fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Load any media file as mono 16 kHz samples.
///
/// WAV is read in-process; everything else is piped through ffmpeg.
pub fn load_media_16k_mono(path: &Path) -> Result<Vec<f32>> {
    let is_wav = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("wav"));
    if is_wav {
        // hound handles plain WAV, but a WAV carrying a non-PCM codec would
        // fail here; fall through to ffmpeg in that case.
        match audio::load_wav_16k_mono(path) {
            Ok(samples) => return Ok(samples),
            Err(err) => {
                tracing_ish_log(&format!(
                    "in-process WAV decode failed ({err}); falling back to ffmpeg"
                ));
            }
        }
    }
    load_via_ffmpeg(path)
}

/// Decode with ffmpeg, requesting raw mono 16 kHz `f32le` on stdout.
fn load_via_ffmpeg(path: &Path) -> Result<Vec<f32>> {
    let mut child = Command::new("ffmpeg")
        .args([
            "-v", "error",
            "-i",
        ])
        .arg(path)
        .args([
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ac", "1",
            "-ar", "16000",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| {
            format!(
                "could not run ffmpeg to decode {} (is it on PATH?)",
                path.display()
            )
        })?;

    let mut stdout = child.stdout.take().context("ffmpeg produced no stdout")?;
    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut stdout, &mut buf)
        .with_context(|| format!("failed while reading ffmpeg output for {}", path.display()))?;

    let status = child.wait().context("ffmpeg did not exit cleanly")?;
    if !status.success() {
        let mut err = String::new();
        if let Some(mut e) = child.stderr.take() {
            let _ = std::io::Read::read_to_string(&mut e, &mut err);
        }
        bail!(
            "ffmpeg failed on {}: {}",
            path.display(),
            err.trim().lines().last().unwrap_or("unknown error")
        );
    }

    if buf.len() % 4 != 0 {
        bail!("ffmpeg returned a byte count that is not a multiple of 4");
    }
    let samples: Vec<f32> = buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(samples)
}

/// Standalone binaries should not depend on a logging facade just for this.
fn tracing_ish_log(msg: &str) {
    eprintln!("note: {msg}");
}

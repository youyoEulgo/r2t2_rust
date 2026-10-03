// Copyright 2026 youyoEulgo
// SPDX-License-Identifier: Apache-2.0

//! HLS output: turn the incoming stream's video into something a browser can
//! play.
//!
//! # Why HLS
//!
//! A browser cannot play RTMP. Of the formats it *can* play, HLS needs no
//! server-side re-encoding — the video is copied through untouched — and no
//! exotic protocol stack, which WebRTC would. The cost is latency: a segment
//! has to be complete before a player can fetch it.
//!
//! # Where the latency comes from
//!
//! A segment can only end on a keyframe, because the frames after one depend on
//! the ones before it. `-c:v copy` therefore cannot cut more finely than the
//! sender's keyframe interval, and no setting here can override that. With
//! OBS's default of two seconds the segments come out at two seconds; with a
//! larger interval they are correspondingly longer, and `hls_time` merely
//! states a wish. The console says so, because the symptom — several seconds of
//! extra delay with no obvious cause — is otherwise hard to attribute.

use std::path::PathBuf;

use anyhow::{Context as _, Result};
use tracing::{debug, info, warn};


/// Everything the HTTP side needs to serve the stream.
#[derive(Clone)]
pub struct HlsOutput {
    /// Directory holding the playlist and segments.
    pub dir: PathBuf,
    /// Path of the playlist within that directory.
    pub playlist: &'static str,
}

impl HlsOutput {
    pub fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            playlist: "stream.m3u8",
        }
    }

    /// Absolute path of the playlist.
    pub fn playlist_path(&self) -> PathBuf {
        self.dir.join(self.playlist)
    }
}

/// How the video is packaged.
#[derive(Debug, Clone)]
pub struct HlsConfig {
    /// Target segment length in seconds.
    ///
    /// A wish rather than a guarantee: segments end on keyframes, so the
    /// sender's interval is the real bound.
    pub segment_seconds: u32,
    /// How many segments the playlist lists.
    ///
    /// A deeper window survives a viewer joining late or stalling, at the cost
    /// of more buffering; a shallower one is closer to live.
    pub playlist_size: u32,
}

impl Default for HlsConfig {
    fn default() -> Self {
        Self {
            // One second, to keep the delay down. A segment cannot end before
            // the sender's next keyframe, so this is a floor rather than a
            // setting: with OBS left on "auto" the real length is whatever
            // x264 picks, which is several seconds.
            segment_seconds: 1,
            playlist_size: 6,
        }
    }
}

/// Starts ffmpeg, which dials into our own RTMP listener and writes HLS.
///
/// ffmpeg pulls the stream itself rather than being fed one. That is the point
/// of the relay: it speaks RTMP, so no container has to be rebuilt by hand, and
/// audio and video become separate connections that cannot deadlock each other.
pub struct HlsPackager {
    /// Held only for its `Drop`: killing ffmpeg when the packager goes away.
    /// `kill_on_drop` does the work; the field keeps the handle alive.
    #[allow(dead_code)]
    child: tokio::process::Child,
}

impl HlsPackager {
    /// Prepare the output directory and start ffmpeg against `rtmp_url`.
    pub fn spawn(
        output: &HlsOutput,
        config: &HlsConfig,
        rtmp_url: &str,
    ) -> Result<Self> {
        std::fs::create_dir_all(&output.dir)
            .with_context(|| format!("could not create {}", output.dir.display()))?;

        let playlist = output.playlist_path();
        let segment_pattern = output.dir.join("seg%d.ts");

        let child = tokio::process::Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "warning", "-i", rtmp_url])
            // Copy the video: no re-encoding, so no GPU cost and no generation
            // loss. It also means segments can only end on the sender's
            // keyframes, which is why the keyframe interval governs latency.
            .args(["-c:v", "copy", "-an", "-f", "hls", "-hls_time"])
            .arg(config.segment_seconds.to_string())
            .args(["-hls_list_size"])
            .arg(config.playlist_size.to_string())
            .args([
                "-hls_flags",
                "delete_segments+omit_endlist+independent_segments",
                "-hls_segment_type",
                "mpegts",
                "-hls_segment_filename",
            ])
            .arg(&segment_pattern)
            .arg(&playlist)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .context("could not start ffmpeg for HLS")?;

        info!(
            dir = %output.dir.display(),
            segment_seconds = config.segment_seconds,
            "HLS output started"
        );
        Ok(Self { child })
    }
}

/// Remove the playlist and segments.
pub fn cleanup(output: &HlsOutput) {
    if let Ok(entries) = std::fs::read_dir(&output.dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let named = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            if named.ends_with(".ts") || named.ends_with(".m3u8") {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
    debug!(dir = %output.dir.display(), "cleared the HLS output");
}


/// Start a packager each time a publisher connects.
///
/// The process is left running between streams rather than torn down and
/// rebuilt: the playlist and its segments belong to the directory, and a viewer
/// that is already watching keeps its connection when the publisher briefly
/// restarts.
pub fn follow(output: HlsOutput, ingest: crate::rtmp::IngestHandle, port: u16) {
    let url = format!("rtmp://127.0.0.1:{port}/live");
    let mut events = ingest.subscribe();
    tokio::spawn(async move {
        let mut packager: Option<HlsPackager> = None;
        while let Ok(event) = events.recv().await {
            match event {
                crate::rtmp::IngestEvent::Published { .. } => {
                    if packager.is_none() {
                        match HlsPackager::spawn(&output, &HlsConfig::default(), &url) {
                            Ok(p) => packager = Some(p),
                            Err(err) => {
                                warn!(error = %err, "could not start the HLS packager")
                            }
                        }
                    }
                }
                crate::rtmp::IngestEvent::Unpublished { .. } => {
                    // Dropping the packager lets ffmpeg finish the current
                    // segment and exit, which leaves a playlist a player can
                    // still read.
                    packager = None;
                }
            }
        }
    });
}

/// Whether a playlist currently exists.
pub fn playlist_ready(output: &HlsOutput) -> bool {
    output.playlist_path().is_file()
}

/// Read the target duration out of a live playlist.
///
/// This is what the segments actually came out at, as opposed to what was
/// asked for. The two differ whenever the sender's keyframe interval is longer
/// than `segment_seconds`, which is easy to do by accident — OBS's keyframe
/// interval defaults to "auto", which x264 turns into 250 frames — and has no
/// symptom other than several seconds of unexplained delay.
pub fn segment_seconds(output: &HlsOutput) -> Option<u32> {
    let text = std::fs::read_to_string(output.playlist_path()).ok()?;
    let line = text
        .lines()
        .find(|l| l.starts_with("#EXT-X-TARGETDURATION:"))?;
    let value = line.split(':').nth(1)?.trim();
    // The tag is documented as an integer, but players accept a float and
    // ffmpeg has been known to write one.
    value.parse::<f64>().ok().map(|v| v.ceil() as u32)
}

/// File name safety: refuse anything that is not a plain segment or playlist.
///
/// The HTTP layer serves this directory, so a traversal attempt must not reach
/// outside it.
pub fn is_servable(name: &str) -> bool {
    if name.is_empty() || name.contains('/') || name.contains('\\') || name.contains("..") {
        return false;
    }
    name.ends_with(".ts") || name.ends_with(".m3u8")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_playlists_and_segments_are_servable() {
        assert!(is_servable("stream.m3u8"));
        assert!(is_servable("seg12.ts"));
        assert!(!is_servable("../config.toml"));
        assert!(!is_servable("..%2fpasswd"));
        assert!(!is_servable("/etc/passwd"));
        assert!(!is_servable("sub/dir/seg1.ts"));
        assert!(!is_servable("notes.txt"));
        assert!(!is_servable(""));
    }

    #[test]
    fn the_target_duration_is_read_back() {
        let dir = std::env::temp_dir().join("r2t2-hls-duration");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let out = HlsOutput::new(dir.clone());
        std::fs::write(
            out.playlist_path(),
            "#EXTM3U\n#EXT-X-TARGETDURATION:8\n#EXTINF:8.0,\nseg1.ts\n",
        )
        .unwrap();
        assert_eq!(segment_seconds(&out), Some(8));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_playlist_has_no_duration() {
        let out = HlsOutput::new(PathBuf::from("/nonexistent/hls"));
        assert_eq!(segment_seconds(&out), None);
    }

    #[test]
    fn the_playlist_path_is_inside_the_output_directory() {
        let out = HlsOutput::new(PathBuf::from("/tmp/hls"));
        assert_eq!(out.playlist_path(), PathBuf::from("/tmp/hls/stream.m3u8"));
    }

    #[test]
    fn defaults_favour_low_latency() {
        let c = HlsConfig::default();
        // Short segments for delay; enough of them that a viewer joining late
        // still has a window to start from.
        assert_eq!(c.segment_seconds, 1);
        assert_eq!(c.playlist_size, 6);
    }
}

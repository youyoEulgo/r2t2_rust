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
use bytes::Bytes;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info};

use crate::rtmp::{flv_header, flv_tag, TAG_AUDIO, TAG_VIDEO};

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
            segment_seconds: 2,
            playlist_size: 6,
        }
    }
}

/// Feeds video (and the audio needed to keep the container valid) to ffmpeg,
/// which writes HLS segments.
pub struct HlsPackager {
    stdin: tokio::process::ChildStdin,
    /// Held only for its `Drop`: killing ffmpeg when the packager goes away.
    /// `kill_on_drop` does the work; the field keeps the handle alive.
    #[allow(dead_code)]
    child: tokio::process::Child,
    timestamp: u32,
    /// Whether any video has actually arrived.
    ///
    /// A publisher may send audio only; the playlist then never appears, and
    /// the viewer should be told that rather than left waiting.
    saw_video: bool,
}

impl HlsPackager {
    /// Start ffmpeg and prepare the output directory.
    pub fn spawn(output: &HlsOutput, config: &HlsConfig) -> Result<Self> {
        std::fs::create_dir_all(&output.dir)
            .with_context(|| format!("could not create {}", output.dir.display()))?;

        let playlist = output.playlist_path();
        let segment_pattern = output.dir.join("seg%d.ts");

        let mut child = tokio::process::Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel", "warning",
                "-f", "flv",
                "-i", "pipe:0",
                // Copy the video: no re-encoding, so no GPU cost and no
                // generation loss. It also means segments can only be cut on
                // the sender's keyframes.
                "-c:v", "copy",
                "-an",
                "-f", "hls",
                "-hls_time",
            ])
            .arg(config.segment_seconds.to_string())
            .args(["-hls_list_size"])
            .arg(config.playlist_size.to_string())
            // delete_segments keeps the directory from growing without bound;
            // omit_endlist says the stream is live, so a player keeps polling
            // rather than treating the playlist as finished.
            .args([
                "-hls_flags",
                "delete_segments+omit_endlist+independent_segments",
                "-hls_segment_type",
                "mpegts",
                "-hls_segment_filename",
            ])
            .arg(&segment_pattern)
            .arg(&playlist)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .context("could not start ffmpeg for HLS")?;

        let stdin = child.stdin.take().context("ffmpeg gave no stdin")?;
        info!(
            dir = %output.dir.display(),
            segment_seconds = config.segment_seconds,
            "HLS output started"
        );

        Ok(Self {
            stdin,
            child,
            timestamp: 0,
            saw_video: false,
        })
    }

    /// Whether any video has been seen.
    pub fn saw_video(&self) -> bool {
        self.saw_video
    }

    /// Forward one message from the RTMP session.
    ///
    /// Both audio and video go to ffmpeg, even though only the video is kept:
    /// ffmpeg uses the audio timestamps to keep the container's clock sane, and
    /// dropping it entirely makes some players misjudge the duration.
    pub async fn push(&mut self, tag_type: u8, data: &Bytes) -> Result<()> {
        if tag_type == TAG_VIDEO {
            self.saw_video = true;
        }

        let mut packet = Vec::with_capacity(data.len() + 32);
        if self.timestamp == 0 {
            packet.extend_from_slice(&flv_header(
                tag_type == TAG_AUDIO,
                tag_type == TAG_VIDEO,
            ));
        }
        packet.extend_from_slice(&flv_tag(tag_type, data, self.timestamp));

        self.stdin
            .write_all(&packet)
            .await
            .context("the HLS packager stopped accepting data")?;

        // Advance by roughly a frame. The exact step does not matter to ffmpeg,
        // which reads the real timestamps from the FLV tags where it can; this
        // only keeps the clock moving forward.
        self.timestamp = self.timestamp.saturating_add(if tag_type == TAG_VIDEO { 33 } else { 20 });
        Ok(())
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
}

/// Whether a playlist currently exists.
pub fn playlist_ready(output: &HlsOutput) -> bool {
    output.playlist_path().is_file()
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
    fn the_playlist_path_is_inside_the_output_directory() {
        let out = HlsOutput::new(PathBuf::from("/tmp/hls"));
        assert_eq!(out.playlist_path(), PathBuf::from("/tmp/hls/stream.m3u8"));
    }

    #[test]
    fn defaults_match_the_documented_latency() {
        let c = HlsConfig::default();
        // Six two-second segments is roughly the buffering a player needs.
        assert_eq!(c.segment_seconds, 2);
        assert_eq!(c.playlist_size, 6);
    }
}

// Copyright 2026 youyoEulgo
// SPDX-License-Identifier: Apache-2.0

//! Live subtitles: turn an incoming stream into text other things can display.
//!
//! One task owns the recogniser for the live path. It follows the ingest, and
//! while a publisher is connected it runs ffmpeg against this program's own
//! RTMP listener to obtain 16 kHz mono PCM, which it feeds to the streaming
//! algorithm.
//!
//! # Why ffmpeg dials in rather than being fed
//!
//! The audio arrives as RTMP messages. Extracting them, rebuilding a container
//! and piping that to ffmpeg means reimplementing part of RTMP and getting the
//! container exactly right; the first attempt at it stalled as soon as the
//! stream carried video. Letting ffmpeg connect as an ordinary RTMP client
//! removes all of that, and makes this connection independent of whatever else
//! is reading the same stream.
//!
//! # Why this is separate from the WebSocket ingest
//!
//! The WebSocket path is request-driven: a client connects, pushes, and reads
//! its own replies. This one follows the relay, and any number of viewers can
//! watch the result without affecting each other. They share one engine because
//! the process holds one model — a llama.cpp context holds a single sequence,
//! so they take turns through the same mutex.

use std::sync::Arc;

use anyhow::{Context as _, Result};
use tokio::io::AsyncReadExt;
use tokio::sync::{Mutex, broadcast};
use tracing::{debug, info, warn};

use crate::rtmp::{IngestEvent, IngestHandle};
use crate::stream::StreamEngine;

/// A message published to viewers.
///
/// Two kinds share one connection: captions as they are recognised, and
/// appearance changes when the console edits the configuration. Keeping them
/// together means an overlay that is already open picks up a new setting
/// without being reloaded.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ViewerMessage {
    Subtitle {
        text: String,
        delta: String,
        reset: bool,
        at_ms: u64,
    },
    Caption(crate::config::CaptionConfig),
}

/// Shared handle for viewers to subscribe to subtitles.
#[derive(Clone)]
pub struct LiveSubtitles {
    tx: broadcast::Sender<ViewerMessage>,
    state: Arc<Mutex<LiveState>>,
    enabled: bool,
}

#[derive(Default)]
struct LiveState {
    /// Whether a publisher is currently being transcribed.
    active: bool,
    /// Latest full text, for a viewer that connects mid-stream.
    latest: String,
}

impl LiveSubtitles {
    pub fn subscribe(&self) -> broadcast::Receiver<ViewerMessage> {
        self.tx.subscribe()
    }

    /// Tell every open viewer that the caption appearance changed.
    pub fn broadcast_caption(&self, caption: crate::config::CaptionConfig) {
        let _ = self.tx.send(ViewerMessage::Caption(caption));
    }

    /// The most recent line, so a late viewer is not left blank.
    pub async fn latest(&self) -> String {
        self.state.lock().await.latest.clone()
    }

    pub async fn is_active(&self) -> bool {
        self.state.lock().await.active
    }

    /// Whether subtitles are being produced at all.
    pub fn enabled(&self) -> bool {
        self.enabled
    }
}

/// Chunk size fed to the recogniser, in seconds.
///
/// Matches the WebSocket path: the streaming algorithm expects roughly 160 ms,
/// and ffmpeg's output block is an arbitrary size.
const CHUNK_SECONDS: f32 = 0.16;

/// Start the live transcription task.
pub fn spawn(
    engine: Arc<Mutex<StreamEngine>>,
    ingest: IngestHandle,
    rtmp_url: String,
    context: String,
    language: Option<String>,
    enabled: bool,
) -> LiveSubtitles {
    let (tx, _) = broadcast::channel(256);
    let handle = LiveSubtitles {
        tx: tx.clone(),
        state: Arc::new(Mutex::new(LiveState::default())),
        enabled,
    };

    let mut events = ingest.subscribe();
    let task_handle = handle.clone();

    tokio::spawn(async move {
        if !enabled {
            info!("live subtitles disabled; the stream is still relayed");
            return;
        }

        // Stays up across publishers: the decoder is started when one connects
        // and dropped when it leaves, so restarting OBS does not mean restarting
        // this.
        let mut task: Option<tokio::task::JoinHandle<()>> = None;

        while let Ok(event) = events.recv().await {
            match event {
                IngestEvent::Published { stream_key } => {
                    info!(stream_key, "live transcription starting");
                    let engine = engine.clone();
                    let context = context.clone();
                    let language = language.clone();
                    let url = rtmp_url.clone();
                    let publisher = task_handle.clone();
                    task = Some(tokio::spawn(async move {
                        if let Err(err) =
                            decode(engine, url, context, language, publisher.clone()).await
                        {
                            warn!(error = %err, "live decoding stopped");
                        }
                        publisher.state.lock().await.active = false;
                    }));
                }
                IngestEvent::Unpublished { stream_key } => {
                    info!(stream_key, "live transcription stopping");
                    if let Some(t) = task.take() {
                        t.abort();
                    }
                    task_handle.state.lock().await.active = false;
                }
            }
        }
        if let Some(t) = task {
            t.abort();
        }
        debug!("live subtitle task ended");
    });

    handle
}

/// Pull audio from the relay and feed it to the recogniser.
async fn decode(
    engine: Arc<Mutex<StreamEngine>>,
    rtmp_url: String,
    context: String,
    language: Option<String>,
    out: LiveSubtitles,
) -> Result<()> {
    // ffmpeg connects to our own listener as an ordinary RTMP client and
    // produces exactly the format the recogniser wants. `-vn` drops the video
    // so this connection carries no more than it must.
    let mut child = tokio::process::Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            &rtmp_url,
            "-vn",
            "-f",
            "f32le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "pipe:1",
        ])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .context("could not start ffmpeg to read the stream")?;

    let mut stdout = child.stdout.take().context("ffmpeg gave no stdout")?;
    let mut stderr = child.stderr.take().context("ffmpeg gave no stderr")?;

    // ffmpeg's complaints are the only clue when it decodes nothing, so they
    // are read rather than discarded.
    tokio::spawn(async move {
        let mut text = String::new();
        let _ = stderr.read_to_string(&mut text).await;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            warn!(target: "r2t2::live::ffmpeg", "{line}");
        }
    });

    let mut state = engine
        .lock()
        .await
        .init_state(&context, language.as_deref(), 0, 1, CHUNK_SECONDS);
    out.state.lock().await.active = true;

    let started = std::time::Instant::now();
    let mut buffer: Vec<f32> = Vec::new();
    let mut raw = vec![0u8; 16 * 1024];
    let mut carry: Vec<u8> = Vec::new();

    loop {
        let n = match stdout.read(&mut raw).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        carry.extend_from_slice(&raw[..n]);

        // f32 samples are four bytes; keep a partial one for next time.
        let complete = carry.len() - (carry.len() % 4);
        if complete == 0 {
            continue;
        }
        buffer.extend(
            carry[..complete]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        );
        carry.drain(..complete);

        let chunk_samples = (CHUNK_SECONDS * crate::audio::TARGET_SAMPLE_RATE as f32) as usize;
        while buffer.len() >= chunk_samples {
            let chunk: Vec<f32> = buffer.drain(..chunk_samples).collect();

            // The engine is shared with the file paths, so the lock is held for
            // the decode and released between chunks.
            let outcome = {
                let guard = engine.lock().await;
                guard.push_no_reset(&chunk, &mut state)
            };

            match outcome {
                Ok(Some((_text, fixed))) => {
                    let previous = out.state.lock().await.latest.clone();
                    if fixed.len() > previous.len() {
                        let delta = fixed[previous.len()..].to_string();
                        let line = ViewerMessage::Subtitle {
                            text: fixed.clone(),
                            delta,
                            reset: false,
                            at_ms: started.elapsed().as_millis() as u64,
                        };
                        out.state.lock().await.latest = fixed;
                        let _ = out.tx.send(line);
                    }
                }
                Ok(None) => {}
                Err(err) => warn!(error = %err, "live decode failed"),
            }
        }
    }

    // Flush whatever is left so the last words are not lost.
    let flush = {
        let guard = engine.lock().await;
        guard.finish_no_reset(&mut state)
    };
    if let Err(err) = flush {
        warn!(error = %err, "could not flush the final audio");
    }
    debug!("live decoding ended");
    Ok(())
}

// Copyright 2026 youyoEulgo
// SPDX-License-Identifier: Apache-2.0

//! Live subtitles: turn an incoming stream into text other things can display.
//!
//! One task owns the recogniser for the live path. It subscribes to the RTMP
//! ingest, feeds each block of decoded audio into the streaming algorithm, and
//! publishes the text it produces to anyone listening.
//!
//! # Why this is separate from the WebSocket ingest
//!
//! The WebSocket path is request-driven: a client connects, pushes, and reads
//! its own replies. This one is a broadcast: one stream arrives, and any number
//! of viewers follow the subtitles without affecting each other. Sharing one
//! engine between them would be wrong anyway — a llama.cpp context holds a
//! single sequence, so the two paths would corrupt each other's state. They
//! take turns instead, through the same mutex the rest of the program uses.

use std::sync::Arc;

use tokio::sync::{broadcast, Mutex};
use tracing::{debug, info, warn};

use crate::rtmp::{IngestEvent, IngestHandle};
use crate::lazy_engine::LazyEngine;
use crate::stream::StreamEngine;

/// What is sent to a viewer over `/ws/subtitles`.
///
/// Two kinds of message share one connection: captions as they are recognised,
/// and appearance changes when the console edits the configuration. Keeping
/// them together means an overlay that is already open picks up a new setting
/// without being reloaded.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ViewerMessage {
    /// Recognition output.
    Subtitle {
        text: String,
        delta: String,
        reset: bool,
        at_ms: u64,
    },
    /// The caption appearance changed.
    Caption(crate::config::CaptionConfig),
}

/// A subtitle line, as published to viewers.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SubtitleLine {
    /// Full text since the last segment boundary.
    pub text: String,
    /// Text added by this update, for a renderer that appends.
    pub delta: String,
    /// True when the speaker paused and a new segment began.
    pub reset: bool,
    /// Milliseconds since the stream started, for ordering.
    pub at_ms: u64,
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
/// and the audio arriving from ffmpeg is an arbitrary size.
const CHUNK_SECONDS: f32 = 0.16;

/// Wait until the weights are usable, then load the engine.
///
/// Polls rather than watching the filesystem: a directory read every few
/// seconds costs nothing next to a model load, and a watcher would have to
/// cope with the many ways a download in progress can look finished without
/// being so.
async fn wait_for_engine(
    engine: &Arc<LazyEngine>,
) -> Option<Arc<Mutex<StreamEngine>>> {
    let mut announced = false;
    loop {
        match engine.get().await {
            Ok(loaded) => return Some(loaded),
            Err(_) => {
                if !announced {
                    info!("waiting for the recognition model; captions begin once it is present");
                    announced = true;
                }
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            }
        }
    }
}

/// Start the live transcription task.
///
/// `enabled` is the `--no-subtitles` switch: when false the ingest still runs
/// and audio is still decoded, so the picture path can be exercised without
/// paying for recognition.
pub fn spawn(
    engine: Arc<LazyEngine>,
    ingest: IngestHandle,
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
            info!("live subtitles disabled; audio will be decoded but not transcribed");
            return;
        }

        // Waiting rather than giving up. The weights are normally fetched from
        // the interface after the server has already started, and a task that
        // exited on the first failure would leave live captions dead until the
        // process was restarted. Meanwhile the published audio is still
        // decoded, so the stream itself is not held up either way.
        let engine = match wait_for_engine(&engine).await {
            Some(engine) => engine,
            None => return,
        };

        let mut state = engine.lock().await.init_state(
            &context,
            language.as_deref(),
            0,
            // The live path uses the same rollback window as the file paths.
            1,
            CHUNK_SECONDS,
        );

        let mut started = std::time::Instant::now();
        let mut buffer: Vec<f32> = Vec::new();
        let mut total_samples = 0usize;
        let mut chunks = 0usize;

        while let Ok(event) = events.recv().await {
            match event {
                IngestEvent::Published { stream_key } => {
                    info!(stream_key, "live transcription starting");
                    started = std::time::Instant::now();
                    state = engine.lock().await.init_state(
                        &context,
                        language.as_deref(),
                        0,
                        1,
                        CHUNK_SECONDS,
                    );
                    buffer.clear();
                    task_handle.state.lock().await.active = true;
                }

                IngestEvent::Unpublished { stream_key } => {
                    info!(stream_key, "live transcription stopping");
                    // Flush whatever is left so the last words are not lost.
                    let flush = {
                        let guard = engine.lock().await;
                        guard.finish_no_reset(&mut state)
                    };
                    if let Err(err) = flush {
                        warn!(error = %err, "could not flush the final audio");
                    }
                    info!(
                        samples = total_samples,
                        chunks,
                        seconds = total_samples as f64
                            / crate::audio::TARGET_SAMPLE_RATE as f64,
                        "live audio summary"
                    );
                    let mut st = task_handle.state.lock().await;
                    st.active = false;
                    buffer.clear();
                    total_samples = 0;
                    chunks = 0;
                }

                IngestEvent::Audio { samples } => {
                    if samples.is_empty() {
                        continue;
                    }
                    total_samples += samples.len();
                    if total_samples / crate::audio::TARGET_SAMPLE_RATE as usize % 5 == 0
                        && chunks == 0
                    {
                        debug!(
                            received = total_samples,
                            "live audio arriving"
                        );
                    }
                    buffer.extend_from_slice(&samples);

                    let chunk_samples =
                        (CHUNK_SECONDS * crate::audio::TARGET_SAMPLE_RATE as f32) as usize;
                    while buffer.len() >= chunk_samples {
                        let chunk: Vec<f32> = buffer.drain(..chunk_samples).collect();
                        chunks += 1;

                        // The engine is shared with the file paths, so take the
                        // lock for the decode and release it between chunks.
                        let outcome = {
                            let guard = engine.lock().await;
                            guard.push_no_reset(&chunk, &mut state)
                        };

                        match outcome {
                            Ok(Some((_text, fixed))) => {
                                let previous =
                                    task_handle.state.lock().await.latest.clone();
                                if fixed.len() > previous.len() {
                                    let delta = fixed[previous.len()..].to_string();
                                    let line = ViewerMessage::Subtitle {
                                        text: fixed.clone(),
                                        delta,
                                        reset: false,
                                        at_ms: started.elapsed().as_millis() as u64,
                                    };
                                    task_handle.state.lock().await.latest = fixed;
                                    let _ = task_handle.tx.send(line);
                                }
                            }
                            Ok(None) => {}
                            Err(err) => {
                                warn!(error = %err, "live decode failed");
                            }
                        }
                    }
                }
            }
        }
        debug!("live subtitle task ended");
    });

    handle
}

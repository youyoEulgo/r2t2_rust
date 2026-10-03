// Copyright 2026 youyoEulgo
// SPDX-License-Identifier: Apache-2.0

//! `r2t2 serve` — live speech recognition over WebSocket.
//!
//! Speaks the same protocol as the reference Python server, so existing clients
//! work unchanged:
//!
//! 1. The client sends a JSON header; `requestId` is required.
//! 2. Binary frames carry raw little-endian `int16` mono PCM at 16 kHz. The
//!    first frame may instead be a whole WAV file.
//! 3. `msg.text` in each reply is the **new** text since the last message;
//!    concatenate them client-side.
//! 4. A text frame equal to `YOUDAO_ONETIME_ASR_STREAM_EOS` ends the stream.
//!
//! # Concurrency
//!
//! There is one GPU and one llama.cpp context, and a context holds a single KV
//! cache with one sequence. Two concurrent `llama_decode` calls on it would
//! interleave and corrupt each other, so every decode is serialised behind a
//! mutex. Each connection keeps its own [`StreamState`] and VAD, so only the
//! decode step is shared.
//!
//! The reference server reaches the same guarantee by running a single Sanic
//! worker with asyncio.
//!
//! # Segmentation
//!
//! Voice activity detection bounds each segment: when the detector reports
//! sustained silence the ASR state is reset and the client is told with
//! `"reset": true`. Without this a long stream would accumulate audio without
//! limit and never produce a natural place to break the text.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context as _, Result};
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use clap::Args;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, Mutex};
use tracing::{debug, error, info, warn};

use crate::audio::TARGET_SAMPLE_RATE;
use crate::cli::CommonArgs;
use crate::stream::{StreamEngine, StreamState};
use crate::vad::{Segmenter, Sensitivity, VadConfig, FRAME_SAMPLES};

/// Ends a stream. Must match the reference server byte for byte.
const EOS: &str = "YOUDAO_ONETIME_ASR_STREAM_EOS";

/// Chunk size the streaming algorithm decodes at.
const CHUNK_SECONDS: f32 = 0.16;
const CHUNK_SAMPLES: usize = (TARGET_SAMPLE_RATE as f32 * CHUNK_SECONDS) as usize;
/// Extra audio the first chunk carries, so the opening decode has context.
const LOOKAHEAD_SAMPLES: usize = CHUNK_SAMPLES;

/// Maximum tokens per decode step.
const MAX_NEW_TOKENS: i32 = 10;

/// Trailing tokens left unfixed when prompting.
const UNFIXED_TOKEN_NUM: usize = 1;

/// Repeats before a run of identical output counts as a stuck decoder.
const QUALITY_REPEAT_THRESHOLD: usize = 5;

/// How long to wait for the next frame before giving up on a connection.
const RECV_TIMEOUT: Duration = Duration::from_secs(120);

/// Arguments for `r2t2 serve`.
#[derive(Debug, Args)]
pub struct ServeArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    /// Address to bind.
    #[arg(long = "bind", value_name = "ADDR", default_value = "0.0.0.0")]
    pub bind: String,

    /// Port to listen on.
    #[arg(short = 'p', long = "port", value_name = "PORT", default_value_t = 8272)]
    pub port: u16,

    /// VAD sensitivity: quality | lowbitrate | aggressive | veryaggressive.
    #[arg(long = "vad-sensitivity", value_name = "LEVEL", default_value = "aggressive")]
    pub vad_sensitivity: String,

    /// Milliseconds of silence that end a segment.
    #[arg(long = "vad-min-silence-ms", value_name = "MS", default_value_t = 400)]
    pub vad_min_silence_ms: u64,

    /// Milliseconds of speech that start one.
    #[arg(long = "vad-min-speech-ms", value_name = "MS", default_value_t = 160)]
    pub vad_min_speech_ms: u64,

    /// Do not serve the web interface; WebSocket only.
    #[arg(long = "no-web")]
    pub no_web: bool,

    /// RTMP port to receive a stream on, for OBS and the like.
    ///
    /// Point OBS at `rtmp://<host>:<port>/live` with any stream key.
    #[arg(long = "rtmp-port", value_name = "PORT", default_value_t = 1935)]
    pub rtmp_port: u16,

    /// Do not accept RTMP; use only the WebSocket ingest.
    #[arg(long = "no-rtmp")]
    pub no_rtmp: bool,

    /// Decode incoming RTMP audio but produce no subtitles.
    ///
    /// Useful when the stream is being forwarded for its picture alone, or
    /// while testing the video path without paying for recognition.
    #[arg(long = "no-subtitles")]
    pub no_subtitles: bool,

    /// Do not repackage the incoming video for the preview player.
    #[arg(long = "no-video")]
    pub no_video: bool,

    /// Directory for HLS segments.
    ///
    /// Defaults to a directory under the work directory, cleared on start.
    #[arg(long = "hls-dir", value_name = "DIR")]
    pub hls_dir: Option<PathBuf>,

    /// Where uploaded files and their results are kept.
    ///
    /// Defaults to `~/.local/share/r2t2/work`, so results survive a reboot.
    #[arg(long = "work-dir", value_name = "DIR")]
    pub work_dir: Option<PathBuf>,
}

/// Run the server. Blocks until interrupted.
pub async fn run(args: ServeArgs) -> Result<()> {
    let (model, mmproj) = args.common.resolve_model()?;
    info!(model = %model.display(), mmproj = %mmproj.display(), "loading model");

    // Create the configuration file if it is not there, so there is something
    // to edit and the console has something to change.
    if let Err(err) = crate::config::Config::ensure_file(&crate::paths::config_file()) {
        warn!(error = %err, "could not create the configuration file; defaults will be used");
    }

    let cfg = args.common.engine_config(model.clone(), mmproj.clone());
    // Exactly one model in the process. The weights are several gigabytes and
    // a consumer GPU holds one copy, so every path — WebSocket ingest, RTMP
    // subtitles, and file uploads — shares this engine and takes turns.
    let engine = Arc::new(Mutex::new(
        StreamEngine::load(&cfg, MAX_NEW_TOKENS)
            .context("failed to initialise the llama.cpp engine")?,
    ));
    info!("model ready");

    let vad_config = VadConfig {
        sensitivity: Sensitivity::parse(&args.vad_sensitivity)?,
        min_silence_frames: (args.vad_min_silence_ms / 20).max(1) as usize,
        min_speech_frames: (args.vad_min_speech_ms / 20).max(1) as usize,
        ..Default::default()
    };
    info!(?vad_config, "VAD configured");

    // The web interface gets a second engine handle rather than sharing the
    // streaming one: a llama.cpp context holds one KV cache and one sequence,
    // and the streaming path keeps a long-lived state in it. Two handles cost
    // a second copy of the KV cache but keep an upload from disturbing a live
    // connection's sequence.
    let web = if args.no_web {
        None
    } else {
        let work_dir = args
            .work_dir
            .clone()
            .unwrap_or_else(crate::paths::work_dir);
        std::fs::create_dir_all(&work_dir)
            .with_context(|| format!("could not create {}", work_dir.display()))?;

        info!(dir = %work_dir.display(), "web interface enabled");
        Some(Arc::new(crate::web::WebState::new(
            work_dir,
            engine.clone(),
        )))
    };

    // RTMP ingest runs alongside the WebSocket one, and each is independently
    // switchable so the subtitle path and the picture path can be tested on
    // their own.
    // The HLS output is rebuilt from scratch each run: segments from an
    // earlier stream would be served alongside the new ones and confuse a
    // player about where the stream begins.
    let hls_setup = if args.no_rtmp || args.no_video {
        None
    } else {
        let dir = args
            .hls_dir
            .clone()
            .unwrap_or_else(|| crate::paths::work_dir().join("hls"));
        let output = crate::hls::HlsOutput::new(dir);
        crate::hls::HlsPackager::cleanup(&output);
        Some(crate::rtmp::HlsSetup {
            output,
            config: crate::hls::HlsConfig::default(),
        })
    };

    let ingest = if args.no_rtmp {
        None
    } else {
        Some(crate::rtmp::serve(args.rtmp_port, hls_setup.clone()).await?)
    };

    let live = match &ingest {
        None => None,
        Some(handle) => {
            info!("live subtitle path ready");
            Some(crate::live::spawn(
                engine.clone(),
                handle.clone(),
                args.common.context.clone(),
                args.common.forced_language().map(str::to_owned),
                !args.no_subtitles,
            ))
        }
    };

    // Tell the HLS route where to read from before any request arrives.
    if let Some(hls) = &hls_setup {
        crate::web::set_hls_dir(hls.output.dir.clone());
    }

    let state = Arc::new(AppState {
        engine: engine.clone(),
        vad_config,
        context: args.common.context.clone(),
        web,
        ingest,
        live,
        rtmp_port: args.rtmp_port,
        hls: hls_setup.map(|h| h.output),
    });

    let mut app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/asr_stream_api_v1", get(handler_v1))
        // Viewers subscribe here; the live pipeline publishes to it.
        .route("/ws/subtitles", get(handler_subtitles))
        .route("/api/live", get(handler_live_status))
        // The caption appearance, shared by the console and the overlay.
        .route(
            "/api/live/caption",
            get(handler_caption_get).post(handler_caption_set),
        );
    if state.web.is_some() {
        app = app.merge(crate::web::routes());
    }
    let app = app.with_state(state);

    let addr: SocketAddr = format!("{}:{}", args.bind, args.port)
        .parse()
        .with_context(|| format!("invalid bind address {}:{}", args.bind, args.port))?;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("could not bind {addr}"))?;
    info!(%addr, "listening: ws://{addr}/asr_stream_api_v1");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("server error")?;
    Ok(())
}

// --------------------------------------------------------------------------- //
// wire types
// --------------------------------------------------------------------------- //

/// The client's opening message.
///
/// Fields the server does not act on are still declared: `serde` ignores
/// unknown keys by default, so this documents the protocol and keeps the
/// handshake strictly validated rather than accidentally lenient. `channels`
/// and `sample_rate` are accepted but unused because the server requires
/// 16 kHz mono anyway, and the reference client always sends both.
#[derive(Debug, Deserialize)]
struct Header {
    #[serde(rename = "requestId")]
    request_id: Option<String>,
    /// Language hint; absent means let the model decide.
    language: Option<String>,
    /// Whether to segment with voice activity detection. Defaults to on.
    #[serde(default)]
    use_vad: Option<bool>,

    // ---- accepted for protocol compatibility, not acted on ----------------
    /// Sent by the reference client; the server does not authenticate.
    #[serde(default, rename = "secret_key")]
    _secret_key: Option<String>,
    /// Channel count the client claims. Audio is taken as 16 kHz mono.
    #[serde(default)]
    _channels: Option<u32>,
    /// Sample rate the client claims. Audio is taken as 16 kHz mono.
    #[serde(default)]
    _sample_rate: Option<u32>,
    /// Client mode selector; unused here.
    #[serde(default)]
    _mode: Option<String>,
}

#[derive(Debug, Serialize)]
struct OutMsg {
    text: String,
    reset: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    asr_cost_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_cost_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lag_ms: Option<f64>,
}

#[derive(Debug, Serialize)]
struct OutEnvelope {
    status: &'static str,
    #[serde(rename = "requestId")]
    request_id: Option<String>,
    msg: OutMsg,
}

impl OutEnvelope {
    fn ok(request_id: Option<&str>, msg: OutMsg) -> Self {
        Self {
            status: "success",
            request_id: request_id.map(str::to_owned),
            msg,
        }
    }

    fn error(request_id: Option<&str>, text: impl Into<String>) -> Self {
        Self {
            status: "error",
            request_id: request_id.map(str::to_owned),
            msg: OutMsg {
                text: text.into(),
                reset: false,
                asr_cost_ms: None,
                total_cost_ms: None,
                lag_ms: None,
            },
        }
    }
}

// --------------------------------------------------------------------------- //
// shared state
// --------------------------------------------------------------------------- //

pub struct AppState {
    /// The process's one engine, serialised: see the module docs on
    /// concurrency. Shared with the live and web paths.
    engine: Arc<Mutex<StreamEngine>>,
    vad_config: VadConfig,
    context: String,
    /// Present when the web interface is enabled. Its engine handle shares the
    /// same context, so an upload and a live stream cannot decode at once.
    pub web: Option<Arc<crate::web::WebState>>,
    /// Present when RTMP ingest is enabled.
    pub ingest: Option<crate::rtmp::IngestHandle>,
    /// Present when RTMP ingest is enabled and feeds the live subtitle path.
    pub live: Option<crate::live::LiveSubtitles>,
    /// Port the RTMP listener bound, for display.
    pub rtmp_port: u16,
    /// Where HLS segments are written, when the video path is on.
    pub hls: Option<crate::hls::HlsOutput>,
}

/// Report what the live path is doing, for the console to display.
async fn handler_live_status(State(app): State<Arc<AppState>>) -> impl IntoResponse {
    let rtmp_enabled = app.ingest.is_some();
    let (publishing, stream_key) = match &app.ingest {
        Some(ingest) => (ingest.is_publishing().await, ingest.stream_key().await),
        None => (false, String::new()),
    };
    let (subtitles_enabled, latest) = match &app.live {
        Some(live) => (live.enabled(), live.latest().await),
        None => (false, String::new()),
    };
    let (video_enabled, video_ready) = match &app.hls {
        Some(output) => (true, crate::hls::playlist_ready(output)),
        None => (false, false),
    };

    Json(serde_json::json!({
        "rtmp_enabled": rtmp_enabled,
        "rtmp_port": app.rtmp_port,
        "subtitles_enabled": subtitles_enabled,
        "publishing": publishing,
        "stream_key": stream_key,
        "latest": latest,
        "video_enabled": video_enabled,
        "video_ready": video_ready,
    }))
}

/// Report the caption appearance.
async fn handler_caption_get() -> impl IntoResponse {
    Json(crate::config::Config::load_default().caption)
}

/// Replace the caption appearance and save it.
///
/// Saved immediately rather than on shutdown: the point of a config file is
/// that an edit survives, and a crash should not cost someone their settings.
async fn handler_caption_set(
    State(app): State<Arc<AppState>>,
    Json(new): Json<crate::config::CaptionConfig>,
) -> Response {
    let mut cfg = crate::config::Config::load_default();
    cfg.caption = new.sanitized();

    let path = crate::paths::config_file();
    match cfg.save(&path) {
        Ok(()) => {
            app.live.as_ref().map(|l| l.broadcast_caption(cfg.caption.clone()));
            Json(cfg.caption).into_response()
        }
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("could not save the configuration: {err:#}"),
        )
            .into_response(),
    }
}

/// Subscribe to the live subtitle stream.
///
/// Read-only: a viewer sends nothing. On connect the latest line is sent
/// first, so a viewer that joins mid-stream is not left blank until the next
/// update.
async fn handler_subtitles(
    ws: WebSocketUpgrade,
    State(app): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| subtitle_socket(socket, app))
}

async fn subtitle_socket(socket: WebSocket, app: Arc<AppState>) {
    let (mut sink, stream) = socket.split();

    let Some(live) = app.live.clone() else {
        let _ = sink
            .send(Message::Text(
                serde_json::json!({"error": "live subtitles are not enabled"})
                    .to_string()
                    .into(),
            ))
            .await;
        return;
    };

    // A status line first, then the most recent text, then updates.
    let status = serde_json::json!({
        "type": "status",
        "enabled": live.enabled(),
        "active": live.is_active().await,
        "latest": live.latest().await,
    });
    if sink
        .send(Message::Text(status.to_string().into()))
        .await
        .is_err()
    {
        return;
    }

    let mut updates = live.subscribe();

    // Forward updates until the viewer goes away. The reader task exists only
    // to notice a close, since a viewer never sends anything meaningful.
    let mut reader = stream;
    loop {
        tokio::select! {
            message = updates.recv() => match message {
                Ok(line) => {
                    let Ok(payload) = serde_json::to_string(&line) else {
                        continue;
                    };
                    if sink
                        .send(Message::Text(payload.into()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    debug!(skipped = n, "a subtitle viewer fell behind");
                }
                Err(broadcast::error::RecvError::Closed) => break,
            },
            incoming = reader.next() => match incoming {
                None | Some(Err(_)) => break,
                Some(Ok(Message::Close(_))) => break,
                Some(Ok(_)) => {}
            },
        }
    }
}

/// Resolves when the process should stop accepting work.
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    info!("shutting down");
}

async fn handler_v1(ws: WebSocketUpgrade, State(state): State<Arc<AppState>>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    if let Err(err) = run_session(socket, state).await {
        warn!(error = %err, "session ended with error");
    }
}

/// Milliseconds since process start. Only differences are ever reported.
fn now_ms() -> f64 {
    use std::sync::OnceLock;
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

async fn run_session(socket: WebSocket, state: Arc<AppState>) -> Result<()> {
    let (mut sink, mut stream) = socket.split();

    // ---- handshake ------------------------------------------------------
    let header = match recv_with_timeout(&mut stream, RECV_TIMEOUT).await? {
        Some(Message::Text(t)) => {
            serde_json::from_str::<Header>(&t).context("malformed JSON header")?
        }
        Some(_) => bail!("expected a JSON header as the first frame"),
        None => bail!("connection closed before the header arrived"),
    };

    let request_id = header.request_id.clone();
    let Some(rid) = request_id.as_deref() else {
        let env = OutEnvelope::error(None, "missing requestId in header");
        let _ = sink.send(Message::Text(serde_json::to_string(&env)?.into())).await;
        bail!("header has no requestId");
    };

    let language = header.language.as_deref().filter(|s| !s.is_empty());
    let use_vad = header.use_vad.unwrap_or(true);
    info!(rid, language = language.unwrap_or("<auto>"), use_vad, "session started");

    // ---- per-connection state -------------------------------------------
    let mut stream_state: StreamState = {
        let engine = state.engine.lock().await;
        engine.init_state(&state.context, language, 0, UNFIXED_TOKEN_NUM, CHUNK_SECONDS)
    };
    let mut segmenter = Segmenter::new(state.vad_config.clone())?;

    // Audio not yet formed into a full chunk.
    let mut pending: Vec<f32> = Vec::new();
    // Partial VAD frame.
    let mut vad_pending: Vec<f32> = Vec::new();
    // Length of `last_fixed_text` already sent, so updates are incremental.
    let mut emitted_len: usize = 0;
    let mut is_first_frame = true;
    let mut chunk_index: usize = 0;

    loop {
        let msg = match recv_with_timeout(&mut stream, RECV_TIMEOUT).await {
            Ok(Some(m)) => m,
            Ok(None) => {
                info!(rid, "client disconnected");
                break;
            }
            Err(err) => {
                info!(rid, error = %err, "receive ended");
                break;
            }
        };
        let recv_time = now_ms();

        let bytes = match msg {
            Message::Binary(b) => b,
            Message::Text(t) if t == EOS => {
                info!(rid, "EOS received");
                let t0 = now_ms();
                let (final_text, cost) = {
                    let guard = state.engine.lock().await;
                    (guard.finish_no_reset(&mut stream_state)?, now_ms() - t0)
                };
                let new_text = take_increment(&mut emitted_len, &final_text);
                let env = OutEnvelope::ok(
                    Some(rid),
                    OutMsg {
                        text: new_text,
                        reset: true,
                        asr_cost_ms: Some(round1(cost)),
                        total_cost_ms: Some(round1(cost)),
                        lag_ms: Some(round1(now_ms() - recv_time)),
                    },
                );
                let _ = sink.send(Message::Text(serde_json::to_string(&env)?.into())).await;
                let _ = sink.send(Message::Close(None)).await;
                info!(rid, "stream finished");
                return Ok(());
            }
            Message::Text(t) => {
                debug!(rid, text = %t, "ignoring non-EOS text frame");
                continue;
            }
            Message::Ping(_) | Message::Pong(_) => continue,
            Message::Close(_) => {
                info!(rid, "close frame received");
                break;
            }
        };

        // The first binary frame may be a whole WAV file; later ones are raw
        // PCM. This lets a client push a file without stripping the header.
        let pcm = if is_first_frame {
            decode_first_frame(&bytes)?
        } else {
            pcm_from_i16_le(&bytes)
        };
        is_first_frame = false;
        if pcm.is_empty() {
            continue;
        }
        pending.extend_from_slice(&pcm);

        // ---- VAD ---------------------------------------------------------
        let mut seg_closed = false;
        if use_vad {
            vad_pending.extend_from_slice(&pcm);
            while vad_pending.len() >= FRAME_SAMPLES {
                let frame: Vec<f32> = vad_pending.drain(..FRAME_SAMPLES).collect();
                let frame_i16: Vec<i16> = frame
                    .iter()
                    .map(|s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
                    .collect();
                if let Some(seg) = segmenter.push_frame(&frame_i16)? {
                    debug!(rid, start = seg.start_frame, frames = seg.frames, "VAD closed a segment");
                    seg_closed = true;
                }
            }
        }

        // ---- decode every complete chunk ---------------------------------
        let mut new_text = String::new();
        let mut decode_cost = 0.0f64;

        while pending.len() >= CHUNK_SAMPLES {
            // The opening chunk is longer, mirroring the CLI's lookahead, so
            // the first decode has enough context to commit anything at all.
            let take = if chunk_index == 0 {
                (CHUNK_SAMPLES + LOOKAHEAD_SAMPLES).min(pending.len())
            } else {
                CHUNK_SAMPLES
            };
            let chunk: Vec<f32> = pending.drain(..take).collect();
            stream_state.chunk_size_sec = chunk.len() as f32 / TARGET_SAMPLE_RATE as f32;
            stream_state.chunk_size_samples = chunk.len();

            let t0 = now_ms();
            let outcome = {
                let guard = state.engine.lock().await;
                guard.push_no_reset(&chunk, &mut stream_state)
            };
            decode_cost += now_ms() - t0;
            chunk_index += 1;

            match outcome {
                Ok(Some((_text, fixed))) => {
                    // Repair decoder loops before the text goes out. The
                    // stream protocol is append-only -- an increment already
                    // sent cannot be retracted -- so a repetition has to be
                    // caught here rather than cleaned up at the end.
                    let inc = take_increment(&mut emitted_len, &fixed);
                    if !inc.is_empty() {
                        let repaired =
                            crate::quality::fix_repetitions(&inc, QUALITY_REPEAT_THRESHOLD);
                        new_text.push_str(&repaired);
                    }
                }
                Ok(None) => {}
                Err(err) => {
                    error!(rid, error = %err, "decode failed");
                    let env = OutEnvelope::error(Some(rid), format!("decode failed: {err}"));
                    let _ = sink.send(Message::Text(serde_json::to_string(&env)?.into())).await;
                    return Ok(());
                }
            }
        }

        // ---- segment boundary --------------------------------------------
        let mut reset = false;
        if seg_closed {
            // Flush the segment, then start a clean state so the next one does
            // not inherit this one's audio or accumulated text.
            let t0 = now_ms();
            let finish_text = {
                let guard = state.engine.lock().await;
                guard.finish_no_reset(&mut stream_state).unwrap_or_default()
            };
            decode_cost += now_ms() - t0;
            new_text.push_str(&take_increment(&mut emitted_len, &finish_text));

            let engine = state.engine.lock().await;
            stream_state = engine.init_state(
                &state.context,
                language,
                0,
                UNFIXED_TOKEN_NUM,
                CHUNK_SECONDS,
            );
            drop(engine);

            segmenter = Segmenter::new(state.vad_config.clone())?;
            vad_pending.clear();
            emitted_len = 0;
            chunk_index = 0;
            reset = true;
            info!(rid, "segment closed, state reset");
        }

        let env = OutEnvelope::ok(
            Some(rid),
            OutMsg {
                text: new_text,
                reset,
                asr_cost_ms: Some(round1(decode_cost)),
                total_cost_ms: Some(round1(decode_cost)),
                lag_ms: Some(round1(now_ms() - recv_time)),
            },
        );
        if sink
            .send(Message::Text(serde_json::to_string(&env)?.into()))
            .await
            .is_err()
        {
            info!(rid, "client went away");
            break;
        }
    }

    Ok(())
}

/// Return the part of `full` that has not been sent yet, and advance the mark.
///
/// Text can in principle shrink between updates (the model may revise within
/// its rollback window), so a shorter string yields nothing rather than
/// re-sending a negative-length slice.
fn take_increment(emitted_len: &mut usize, full: &str) -> String {
    if full.len() > *emitted_len {
        let inc = full[*emitted_len..].to_string();
        *emitted_len = full.len();
        inc
    } else {
        String::new()
    }
}

async fn recv_with_timeout(
    stream: &mut (impl StreamExt<Item = Result<Message, axum::Error>> + Unpin),
    timeout: Duration,
) -> Result<Option<Message>> {
    match tokio::time::timeout(timeout, stream.next()).await {
        Err(_) => bail!("timed out waiting for a frame after {timeout:?}"),
        Ok(None) => Ok(None),
        Ok(Some(Ok(m))) => Ok(Some(m)),
        Ok(Some(Err(e))) => bail!("websocket error: {e}"),
    }
}

/// Decode the first binary frame, which may be a WAV file or raw PCM.
fn decode_first_frame(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        decode_wav_bytes(bytes)
    } else {
        Ok(pcm_from_i16_le(bytes))
    }
}

/// Raw little-endian `int16` PCM to `f32` in `[-1, 1]`.
fn pcm_from_i16_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect()
}

/// In-memory WAV decode, mirroring `audio::load_wav_16k_mono`.
fn decode_wav_bytes(bytes: &[u8]) -> Result<Vec<f32>> {
    let reader = hound::WavReader::new(std::io::Cursor::new(bytes))
        .context("could not parse the WAV header")?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .context("bad WAV sample data")?,
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .collect::<Result<Vec<_>, _>>()
                .context("bad WAV sample data")?
                .into_iter()
                .map(|s| s as f32 / max)
                .collect()
        }
    };

    let channels = spec.channels.max(1) as usize;
    let mono: Vec<f32> = if channels == 1 {
        samples
    } else {
        samples
            .chunks_exact(channels)
            .map(|f| f.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    if spec.sample_rate == TARGET_SAMPLE_RATE || mono.is_empty() {
        return Ok(mono);
    }
    // Linear resample, matching the CLI's behaviour.
    let from = spec.sample_rate as f64;
    let to = TARGET_SAMPLE_RATE as f64;
    let duration = mono.len() as f64 / from;
    let out_len = (duration * to).round() as usize;
    if out_len == 0 {
        return Ok(Vec::new());
    }
    let step_in = duration / mono.len() as f64;
    let step_out = duration / out_len as f64;
    let mut out = Vec::with_capacity(out_len);
    let mut j = 0usize;
    for i in 0..out_len {
        let t = i as f64 * step_out;
        while j + 1 < mono.len() && (j + 1) as f64 * step_in <= t {
            j += 1;
        }
        if j + 1 >= mono.len() {
            out.push(mono[mono.len() - 1]);
            continue;
        }
        let t0 = j as f64 * step_in;
        let t1 = (j + 1) as f64 * step_in;
        let frac = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
        out.push(mono[j] + (mono[j + 1] - mono[j]) * frac as f32);
    }
    Ok(out)
}

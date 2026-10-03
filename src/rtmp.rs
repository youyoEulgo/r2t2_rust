// Copyright 2026 youyoEulgo
// SPDX-License-Identifier: Apache-2.0

//! RTMP ingest: receive a stream pushed by OBS or anything else that speaks
//! RTMP, and hand its audio to the recogniser.
//!
//! # Why RTMP
//!
//! It is what OBS offers out of the box: "custom" service, a server URL, and a
//! stream key. Nothing else OBS can push is as easy to point at a local
//! process, so accepting RTMP is what makes this usable as a drop-in target.
//!
//! # What arrives, and what has to happen to it
//!
//! RTMP carries FLV tags. The audio tag is normally **AAC, already encoded**,
//! and the recogniser needs 16 kHz mono PCM. Decoding and resampling are not
//! reimplemented here: the tag is forwarded to `ffmpeg`, which this project
//! already depends on, through a pipe. That keeps the audio path short and
//! supports whatever codec the sender chooses.
//!
//! # Scope
//!
//! One publisher at a time. This is a local ingest point for a single stream,
//! not a multi-tenant server; a second publisher is rejected rather than
//! silently mixed in with the first.

use std::sync::Arc;

use anyhow::{Context as _, Result};
use bytes::Bytes;
use rml_rtmp::handshake::{Handshake, HandshakeProcessResult, PeerType};
use rml_rtmp::sessions::{
    ServerSession, ServerSessionConfig, ServerSessionEvent, ServerSessionResult,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::{broadcast, Mutex};
use tracing::{debug, info, warn};

/// What the ingest reports to the rest of the program.
#[derive(Debug, Clone)]
pub enum IngestEvent {
    /// A publisher connected and started sending on this stream key.
    Published { stream_key: String },
    /// The publisher stopped.
    Unpublished { stream_key: String },
    /// Decoded mono 16 kHz PCM, ready for the recogniser.
    Audio { samples: Vec<f32> },
}

/// Shared handle to the ingest, for the rest of the program to observe.
#[derive(Clone)]
pub struct IngestHandle {
    tx: broadcast::Sender<IngestEvent>,
    state: Arc<Mutex<IngestState>>,
}

#[derive(Default)]
struct IngestState {
    publishing: bool,
    stream_key: String,
}

impl IngestHandle {
    /// Whether someone is currently pushing.
    pub async fn is_publishing(&self) -> bool {
        self.state.lock().await.publishing
    }

    /// The stream key of the current publisher, if any.
    pub async fn stream_key(&self) -> String {
        self.state.lock().await.stream_key.clone()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<IngestEvent> {
        self.tx.subscribe()
    }
}

/// Run the RTMP listener until the process ends.
///
/// `port` is where OBS points its "server URL"; the stream key is whatever the
/// user types there, and is reported back so the interface can show it.
pub async fn serve(port: u16, hls: Option<HlsSetup>) -> Result<IngestHandle> {
    let (tx, _) = broadcast::channel(256);
    let handle = IngestHandle {
        tx: tx.clone(),
        state: Arc::new(Mutex::new(IngestState::default())),
    };

    let listener = TcpListener::bind(("0.0.0.0", port))
        .await
        .with_context(|| format!("could not bind the RTMP port {port}"))?;
    info!(port, "RTMP ingest listening; point OBS at rtmp://<host>:{port}/live");

    let listener_handle = handle.clone();
    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((socket, peer)) => {
                    let h = listener_handle.clone();
                    let hls = hls.clone();
                    tokio::spawn(async move {
                        if let Err(err) = handle_connection(socket, h, hls.as_ref()).await {
                            debug!(%peer, error = %err, "RTMP connection ended");
                        }
                    });
                }
                Err(err) => {
                    warn!(error = %err, "RTMP accept failed");
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            }
        }
    });

    Ok(handle)
}

/// HLS settings carried into a connection.
#[derive(Clone)]
pub struct HlsSetup {
    pub output: crate::hls::HlsOutput,
    pub config: crate::hls::HlsConfig,
}

/// The outputs a publishing session feeds.
///
/// Held together so a single `&mut` carries them through the event loop, and so
/// they start and stop as a unit: both come from the same stream and there is
/// no sense in one outliving the other.
#[derive(Default)]
struct Outputs {
    audio: Option<AudioDecoder>,
    video: Option<crate::hls::HlsPackager>,
}

impl Outputs {
    fn start(
        events: broadcast::Sender<IngestEvent>,
        hls: Option<(&crate::hls::HlsOutput, &crate::hls::HlsConfig)>,
    ) -> Result<Self> {
        let audio = Some(AudioDecoder::spawn(events)?);
        let video = match hls {
            Some((output, config)) => Some(crate::hls::HlsPackager::spawn(output, config)?),
            None => None,
        };
        Ok(Self { audio, video })
    }
}

/// Drive one RTMP connection: feed bytes in, act on the events that come out.
async fn handle_connection(
    socket: tokio::net::TcpStream,
    handle: IngestHandle,
    hls: Option<&HlsSetup>,
) -> Result<()> {
    let mut socket = socket;

    // The handshake is a separate protocol step that the session object does
    // not perform: it must be completed before any chunks are fed in, and its
    // leftover bytes are the first chunks of the session.
    let mut handshake = Handshake::new(PeerType::Server);
    let p0p1 = handshake
        .generate_outbound_p0_and_p1()
        .context("could not build the RTMP handshake")?;
    socket.write_all(&p0p1).await?;

    let mut carry: Vec<u8> = Vec::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = socket.read(&mut buf).await?;
        if n == 0 {
            return Ok(());
        }
        carry.extend_from_slice(&buf[..n]);

        match handshake
            .process_bytes(&carry)
            .context("malformed RTMP handshake")?
        {
            HandshakeProcessResult::InProgress { response_bytes } => {
                socket.write_all(&response_bytes).await?;
                carry.clear();
            }
            HandshakeProcessResult::Completed {
                response_bytes,
                remaining_bytes,
            } => {
                socket.write_all(&response_bytes).await?;
                carry = remaining_bytes;
                break;
            }
        }
    }

    let (mut session, initial_results) = ServerSession::new(ServerSessionConfig::new())
        .context("could not start an RTMP session")?;
    let mut pending = initial_results;
    for result in pending.drain(..) {
        if let ServerSessionResult::OutboundResponse(p) = result {
            socket.write_all(&p.bytes).await?;
        }
    }

    // Whether this connection has been allowed to publish, and under what key.
    let mut approved_stream: Option<String> = None;
    // The decoder feeding off this connection, created once publishing starts.
    let mut outputs = Outputs::default();

    // Anything the handshake did not consume belongs to the session.
    if !carry.is_empty() {
        feed_session(
            &mut session,
            &carry,
            &mut socket,
            &handle,
            &mut approved_stream,
            &mut outputs,
            hls,
        )
        .await?;
    }

    loop {
        let n = socket.read(&mut buf).await?;
        if n == 0 {
            break;
        }

        let results = session
            .handle_input(&buf[..n])
            .context("malformed RTMP data")?;

        for result in results {
            match result {
                ServerSessionResult::OutboundResponse(packet) => {
                    socket.write_all(&packet.bytes).await?;
                }
                ServerSessionResult::RaisedEvent(event) => {
                    handle_event(
                        event,
                        &mut session,
                        &mut socket,
                        &handle,
                        &mut approved_stream,
                        &mut outputs,
                        hls,
                    )
                    .await?;
                }
                ServerSessionResult::UnhandleableMessageReceived(msg) => {
                    debug!(?msg, "unhandled RTMP message");
                }
            }
        }
    }

    if let Some(key) = approved_stream {
        let _ = handle.tx.send(IngestEvent::Unpublished { stream_key: key });
        handle.state.lock().await.publishing = false;
    }
    Ok(())
}

/// Feed leftover bytes into the session and act on what comes back.
#[allow(clippy::too_many_arguments)]
async fn feed_session(
    session: &mut ServerSession,
    bytes: &[u8],
    socket: &mut tokio::net::TcpStream,
    handle: &IngestHandle,
    approved_stream: &mut Option<String>,
    outputs: &mut Outputs,
    hls: Option<&HlsSetup>,
) -> Result<()> {
    let results = session
        .handle_input(bytes)
        .context("malformed RTMP data after the handshake")?;
    for result in results {
        match result {
            ServerSessionResult::OutboundResponse(packet) => {
                socket.write_all(&packet.bytes).await?;
            }
            ServerSessionResult::RaisedEvent(event) => {
                handle_event(event, session, socket, handle, approved_stream, outputs, hls).await?;
            }
            ServerSessionResult::UnhandleableMessageReceived(msg) => {
                debug!(?msg, "unhandled RTMP message");
            }
        }
    }
    Ok(())
}

/// Act on one session event.
#[allow(clippy::too_many_arguments)]
async fn handle_event(
    event: ServerSessionEvent,
    session: &mut ServerSession,
    socket: &mut tokio::net::TcpStream,
    handle: &IngestHandle,
    approved_stream: &mut Option<String>,
    outputs: &mut Outputs,
    hls: Option<&HlsSetup>,
) -> Result<()> {
    match event {
        ServerSessionEvent::ConnectionRequested { request_id, app_name } => {
            info!(app_name, "RTMP client connected");
            let results = session.accept_request(request_id)?;
            for r in results {
                if let ServerSessionResult::OutboundResponse(p) = r {
                    socket.write_all(&p.bytes).await?;
                }
            }
        }

        ServerSessionEvent::PublishStreamRequested {
            request_id,
            app_name,
            stream_key,
            mode,
        } => {
            info!(request_id, %app_name, %stream_key, ?mode, "publish requested");
            // One publisher at a time: a second stream would interleave two
            // speakers into one transcript with no way to tell them apart.
            if handle.state.lock().await.publishing {
                warn!(stream_key, "rejecting a second publisher");
                let results = session.reject_request(request_id, "NetStream.Publish.BadName", "another stream is already publishing")?;
                for r in results {
                    if let ServerSessionResult::OutboundResponse(p) = r {
                        socket.write_all(&p.bytes).await?;
                    }
                }
                return Ok(());
            }

            info!(app_name, stream_key, "publisher accepted");
            let results = session.accept_request(request_id)?;
            for r in results {
                if let ServerSessionResult::OutboundResponse(p) = r {
                    socket.write_all(&p.bytes).await?;
                }
            }

            *approved_stream = Some(stream_key.clone());
            {
                let mut state = handle.state.lock().await;
                state.publishing = true;
                state.stream_key = stream_key.clone();
            }
            *outputs = Outputs::start(handle.tx.clone(), hls.as_ref().map(|h| (&h.output, &h.config)))?;
            let _ = handle.tx.send(IngestEvent::Published { stream_key });
        }

        ServerSessionEvent::AudioDataReceived { data, .. } => {
            if let Some(decoder) = outputs.audio.as_mut() {
                decoder.push(&data).await?;
            }
            if let Some(packager) = outputs.video.as_mut() {
                packager.push(crate::rtmp::TAG_AUDIO, &data).await?;
            }
        }

        ServerSessionEvent::VideoDataReceived { data, .. } => {
            if let Some(packager) = outputs.video.as_mut() {
                packager.push(crate::rtmp::TAG_VIDEO, &data).await?;
            }
        }

        ServerSessionEvent::PublishStreamFinished { stream_key, .. } => {
            info!(stream_key, "publisher stopped");
            *approved_stream = None;
            *outputs = Outputs::default();
            let mut state = handle.state.lock().await;
            state.publishing = false;
            state.stream_key.clear();
            let _ = handle.tx.send(IngestEvent::Unpublished { stream_key });
        }

        other => debug!(?other, "RTMP event"),
    }
    Ok(())
}

// --------------------------------------------------------------------------- //
// audio decoding
// --------------------------------------------------------------------------- //

/// A minimal FLV header: signature, version, flags saying which streams are
/// present, and the zero-length first tag size the format requires.
pub fn flv_header(has_audio: bool, has_video: bool) -> [u8; 13] {
    let flags = (has_audio as u8) | ((has_video as u8) << 2);
    let mut h = [0u8; 13];
    h[..3].copy_from_slice(b"FLV");
    h[3] = 1; // version
    h[4] = flags;
    // bytes 5..9 are the header size, always 9.
    h[5..9].copy_from_slice(&9u32.to_be_bytes());
    // bytes 9..13 are the first tag size, always zero for a live stream.
    h
}

/// Wrap an RTMP message body into an FLV tag.
///
/// The RTMP payload is the *body* of a tag, not a whole stream, so it has to be
/// given back its 11-byte header and 4-byte trailing size before ffmpeg will
/// accept it.
pub fn flv_tag(tag_type: u8, body: &[u8], timestamp: u32) -> Vec<u8> {
    let len = body.len() as u32;
    let mut out = Vec::with_capacity(body.len() + 15);
    out.push(tag_type);
    out.extend_from_slice(&len.to_be_bytes()[1..]);
    out.extend_from_slice(&timestamp.to_be_bytes()[1..]);
    out.push((timestamp >> 24) as u8);
    out.extend_from_slice(&[0, 0, 0]); // stream id
    out.extend_from_slice(body);
    out.extend_from_slice(&(len + 11).to_be_bytes());
    out
}

/// FLV tag type for audio.
pub const TAG_AUDIO: u8 = 0x08;
/// FLV tag type for video.
pub const TAG_VIDEO: u8 = 0x09;

/// Turns RTMP audio tags into 16 kHz mono PCM.
///
/// The tags are FLV audio messages: a one-byte header followed by the codec's
/// frames. Rather than parse AAC by hand, the tags are wrapped back into a
/// minimal FLV stream and piped to `ffmpeg`, which demuxes, decodes and
/// resamples in one step. ffmpeg is already a dependency of this project, and
/// it handles every codec a sender might pick, which a hand-written AAC decoder
/// would not.
struct AudioDecoder {
    stdin: tokio::process::ChildStdin,
    task: tokio::task::JoinHandle<()>,
    /// Kept so the child is reaped when the decoder is dropped.
    child: tokio::process::Child,
    /// Timestamp of the next tag, in milliseconds.
    ///
    /// The incoming tags carry the sender's clock; ffmpeg needs a consistent
    /// one, and reusing the sender's avoids drift when a stream stalls.
    timestamp: u32,
}

impl AudioDecoder {
    fn spawn(events: broadcast::Sender<IngestEvent>) -> Result<Self> {
        let mut child = tokio::process::Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel", "error",
                // Read FLV from the pipe, since that is what RTMP hands us.
                "-f", "flv",
                "-i", "pipe:0",
                // Drop the video: this stage only needs sound.
                "-vn",
                // The recogniser's format.
                "-f", "f32le",
                "-ar", "16000",
                "-ac", "1",
                "pipe:1",
            ])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .context("could not start ffmpeg to decode the RTMP audio")?;

        let stdin = child.stdin.take().context("ffmpeg gave no stdin")?;
        let mut stdout = child.stdout.take().context("ffmpeg gave no stdout")?;

        // Emit samples in blocks that the streaming algorithm can consume
        // directly; it wants a chunk at a time, not one sample at a time.
        let task = tokio::spawn(async move {
            let mut raw = vec![0u8; 16 * 1024];
            let mut carry: Vec<u8> = Vec::new();
            loop {
                let n = match stdout.read(&mut raw).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => n,
                };
                carry.extend_from_slice(&raw[..n]);

                // f32 samples are 4 bytes; keep a partial one for next time.
                let complete = carry.len() - (carry.len() % 4);
                if complete == 0 {
                    continue;
                }
                let samples: Vec<f32> = carry[..complete]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                carry.drain(..complete);

                if !samples.is_empty()
                    && events.send(IngestEvent::Audio { samples }).is_err()
                {
                    // Nobody is listening yet; keep decoding anyway, the
                    // subtitles will simply start once someone subscribes.
                }
            }
        });

        Ok(Self {
            stdin,
            task,
            child,
            timestamp: 0,
        })
    }

    /// Feed one audio message to the decoder.
    ///
    /// The RTMP payload is the *body* of an FLV audio tag, not a whole stream,
    /// so each message is wrapped back into a tag and prefixed by an FLV header
    /// on the first call. Handing ffmpeg bare tag bodies produces no output at
    /// all: it looks for a container and finds none.
    async fn push(&mut self, data: &Bytes) -> Result<()> {
        let mut packet = Vec::with_capacity(data.len() + 16);

        if self.timestamp == 0 {
            packet.extend_from_slice(&flv_header(true, false));
        }

        // FLV tag: type (1) + data size (3) + timestamp (3) + ts ext (1) +
        // stream id (3), then the body, then the size again.
        packet.push(0x08); // audio
        let len = data.len() as u32;
        packet.extend_from_slice(&len.to_be_bytes()[1..]);
        packet.extend_from_slice(&self.timestamp.to_be_bytes()[1..]);
        packet.push((self.timestamp >> 24) as u8);
        packet.extend_from_slice(&[0, 0, 0]); // stream id
        packet.extend_from_slice(data);
        packet.extend_from_slice(&(len + 11).to_be_bytes());

        self.stdin
            .write_all(&packet)
            .await
            .context("the audio decoder stopped accepting data")?;

        // Advance by roughly one AAC frame; the exact value does not matter to
        // the decoder, only that time moves forward monotonically.
        self.timestamp = self.timestamp.saturating_add(20);
        Ok(())
    }
}

impl Drop for AudioDecoder {
    fn drop(&mut self) {
        self.task.abort();
        let _ = self.child.start_kill();
    }
}

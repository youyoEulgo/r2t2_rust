// Copyright 2026 youyoEulgo
// SPDX-License-Identifier: Apache-2.0

//! RTMP ingest and relay.
//!
//! A publisher (OBS, or anything else that speaks RTMP) connects, and whatever
//! it sends is relayed to every subscriber. This program is one of those
//! subscribers, through ffmpeg — which is the whole point, because ffmpeg
//! already knows RTMP and nothing here has to reimplement it.
//!
//! # Why relay rather than unpack
//!
//! An earlier version lifted the audio messages out of the publisher's stream,
//! rebuilt an FLV container around them by hand, and piped that to ffmpeg. It
//! worked while the stream carried nothing but audio, and stalled as soon as
//! video was present: ffmpeg stopped producing output until its stdout was
//! drained, the task meant to drain it never ran, and each side waited for the
//! other.
//!
//! With relaying, the publisher's messages are forwarded unchanged, ffmpeg
//! dials in as an ordinary RTMP client, and audio and video are separate
//! connections that cannot interfere with one another.

use std::sync::Arc;

use anyhow::{Context as _, Result};
use bytes::Bytes;
use rml_rtmp::chunk_io::Packet;
use rml_rtmp::handshake::{Handshake, HandshakeProcessResult, PeerType};
use rml_rtmp::sessions::{
    ServerSession, ServerSessionConfig, ServerSessionEvent, ServerSessionResult,
};
use rml_rtmp::time::RtmpTimestamp;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::net::tcp::OwnedWriteHalf;
use tokio::sync::{Mutex, broadcast};
use tracing::{debug, info, warn};

/// One message from a publisher, as relayed to subscribers.
#[derive(Debug, Clone)]
pub enum Media {
    Audio { data: Bytes, timestamp: u32 },
    Video { data: Bytes, timestamp: u32 },
}

/// What the ingest reports to the rest of the program.
#[derive(Debug, Clone)]
pub enum IngestEvent {
    Published { stream_key: String },
    Unpublished { stream_key: String },
}

/// Shared handle to the ingest.
#[derive(Clone)]
pub struct IngestHandle {
    events: broadcast::Sender<IngestEvent>,
    media: broadcast::Sender<Media>,
    state: Arc<Mutex<IngestState>>,
}

#[derive(Default)]
struct IngestState {
    publishing: bool,
    stream_key: String,
    /// How many subscribers are attached, for the interface to report.
    subscribers: usize,
}

impl IngestHandle {
    pub async fn is_publishing(&self) -> bool {
        self.state.lock().await.publishing
    }

    pub async fn stream_key(&self) -> String {
        self.state.lock().await.stream_key.clone()
    }

    pub async fn subscribers(&self) -> usize {
        self.state.lock().await.subscribers
    }

    /// Follow publisher connect/disconnect, not the media itself.
    pub fn subscribe(&self) -> broadcast::Receiver<IngestEvent> {
        self.events.subscribe()
    }
}

/// Run the RTMP listener until the process ends.
pub async fn serve(port: u16) -> Result<IngestHandle> {
    let (events, _) = broadcast::channel(64);
    // A subscriber that falls behind skips rather than stalling the publisher,
    // so this is a buffer depth rather than a back-pressure mechanism.
    let (media, _) = broadcast::channel(2048);
    let handle = IngestHandle {
        events,
        media,
        state: Arc::new(Mutex::new(IngestState::default())),
    };

    let listener = TcpListener::bind(("0.0.0.0", port))
        .await
        .with_context(|| format!("could not bind the RTMP port {port}"))?;
    info!(port, "RTMP listening; publish to rtmp://<host>:{port}/live");

    let listener_handle = handle.clone();
    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((socket, peer)) => {
                    let h = listener_handle.clone();
                    tokio::spawn(async move {
                        if let Err(err) = handle_connection(socket, h).await {
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

/// What a connection turned out to be.
#[derive(Clone)]
enum Role {
    Unknown,
    Publisher { stream_key: String },
    Subscriber { stream_id: u32 },
}

/// State shared between the read loop and, for a subscriber, the relay task.
///
/// The session must be reachable from both: reading parses what the client
/// sends, while relaying serialises outgoing chunks through the same
/// serializer, whose state depends on everything sent so far. One mutex around
/// it is the honest representation of that.
struct Shared {
    session: Mutex<ServerSession>,
    out: Mutex<OwnedWriteHalf>,
    role: Mutex<Role>,
    handle: IngestHandle,
    /// Taken by the relay task once this connection turns out to be a
    /// subscriber. Held here from the start so nothing is missed in between.
    media: Mutex<broadcast::Receiver<Media>>,
}

impl Shared {
    async fn send_packet(&self, packet: &Packet) -> Result<()> {
        let mut out = self.out.lock().await;
        out.write_all(&packet.bytes)
            .await
            .context("could not write to the RTMP client")
    }

    async fn role(&self) -> Role {
        self.role.lock().await.clone()
    }
}

/// Drive one RTMP connection.
async fn handle_connection(socket: tokio::net::TcpStream, handle: IngestHandle) -> Result<()> {
    let (mut read_half, mut write_half) = socket.into_split();

    // The handshake is a separate step that the session object does not
    // perform, and it must complete before any chunk is fed in.
    let mut handshake = Handshake::new(PeerType::Server);
    let p0p1 = handshake
        .generate_outbound_p0_and_p1()
        .context("could not build the RTMP handshake")?;
    write_half.write_all(&p0p1).await?;

    let mut carry: Vec<u8> = Vec::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = read_half.read(&mut buf).await?;
        if n == 0 {
            return Ok(());
        }
        carry.extend_from_slice(&buf[..n]);

        match handshake
            .process_bytes(&carry)
            .context("malformed RTMP handshake")?
        {
            HandshakeProcessResult::InProgress { response_bytes } => {
                write_half.write_all(&response_bytes).await?;
                carry.clear();
            }
            HandshakeProcessResult::Completed {
                response_bytes,
                remaining_bytes,
            } => {
                write_half.write_all(&response_bytes).await?;
                carry = remaining_bytes;
                break;
            }
        }
    }

    let (session, initial) = ServerSession::new(ServerSessionConfig::new())
        .context("could not start an RTMP session")?;
    for result in initial {
        if let ServerSessionResult::OutboundResponse(p) = result {
            write_half.write_all(&p.bytes).await?;
        }
    }

    // Subscribed before anything is read. The stream's sequence headers — the
    // ones carrying the codec configuration — arrive right after the publisher
    // starts, so a subscriber that attaches only once its play request has been
    // parsed has already missed them, and ffmpeg then fails with "No start code
    // is found".
    let media = handle.media.subscribe();

    let shared = Arc::new(Shared {
        session: Mutex::new(session),
        out: Mutex::new(write_half),
        role: Mutex::new(Role::Unknown),
        handle: handle.clone(),
        media: Mutex::new(media),
    });

    // Anything the handshake did not consume belongs to the session.
    if !carry.is_empty() {
        process(&shared, &carry).await?;
    }

    let mut relay = maybe_start_relay(&shared).await;

    loop {
        let n = read_half.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        process(&shared, &buf[..n]).await?;

        // A subscriber is only recognised once its play request arrives, which
        // is after this loop has started.
        if relay.is_none() {
            relay = maybe_start_relay(&shared).await;
        }
    }

    if let Some(task) = relay {
        task.abort();
    }
    match shared.role().await {
        Role::Publisher { stream_key } => {
            let mut state = handle.state.lock().await;
            state.publishing = false;
            state.stream_key.clear();
            drop(state);
            let _ = handle.events.send(IngestEvent::Unpublished { stream_key });
        }
        Role::Subscriber { .. } => {
            let mut state = handle.state.lock().await;
            state.subscribers = state.subscribers.saturating_sub(1);
        }
        Role::Unknown => {}
    }
    Ok(())
}

/// Start the relay task if this connection has turned out to be a subscriber.
async fn maybe_start_relay(shared: &Arc<Shared>) -> Option<tokio::task::JoinHandle<()>> {
    match shared.role().await {
        Role::Subscriber { stream_id } => {
            // The receiver has been buffering since the connection opened.
            let mut media = std::mem::replace(
                &mut *shared.media.lock().await,
                shared.handle.media.subscribe(),
            );
            let shared = shared.clone();
            Some(tokio::spawn(async move {
                loop {
                    let (data, timestamp, is_audio) = match media.recv().await {
                        Ok(Media::Audio { data, timestamp }) => (data, timestamp, true),
                        Ok(Media::Video { data, timestamp }) => (data, timestamp, false),
                        Err(broadcast::error::RecvError::Lagged(n)) => {
                            debug!(skipped = n, "subscriber fell behind");
                            continue;
                        }
                        Err(broadcast::error::RecvError::Closed) => break,
                    };

                    let packet = {
                        let mut session = shared.session.lock().await;
                        let ts = RtmpTimestamp::new(timestamp);
                        let built = if is_audio {
                            session.send_audio_data(stream_id, data, ts, false)
                        } else {
                            session.send_video_data(stream_id, data, ts, false)
                        };
                        match built {
                            Ok(p) => p,
                            Err(err) => {
                                debug!(error = %err, "could not build a relayed packet");
                                continue;
                            }
                        }
                    };

                    if shared.send_packet(&packet).await.is_err() {
                        break;
                    }
                }
                debug!("subscriber relay ended");
            }))
        }
        _ => None,
    }
}

/// Feed bytes into the session and act on the results.
async fn process(shared: &Arc<Shared>, bytes: &[u8]) -> Result<()> {
    let results = {
        let mut session = shared.session.lock().await;
        session
            .handle_input(bytes)
            .context("malformed RTMP data")?
    };

    for result in results {
        match result {
            ServerSessionResult::OutboundResponse(p) => {
                shared.send_packet(&p).await?;
            }
            ServerSessionResult::RaisedEvent(event) => {
                on_event(event, shared).await?;
            }
            ServerSessionResult::UnhandleableMessageReceived(msg) => {
                debug!(?msg, "unhandled RTMP message");
            }
        }
    }
    Ok(())
}

/// Act on one session event.
async fn on_event(event: ServerSessionEvent, shared: &Arc<Shared>) -> Result<()> {
    match event {
        ServerSessionEvent::ConnectionRequested { request_id, app_name } => {
            info!(app_name, "RTMP client connected");
            let results = shared.session.lock().await.accept_request(request_id)?;
            for r in results {
                if let ServerSessionResult::OutboundResponse(p) = r {
                    shared.send_packet(&p).await?;
                }
            }
        }

        ServerSessionEvent::PublishStreamRequested {
            request_id,
            app_name,
            stream_key,
            ..
        } => {
            // One publisher at a time: a second would interleave two speakers
            // into one transcript with no way to tell them apart.
            if shared.handle.state.lock().await.publishing {
                warn!(stream_key, "rejecting a second publisher");
                let results = shared.session.lock().await.reject_request(
                    request_id,
                    "NetStream.Publish.BadName",
                    "another stream is already publishing",
                )?;
                for r in results {
                    if let ServerSessionResult::OutboundResponse(p) = r {
                        shared.send_packet(&p).await?;
                    }
                }
                return Ok(());
            }

            info!(app_name, stream_key, "publisher accepted");
            let results = shared.session.lock().await.accept_request(request_id)?;
            for r in results {
                if let ServerSessionResult::OutboundResponse(p) = r {
                    shared.send_packet(&p).await?;
                }
            }

            {
                let mut state = shared.handle.state.lock().await;
                state.publishing = true;
                state.stream_key = stream_key.clone();
            }
            *shared.role.lock().await = Role::Publisher {
                stream_key: stream_key.clone(),
            };
            let _ = shared
                .handle
                .events
                .send(IngestEvent::Published { stream_key });
        }

        ServerSessionEvent::PlayStreamRequested {
            request_id,
            stream_id,
            stream_key,
            ..
        } => {
            info!(stream_key, "subscriber connected");
            let results = shared.session.lock().await.accept_request(request_id)?;
            for r in results {
                if let ServerSessionResult::OutboundResponse(p) = r {
                    shared.send_packet(&p).await?;
                }
            }
            *shared.role.lock().await = Role::Subscriber { stream_id };
            shared.handle.state.lock().await.subscribers += 1;
        }

        ServerSessionEvent::AudioDataReceived {
            data, timestamp, ..
        } => {
            let _ = shared.handle.media.send(Media::Audio {
                data,
                timestamp: timestamp.value,
            });
        }

        ServerSessionEvent::VideoDataReceived {
            data, timestamp, ..
        } => {
            let _ = shared.handle.media.send(Media::Video {
                data,
                timestamp: timestamp.value,
            });
        }

        ServerSessionEvent::PublishStreamFinished { stream_key, .. } => {
            info!(stream_key, "publisher stopped");
            let mut state = shared.handle.state.lock().await;
            state.publishing = false;
            state.stream_key.clear();
            drop(state);
            let _ = shared
                .handle
                .events
                .send(IngestEvent::Unpublished { stream_key });
        }

        other => debug!(?other, "RTMP event"),
    }
    Ok(())
}

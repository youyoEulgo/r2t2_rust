//! Integration test client for `r2t2 serve`.
//!
//! Streams a WAV file over the real WebSocket protocol, then EOS, and prints
//! the transcript assembled from the incremental `msg.text` values. This
//! exercises the same wire format `ws_client.py` uses, and needs no Python.
//!
//! Run the server first, then:
//!
//! ```text
//! cargo run --release --example ws_client -- --audio resources/test.wav
//! ```

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio_tungstenite::tungstenite::Message;

/// Must match the server.
const EOS: &str = "YOUDAO_ONETIME_ASR_STREAM_EOS";
const CHUNK_MS: usize = 160;

#[derive(Debug, Parser)]
struct Cli {
    /// WebSocket endpoint.
    #[arg(short = 'u', long = "uri", default_value = "ws://127.0.0.1:8272/asr_stream_api_v1")]
    uri: String,

    /// WAV file to stream.
    #[arg(short = 'a', long = "audio", default_value = "resources/test.wav")]
    audio: PathBuf,

    #[arg(short = 'l', long = "language", default_value = "Chinese")]
    language: String,

    /// Disable server-side VAD segmentation.
    #[arg(long = "no-vad")]
    no_vad: bool,
}

#[derive(Debug, Serialize)]
struct Header {
    #[serde(rename = "requestId")]
    request_id: String,
    language: String,
    use_vad: bool,
    channels: u16,
    sample_rate: u32,
    secret_key: String,
    mode: String,
}

#[derive(Debug, Deserialize)]
struct Envelope {
    status: String,
    msg: Option<Body>,
}

#[derive(Debug, Deserialize)]
struct Body {
    #[serde(default)]
    text: String,
    #[serde(default)]
    reset: bool,
    #[serde(default)]
    asr_cost_ms: Option<f64>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Read the WAV and keep the raw frames: the server accepts int16 little
    // endian PCM, which is exactly what a 16-bit WAV body already is.
    let mut reader = hound::WavReader::open(&cli.audio)
        .with_context(|| format!("could not open {}", cli.audio.display()))?;
    let spec = reader.spec();
    if spec.bits_per_sample != 16 {
        bail!("expected 16-bit PCM, got {}-bit", spec.bits_per_sample);
    }
    let samples: Vec<i16> = reader
        .into_samples::<i16>()
        .collect::<Result<Vec<_>, _>>()
        .context("bad WAV sample data")?;

    let pcm: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    let channels = spec.channels.max(1) as usize;
    let per_chunk = (spec.sample_rate as usize * CHUNK_MS / 1000) * channels * 2;

    let request_id = uuid_like();
    println!("requestId  = {request_id}");
    println!("audio      = {} ({} Hz, {} ch, {} samples)",
        cli.audio.display(), spec.sample_rate, spec.channels, samples.len());
    println!("uri        = {}", cli.uri);
    println!();

    let (mut ws, _) = tokio_tungstenite::connect_async(&cli.uri)
        .await
        .with_context(|| format!("could not connect to {}", cli.uri))?;

    let header = Header {
        request_id: request_id.clone(),
        language: cli.language.clone(),
        use_vad: !cli.no_vad,
        channels: spec.channels,
        sample_rate: spec.sample_rate,
        secret_key: "test0102".to_string(),
        mode: "slow".to_string(),
    };
    ws.send(Message::Text(serde_json::to_string(&header)?.into()))
        .await?;

    // Send audio in one task so receiving is never blocked by a full socket.
    let (mut sink, mut stream) = ws.split();
    let sender = tokio::spawn(async move {
        for chunk in pcm.chunks(per_chunk) {
            if sink.send(Message::Binary(chunk.to_vec().into())).await.is_err() {
                return;
            }
        }
        let _ = sink.send(Message::Text(EOS.into())).await;
    });

    let mut transcript = String::new();
    let mut updates = 0usize;
    let mut resets = 0usize;

    while let Some(msg) = stream.next().await {
        let msg = msg.context("websocket error")?;
        let text = match msg {
            Message::Text(t) => t.to_string(),
            Message::Close(_) => break,
            _ => continue,
        };

        let env: Envelope = serde_json::from_str(&text).context("malformed response")?;
        if env.status != "success" {
            eprintln!("server error: {text}");
            break;
        }
        let Some(body) = env.msg else { continue };

        if body.reset {
            resets += 1;
            println!("  --- segment boundary (reset) ---");
        }
        if !body.text.is_empty() {
            updates += 1;
            println!(
                "  [+{:>6.1}ms] {:?}",
                body.asr_cost_ms.unwrap_or(0.0),
                body.text
            );
            transcript.push_str(&body.text);
        }
    }

    let _ = sender.await;

    println!();
    println!("{}", "=".repeat(62));
    println!("updates: {updates}   resets: {resets}");
    println!("TRANSCRIPT: {transcript}");
    println!("{}", "=".repeat(62));
    Ok(())
}

/// A v4-shaped random id, without pulling in the `uuid` crate.
fn uuid_like() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id() as u128;
    let mix = nanos
        .wrapping_mul(6364136223846793005)
        .wrapping_add(pid.wrapping_mul(1442695040888963407));
    format!(
        "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
        (mix >> 96) as u32,
        (mix >> 80) as u16,
        (mix >> 64) as u16 & 0x0fff,
        ((mix >> 48) as u16 & 0x3fff) | 0x8000,
        (mix & 0xffffffffffff) as u64
    )
}

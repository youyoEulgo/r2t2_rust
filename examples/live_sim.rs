// Copyright 2026 youyoEulgo
// SPDX-License-Identifier: Apache-2.0

//! Simulate a live stream: send audio in real time, print text as it arrives.
//!
//! Unlike `ws_client`, which blasts the whole file as fast as the socket
//! accepts it, this paces the sends to wall-clock speed so the output shows
//! what a livestream viewer would actually see.
//!
//! ```text
//! cargo run --release --example live_sim -- --audio resources/wslc.wav
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use serde::Serialize;
use tokio_tungstenite::tungstenite::Message;

const EOS: &str = "YOUDAO_ONETIME_ASR_STREAM_EOS";

#[derive(Debug, Parser)]
struct Cli {
    #[arg(short = 'u', long = "uri", default_value = "ws://127.0.0.1:8272/asr_stream_api_v1")]
    uri: String,

    #[arg(short = 'a', long = "audio", default_value = "resources/test.wav")]
    audio: PathBuf,

    #[arg(short = 'l', long = "language", default_value = "Chinese")]
    language: String,

    /// Milliseconds of audio per send. 160 ms matches the reference client.
    #[arg(long = "frame-ms", default_value_t = 160)]
    frame_ms: u64,

    /// Play faster than real time by this factor.
    #[arg(long = "speed", default_value_t = 1.0)]
    speed: f64,
}

#[derive(Debug, Serialize)]
struct Header {
    #[serde(rename = "requestId")]
    request_id: String,
    language: String,
    use_vad: bool,
    channels: u16,
    sample_rate: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

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
    let per_frame = (spec.sample_rate as u64 * cli.frame_ms / 1000) as usize * channels * 2;
    let frame_interval = Duration::from_secs_f64(
        (cli.frame_ms as f64 / 1000.0) / cli.speed.max(0.01),
    );

    let total_secs = samples.len() as f64 / (spec.sample_rate as f64 * channels as f64);
    println!(
        "streaming {} ({total_secs:.1}s) at {}x, {}/frame",
        cli.audio.display(),
        cli.speed,
        cli.frame_ms
    );
    println!("{}", "-".repeat(60));

    let (mut ws, _) = tokio_tungstenite::connect_async(&cli.uri)
        .await
        .with_context(|| format!("could not connect to {}", cli.uri))?;

    ws.send(Message::Text(
        serde_json::to_string(&Header {
            request_id: format!("live-{}", std::process::id()),
            language: cli.language.clone(),
            use_vad: true,
            channels: spec.channels,
            sample_rate: spec.sample_rate,
        })?
        .into(),
    ))
    .await?;

    let (mut sink, mut stream) = ws.split();

    // Pace the sender to wall clock.
    let sender = tokio::spawn(async move {
        let start = Instant::now();
        for (i, chunk) in pcm.chunks(per_frame).enumerate() {
            // Sleep until this frame's scheduled time.
            let due = frame_interval * i as u32;
            let elapsed = start.elapsed();
            if due > elapsed {
                tokio::time::sleep(due - elapsed).await;
            }
            if sink.send(Message::Binary(chunk.to_vec().into())).await.is_err() {
                return;
            }
        }
        let _ = sink.send(Message::Text(EOS.into())).await;
    });

    let transcript_start = Instant::now();
    let mut transcript = String::new();

    while let Some(msg) = stream.next().await {
        let msg = msg.context("websocket error")?;
        let text = match msg {
            Message::Text(t) => t.to_string(),
            Message::Close(_) => break,
            _ => continue,
        };
        let Ok(env) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        let Some(body) = env.get("msg") else { continue };
        let chunk = body.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let reset = body.get("reset").and_then(|v| v.as_bool()).unwrap_or(false);
        let cost = body.get("asr_cost_ms").and_then(|v| v.as_f64()).unwrap_or(0.0);

        if !chunk.is_empty() {
            transcript.push_str(chunk);
            println!(
                "[{:>6.2}s] +{:>5.0}ms  {chunk}",
                transcript_start.elapsed().as_secs_f64(),
                cost
            );
        }
        if reset {
            println!("{:>10}--- segment boundary ---", "");
        }
    }

    let _ = sender.await;
    println!("{}", "-".repeat(60));
    println!("TRANSCRIPT: {transcript}");
    Ok(())
}

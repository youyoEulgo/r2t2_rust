// Copyright 2026 youyoEulgo
// SPDX-License-Identifier: Apache-2.0

//! Subscribe to the live subtitle stream and print it.
//!
//! This is what a caption renderer does: connect to `/ws/subtitles`, and print
//! each update. Useful on its own for checking that an incoming stream is
//! being transcribed, without opening a browser.
//!
//! Run the server first, start a stream into it (OBS, or `ffmpeg -re -i
//! something -f flv rtmp://127.0.0.1:1935/live`), then:
//!
//! ```text
//! cargo run --release --example sub_client
//! ```

use std::time::Duration;

use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde::Deserialize;
use tokio_tungstenite::tungstenite::Message;

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Update {
    Status {
        enabled: bool,
        active: bool,
        #[serde(default)]
        latest: String,
    },
    Subtitle {
        text: String,
        delta: String,
        reset: bool,
        at_ms: u64,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let url = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "ws://127.0.0.1:8272/ws/subtitles".to_string());

    println!("subscribing to {url}");
    let (mut ws, _) = tokio_tungstenite::connect_async(&url)
        .await
        .with_context(|| format!("could not connect to {url}"))?;

    let mut lines = 0usize;
    while let Some(message) = ws.next().await {
        let message = message.context("websocket error")?;
        let raw = match message {
            Message::Text(t) => t.to_string(),
            Message::Close(_) => break,
            _ => continue,
        };

        match serde_json::from_str::<Update>(&raw) {
            Ok(Update::Status {
                enabled,
                active,
                latest,
            }) => {
                println!(
                    "-- connected: subtitles {} | stream {} | so far: {latest:?}",
                    if enabled { "on" } else { "off" },
                    if active { "live" } else { "idle" },
                );
            }
            Ok(Update::Subtitle {
                text,
                delta,
                reset,
                at_ms,
            }) => {
                if reset {
                    println!("     -- segment boundary --");
                }
                if !delta.is_empty() {
                    lines += 1;
                    println!("[{:>7}ms] +{delta}", at_ms);
                }
                // The running text, so it is obvious whether deltas are
                // accumulating correctly.
                let _ = text;
            }
            Err(err) => eprintln!("unrecognised message ({err}): {raw}"),
        }
    }

    println!("-- stream ended after {lines} updates");
    tokio::time::sleep(Duration::from_millis(50)).await;
    Ok(())
}

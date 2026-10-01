// Copyright 2026 youyoEulgo
// SPDX-License-Identifier: Apache-2.0

//! Confucius4-R2T2 speech recognition, driven from Rust.
//!
//! The crate links llama.cpp's `libllama` and `libmtmd` through FFI -- the same
//! C++ libraries the reference Python decoder uses, minus the Python runtime
//! around them. No conda, no PyTorch, only the NVIDIA driver at runtime. The
//! layers:
//!
//! * [`engine`] -- loading the model and running a decode.
//! * [`stream`] -- the chunked streaming algorithm (Longest Stable Prefix).
//! * [`vad`] -- WebRTC voice activity detection, for segmenting a live stream.
//! * [`subtitle`] -- VAD segments to timed cues to SRT.
//! * [`quality`] -- repetition repair and hallucination detection.
//! * [`media`] -- decoding video and audio input via ffmpeg.
//! * [`model`] -- locating the language model and projector GGUF pair.
//! * [`paths`] -- the data directory, and fetching the model when missing.
//! * [`web`] -- the web interface: static assets and the upload API.
//! * [`audio`] -- decoding WAV input to the mono 16 kHz the model expects.
//! * [`prompt`] -- chat-template prompt construction.
//!
//! One binary builds on this, with a subcommand per mode: `r2t2 transcribe`,
//! `r2t2 serve`, and `r2t2 subtitle`.

pub mod audio;
pub mod cli;
pub mod engine;
pub mod media;
pub mod model;
pub mod paths;
pub mod prompt;
pub mod quality;
pub mod stream;
pub mod subtitle;
pub mod vad;
pub mod web;

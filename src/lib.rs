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
//! * [`audio`] -- decoding WAV input to the mono 16 kHz the model expects.
//! * [`prompt`] -- chat-template prompt construction.
//!
//! Three binaries build on this: `r2t2` is a file-oriented CLI, `r2t2-server`
//! serves the WebSocket protocol, and `r2t2-sub` generates subtitles.

pub mod audio;
pub mod engine;
pub mod media;
pub mod prompt;
pub mod quality;
pub mod stream;
pub mod subtitle;
pub mod vad;

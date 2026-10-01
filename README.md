# r2t2 — Confucius4-R2T2 speech recognition without a Python runtime

Streaming and one-shot ASR for the [Confucius4-R2T2](https://huggingface.co/netease-youdao/Confucius4-R2T2)
model, driven directly from Rust.

This is not a pure-Rust inference engine: it links llama.cpp's `libllama` and
`libmtmd`, the same C++ libraries the reference Python decoder uses through
pybind11. What is gone is the Python runtime around them — no PyTorch, no conda,
no HuggingFace runtime — so model loading, audio encoding and decoding all
happen in process. The only thing a deployment cannot bundle is the NVIDIA
driver.

One binary, one subcommand per mode:

| | |
|---|---|
| `r2t2 transcribe` | transcribe a file, one-shot or streaming |
| `r2t2 serve` | WebSocket service for live audio |
| `r2t2 subtitle` | generate `.srt` subtitles from a video |

## Why

The reference implementation is Python and works, but it carries an 11 GB
environment whose startup cost dominates short runs. Measured here on the sample
audio:

| | Python | Rust |
|---|---|---|
| import / startup | **11.2 s** | ~0.05 s |
| model load | 0.77 s | 0.7 s |
| inference | 0.26 s | 0.26 s |

The inference time is identical — both run the same C++ through llama.cpp. What
disappears is the cost of importing the Python ecosystem, 87% of which was
`import torch` for code paths the llama.cpp route never calls. For a batch job
that is a 9x difference end to end; for a long-running daemon it is mostly a
startup and footprint difference.

## Building

```sh
cargo build --release
```

Produces `target/release/r2t2`.

Requirements:

| | |
|---|---|
| Rust | 1.85+ (edition 2024) |
| CUDA toolkit | for the GPU backend, e.g. `/opt/cuda` |
| NVIDIA driver | at runtime |
| `libclang` | build only, for bindgen |
| `ffmpeg` | only for `r2t2 subtitle` on non-WAV input |

The llama.cpp libraries are vendored in [`vendor/`](vendor/README.md) — no
llama.cpp checkout is needed. Read that file before rebuilding them; it records
why they were built locally and what that means for portability.

## Model weights

Not included; download separately.

```sh
# llama.cpp / GGUF weights — everything below uses these
hf download netease-youdao/Confucius4-R2T2-GGUF \
    --include "Confucius4-R2T2-Q8_0.gguf" "mmproj-Confucius4-R2T2-Q8_0.gguf" \
    --local-dir checkpoints/gguf
```

The GGUF directory must hold **exactly one** `mmproj*.gguf` and **exactly one**
other `*.gguf`. That rule catches the common mistake of dropping several
quantisations into one directory, which would otherwise silently pick one.

## Usage

### Transcribe a file

```sh
r2t2 transcribe -i audio.wav                        # to stdout
r2t2 transcribe -i audio.wav -o transcript.txt
r2t2 transcribe -i audio.wav -l English
r2t2 transcribe -i audio.wav --stream --show-updates   # incremental output
```

### Live audio over WebSocket

```sh
r2t2 serve --gguf-dir checkpoints/gguf --port 8272
```

Speaks the same protocol as the reference Python server, so existing clients
work unchanged:

1. The client sends a JSON header; `requestId` is required.
2. Binary frames carry raw little-endian `int16` mono PCM at 16 kHz. The first
   frame may instead be a whole WAV file.
3. `msg.text` in each reply is the **new** text since the last message.
4. A text frame equal to `YOUDAO_ONETIME_ASR_STREAM_EOS` ends the stream.

Voice activity detection segments the stream: on sustained silence the ASR state
is reset and the reply carries `"reset": true`. Without it a long connection
would accumulate audio without bound and never break the text naturally.

Try it against the sample audio:

```sh
cargo run --release --example ws_client -- --audio resources/test.wav
```

### Generate subtitles

```sh
r2t2 subtitle -i movie.mp4 -o movie.srt
r2t2 subtitle -i movie.mp4 --print                  # cues to stdout
r2t2 subtitle -i movie.mp4 -c "会话容器 WSLC"        # hotwords
```

Video is decoded through `ffmpeg`; WAV is read in process.

Timestamps come from **voice activity detection, not the model**. The model
provides words, the detector provides timing: it segments the whole file first,
then each segment is transcribed independently. This also means a segment's
duration bounds its cue timings, so they cannot overlap.

Long segments are split again on punctuation and length. A split never lands
inside a Latin word, and never exceeds `--max-chars`.

To mux the result into a shareable file:

```sh
ffmpeg -i movie.mp4 -i movie.srt -map 0 -map 1 \
    -c:v copy -c:a copy -c:s srt movie.subbed.mkv
```

## Quality guards

Models fail in specific, recognisable ways on hard audio. `src/quality.rs`
handles two of them:

- **Repetition** — a stuck decoder emitting `的的的的的的` or `Okay. Okay. Okay.`
  is collapsed. Runs at or below `--repeat-threshold` (default 5) are left
  alone, so ordinary doubled characters survive.
- **Hallucination** — a phrase loop running to the *end* of a segment is the
  characteristic shape of invented text on silence or noise. Such segments are
  dropped, because a wrong cue is worse than a missing one. Use
  `--keep-hallucinations` to see what is being discarded.

A third failure is out of reach for heuristics: fluent but wrong text, such as
`会话容器` heard as `绘画容器`. That needs hotwords (`-c`) or a person.

## Layout

```
src/
  engine.rs      llama.cpp FFI, RAII wrappers, decode loop
  stream.rs      streaming algorithm (Longest Stable Prefix)
  vad.rs         WebRTC VAD and the segment state machine
  subtitle.rs    VAD segmentation -> cues -> SRT
  quality.rs     repetition repair, hallucination detection
  media.rs       ffmpeg-backed media decoding
  audio.rs       WAV decoding and resampling
  prompt.rs      chat-template prompt construction
  model.rs       locating the model and projector GGUF pair
  cli/
    mod.rs       shared flags and subcommand dispatch
    transcribe.rs
    serve.rs
    subtitle.rs
  main.rs        entry point
examples/
  ws_client.rs   protocol-level integration client
vendor/          llama.cpp headers and prebuilt libraries
```

Everything the three modes agree on — where the model lives, which language to
expect, how much context to give llama.cpp — is defined once in `cli/mod.rs` and
flattened into each subcommand, so the flags cannot drift apart between modes.

## Streaming algorithm

`stream.rs` ports `R2T2ASRModel.streaming_transcribe`. Each step re-feeds **all**
audio heard so far and prompts with the previous transcript minus the last
`--unfixed-token-num` tokens, so the model can still revise the boundary while
everything before it stays fixed. Decoding `cur_ids[..len-k]` can split a
multi-byte character, so the rollback widens until no replacement character
remains.

Two caveats worth knowing:

- The reference uses the HuggingFace tokenizer; this uses llama.cpp's, because
  dropping the Python dependency is the point. Transcripts match on the sample
  audio, but token boundaries differ, so the rollback granularity is not
  bit-identical in general.
- `fixed_text` is *not* strictly append-only. The reference emits one regression
  in 32 updates on the sample audio, and this port reproduces it exactly. The
  instability comes from the model revising a position before the rollback
  window, which the window cannot cover — it is inherent to the approach.

## Testing

```sh
cargo test
```

Covers prompt construction byte-for-byte against the reference, output parsing,
the VAD state machine (including WebRTC VAD's ~3-frame hangover), cue splitting,
and the quality guards.

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
| `r2t2 transcribe` | a file to subtitles (SRT) or plain text (TXT) |
| `r2t2 mux` | embed a subtitle file into a video, producing an MKV |
| `r2t2 serve` | web interface, plus a WebSocket API for live audio |

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
| `ffmpeg` | for non-WAV input, and for `r2t2 mux` |

The llama.cpp libraries are vendored in [`vendor/`](vendor/README.md) — no
llama.cpp checkout is needed. Read that file before rebuilding them; it records
why they were built locally and what that means for portability.

## Model weights

Everything lives under one directory, following the XDG data specification:

```text
~/.local/share/r2t2/
  models/     the GGUF pair
  work/       uploads and results (the web interface)
```

`$XDG_DATA_HOME` relocates the whole tree.

On first use the program finds `models/` empty and **offers to download** the
2.1 GB Q8_0 pair. Answer `y` and it fetches with a progress bar:

```sh
$ r2t2 transcribe -i movie.mp4
No model found in /home/you/.local/share/r2t2/models.
Download Confucius4-R2T2-Q8_0.gguf (about 2.1 GB) from HuggingFace now? [y/N]
```

It will not download unasked, and when stdin is not a terminal (a pipe, a
script, CI) it skips the prompt entirely and prints the manual command instead,
so an unattended run fails with instructions rather than hanging. `--yes`
answers the prompt in advance.

Behind a mirror, set `HF_ENDPOINT`:

```sh
HF_ENDPOINT=https://hf-mirror.com r2t2 transcribe -i movie.mp4
```

To place the weights elsewhere, pass `--gguf-dir`; an explicit path is trusted
as-is and never triggers a download:

```sh
hf download netease-youdao/Confucius4-R2T2-GGUF \
    --include "Confucius4-R2T2-Q8_0.gguf" "mmproj-Confucius4-R2T2-Q8_0.gguf" \
    --local-dir /somewhere/gguf
r2t2 transcribe -i movie.mp4 --gguf-dir /somewhere/gguf
```

The directory must hold **exactly one** `mmproj*.gguf` and **exactly one**
other `*.gguf`. That rule catches the common mistake of dropping several
quantisations into one directory, which would otherwise silently pick one.

## Usage

Results go to **stdout**; diagnostics go to **stderr** and only with
`--verbose`. Nothing is written to a file unless `-o` asks for it, so a shell
redirect and `-o` are interchangeable:

```sh
r2t2 transcribe -i audio.wav > transcript.txt
r2t2 transcribe -i audio.wav -o transcript.txt      # the same thing
```

By default llama.cpp's own logging is switched off. It narrates every step —
well over a thousand lines for a single short file — and would bury this
program's messages. `--verbose` keeps it.

`-v` prints the version; verbosity is `--verbose` only.

### Transcribe a file

The default output is **SRT**, because it carries the timings and plain text can
always be derived from it. `--format txt` drops the timings instead.

```sh
r2t2 transcribe -i movie.mp4                          # SRT to stdout
r2t2 transcribe -i movie.mp4 --format txt             # plain text
r2t2 transcribe -i movie.mp4 -o movie.srt             # straight to a file
r2t2 transcribe -i audio.wav -l English               # language hint
r2t2 transcribe -i movie.mp4 -c "会话容器 WSLC"        # hotwords
r2t2 transcribe -i audio.wav --format txt --stream    # chunked decoding
```

Audio and video take the same path: the input is decoded to 16 kHz mono first,
so a video container needs no separate handling.

### Embed subtitles into a video

```sh
r2t2 mux --video movie.mp4 --subtitle movie.srt
r2t2 mux --video movie.mp4 --subtitle corrected.srt --output movie.mkv
```

Video and audio are **stream-copied**, never re-encoded, so this takes about a
second regardless of length. The subtitle track is tagged with `--language` so
players can select it by name.

This is a separate subcommand rather than a flag on `transcribe` because it does
no recognition: the subtitle file may have been corrected by something else, and
combining the two is an independent step.

### Web interface

```sh
r2t2 serve --gguf-dir checkpoints/gguf --port 8272
# then open http://127.0.0.1:8272
```

The **转写** panel turns a file into a `.txt` and an `.srt`. The **合并字幕**
panel is separate: give it a video and a subtitle file — one you corrected
elsewhere, for instance — and it produces an MKV with the subtitles embedded.
Keeping them apart means a corrected subtitle can be re-muxed without
re-running recognition.

Language, hotwords, VAD sensitivity, cue length and the quality guards are all
adjustable in the page.

Transcription takes longer than an HTTP request should stay open, so an upload
returns a job id immediately and the page polls for progress. Jobs run one at a
time — a single GPU has one context — and a queued job reports its position.

The frontend lives in `ui/` and is embedded into the binary at build time, so a
deployment is one file:

```sh
cd ui && bun install && bun run build && cd ..
cargo build --release
```

`ui/dist` is committed, so building the Rust binary does **not** require a
JavaScript toolchain. For frontend work, `bun run dev` in `ui/` serves the page
with hot reload and proxies `/api` to a running `r2t2 serve`.

`--no-web` serves only the WebSocket API, and `--work-dir` chooses where uploads
and results are kept (the system temporary directory by default, which is
cleared on reboot).

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
  paths.rs       the data directory, and fetching the model when missing
  web.rs         web interface: static assets, upload API, background jobs
  cli/
    mod.rs       shared flags and subcommand dispatch
    transcribe.rs
    mux.rs
    serve.rs
  main.rs        entry point
ui/              frontend (Bun + Vite + Vue 3); dist/ is embedded at build time
examples/
  ws_client.rs   protocol-level integration client
  live_sim.rs    paced streaming client, for watching live output
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

## Licence

This project is **Apache-2.0** ([LICENSE](LICENSE)), the same licence as the
upstream code it derives from.

Four source files are derived from NetEase Youdao's Confucius4-R2T2, which is
Apache-2.0, and one function from Alibaba's Qwen3-ASR, likewise Apache-2.0.
Each says so at the top of the file, and [NOTICE](NOTICE) records which parts
came from where and what was changed.

**The model weights are not covered by this licence.** They are published
separately by NetEase Youdao under the [NetEase Youdao Model Use License
Agreement](https://github.com/netease-youdao/Confucius4-R2T2/blob/master/MODEL_LICENSE),
and downloading them means accepting its terms. Two of its conditions are worth
knowing before you deploy anything:

- **Commercial scale.** If you or your affiliates exceed 100 million monthly
  active users, or RMB 1 billion in annual revenue, you need a separate licence
  from NetEase Youdao.
- **No distillation.** The model may not be used to improve another AI model,
  except a non-commercial one.

It also disallows high-risk deployments such as medical diagnosis, autonomous
driving, military use, and large-scale biometric surveillance.

`vendor/lib` and `vendor/include` are llama.cpp build products, redistributed
under the MIT licence (full text in NOTICE).

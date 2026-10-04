# r2t2

Speech recognition with [Confucius4-R2T2](https://huggingface.co/netease-youdao/Confucius4-R2T2),
running as a single native executable.

| mode | what it does |
|---|---|
| `r2t2 transcribe` | an audio or video file → subtitles (SRT) or plain text (TXT) |
| `r2t2 mux` | an SRT → embedded subtitle track in an MKV |
| `r2t2 serve` | live captions from an OBS stream, plus a web interface |

Running `r2t2` with no arguments starts `serve`, prints the address and opens it
in a browser.

---

## Dependencies

### Build

| | needed for | Linux | macOS | Windows |
|---|---|---|---|---|
| Rust 1.85+ (edition 2024) | everything | `rustup` | `rustup` | `rustup`, MSVC toolchain |
| `cmake` | building llama.cpp | `apt install cmake` | `brew install cmake` | installer, on `PATH` |
| C/C++ toolchain | building llama.cpp | `build-essential` | `xcode-select --install` | VS Build Tools, **Desktop development with C++** |
| `libclang` | generating FFI bindings | `apt install libclang-dev` | included with Xcode | LLVM installer, on `PATH` |
| OpenMP | llama.cpp threading | `libgomp` (with GCC) | `brew install libomp` | included with MSVC |
| CUDA toolkit | GPU backend (optional) | `nvcc` on `PATH` | not applicable | CUDA installer |

The CUDA toolkit is optional at build time. Without `nvcc` the build is CPU-only
and needs no CUDA at all. The GPU backend is chosen automatically: CUDA on Linux
when `nvcc` is present, Metal on macOS, CPU otherwise.

### Runtime

| | needed for | Linux | macOS | Windows |
|---|---|---|---|---|
| `ffmpeg` | **live captions; any non-WAV input; `r2t2 mux`** | `apt install ffmpeg` | `brew install ffmpeg` | on `PATH` |
| `ffprobe` | reading media properties | ships with ffmpeg | ships with ffmpeg | ships with ffmpeg |
| CUDA toolkit libraries | **only for a GPU build** | see below | not applicable | CUDA installer |
| NVIDIA driver | **only for a GPU build** | `nvidia-utils` | not applicable | NVIDIA driver |

**Plain WAV input does not need ffmpeg**: 16-bit PCM WAV is decoded in process.
Every other format, and every live stream, is decoded by `ffmpeg`.

#### GPU runtime libraries

A build made with CUDA links these dynamically. They come from two different
places and it matters which:

| library | provided by | install |
|---|---|---|
| `libcuda.so.1` | the NVIDIA **driver** | `nvidia-utils` (Linux) or the driver package |
| `libcudart.so.*` | the **CUDA toolkit** | the CUDA toolkit |
| `libcublas.so.*` | the **CUDA toolkit** | the CUDA toolkit |
| `libcublasLt.so.*` | the **CUDA toolkit** | the CUDA toolkit |

Installing the driver alone is **not** enough for a CUDA build: cuBLAS is part
of the toolkit, and NVIDIA ships no static version of it. On a machine with only
the driver, install the toolkit or build a CPU-only binary:

```sh
R2T2_CUDA=0 cargo build --release
```

llama.cpp's own libraries are linked into the executable, so it needs no
directory of `.so` or `.dll` files beside it and works after being copied
elsewhere.

---

## Building

```sh
cargo build --release
```

Produces `target/release/r2t2` (about 113 MB with CUDA, less without). The first
build clones llama.cpp at a pinned revision into `third_party/` and compiles it,
which takes several minutes; later builds reuse the result.

To install it somewhere on `PATH`:

```sh
install -Dm755 target/release/r2t2 ~/.local/bin/r2t2
```

Build options, all optional:

| variable | effect |
|---|---|
| `R2T2_CUDA=0` | CPU-only build, no CUDA dependency at all |
| `R2T2_CUDA_ARCHS` | override the GPU architecture list (default `75-virtual;80-virtual;86-real;89-real`) |
| `R2T2_LIB_DIR` | link prebuilt llama.cpp libraries instead of compiling |
| `R2T2_LLAMA_DIR` | use an existing llama.cpp checkout |
| `R2T2_FORCE_REBUILD=1` | recompile llama.cpp even if cached |

**After changing any of these, remove the cached llama.cpp build**, or the old
configuration is reused:

```sh
rm -rf target/*/build/r2t2_rust-*/out/llama.cpp-build
```

---

## Model weights

Two files are required, about 2.1 GB in total:

```
Confucius4-R2T2-Q8_0.gguf
mmproj-Confucius4-R2T2-Q8_0.gguf
```

They live in `~/.local/share/r2t2/models` (`%APPDATA%\r2t2\models` on Windows,
`$XDG_DATA_HOME/r2t2/models` if that variable is set).

The program is usable without them: `r2t2 serve` starts, and the web interface
offers a download with a progress bar and a cancel button. The model is loaded
on first use, so a finished download needs no restart.

To fetch them by hand instead:

```sh
dir=~/.local/share/r2t2/models
mkdir -p "$dir" && cd "$dir"

for f in Confucius4-R2T2-Q8_0.gguf mmproj-Confucius4-R2T2-Q8_0.gguf; do
    curl -L -O "https://huggingface.co/netease-youdao/Confucius4-R2T2-GGUF/resolve/main/$f"
done
```

Set `HF_ENDPOINT` to a mirror if huggingface.co is unreachable. The built-in
downloader is plain HTTP and needs no Python.

Point `--gguf-dir` elsewhere to use a different directory; an explicit path is
trusted as-is and never triggers a download. It must hold **exactly one**
`mmproj*.gguf` and **exactly one** other `*.gguf`.

---

## Usage

Results go to **stdout**; diagnostics go to **stderr**, and only with
`--verbose`. Nothing is written to a file unless `-o` asks for it, so a shell
redirect and `-o` are interchangeable.

### Transcribe a file

```sh
r2t2 transcribe -i movie.mp4                       # SRT to stdout
r2t2 transcribe -i movie.mp4 --format txt          # plain text
r2t2 transcribe -i movie.mp4 -o movie.srt          # straight to a file
r2t2 transcribe -i audio.wav --stream              # incremental output
```

| option | meaning |
|---|---|
| `-i, --input` | file to transcribe (audio or video) |
| `-o, --output` | write to a file instead of stdout |
| `--format` | `srt` (default) or `txt` |
| `--stream` | emit incrementally rather than after the whole file |
| `--show-updates` | with `--stream`, show each partial revision |
| `--max-tokens` | tokens per decode step, default `512` |

### Embed subtitles into a video

```sh
r2t2 mux --video movie.mp4 --subtitle movie.srt --output movie.mkv
```

| option | meaning |
|---|---|
| `--video` | the video file |
| `--subtitle` | the SRT to embed |
| `--output` | output path (`.mkv`) |
| `-l, --language` | language tag for the subtitle track, default `Chinese` |

### Live captions from a stream

```sh
r2t2 serve
```

It prints the interface address and opens it in a browser. Point OBS at it:

```
OBS → 设置 → 推流 → 服务「自定义」
      服务器   rtmp://<host>:1935/live
      串流密钥 （留空）
```

Video in the stream is ignored; only the audio is transcribed.

Two consumers are available:

| consumer | address | purpose |
|---|---|---|
| programs | `ws://127.0.0.1:8272/ws/subtitles` | JSON subtitle stream |
| people / OBS | `http://127.0.0.1:8272/live` | CC-style caption overlay |

#### Subtitle protocol

`/ws/subtitles` sends one JSON object per line. On connect:

```json
{"type":"status","enabled":true,"active":true,"latest":"已有字幕"}
```

Then one object per update:

```json
{"type":"subtitle","text":"完整当前文本","delta":"本次新增","reset":false,"at_ms":1532}
```

| field | meaning |
|---|---|
| `text` | the authoritative complete current line |
| `delta` | only what was added since the last message |
| `reset` | a new segment began; the previous line is finished |
| `at_ms` | milliseconds since the current publisher connected |

A client may connect at any time; `status.latest` carries the current text.
Appearance changes arrive on the same connection as `{"type":"caption",…}`.

#### Caption appearance

Set in the web interface, saved to `~/.local/share/r2t2/config.toml`:

| setting | default | meaning |
|---|---|---|
| 显示行数 | 2 | lines kept on screen |
| 每行字数 | 20 | characters before a line is pushed up |
| 字号 | 48 | pixels |
| 文字颜色 | `#ffffff` | |
| 距底部 | `6%` | distance from the bottom |
| 不显示背景条 | off | for compositing onto video |

Line length counts two Latin letters as one CJK character. Changes are pushed to
open overlays over the existing WebSocket, so OBS needs no refresh.

### Web interface

Served by `r2t2 serve` at `http://127.0.0.1:8272`:

| tab | |
|---|---|
| 转写 | upload a file, get TXT and SRT |
| 合并字幕 | embed an SRT into a video |
| 直播字幕 | live stream state, caption appearance |

### Running with no arguments

```sh
r2t2
```

is equivalent to `r2t2 serve`. Server flags are accepted before the subcommand
as well as after it:

```sh
r2t2 --port 9000          # same as: r2t2 serve --port 9000
r2t2 --no-rtmp            # WebSocket ingest only
r2t2 --open=false         # do not launch a browser
```

### Server options

| option | default | meaning |
|---|---|---|
| `-p, --port` | `8272` | HTTP and WebSocket port |
| `--bind` | `0.0.0.0` | interface to bind |
| `--rtmp-port` | `1935` | RTMP ingest port |
| `--no-rtmp` | | disable RTMP; WebSocket ingest only |
| `--no-subtitles` | | accept RTMP audio without transcribing it |
| `--no-web` | | WebSocket API only, no interface |
| `--work-dir` | `~/.local/share/r2t2/work` | where uploads and results are kept |
| `--open` | `true` | open a browser on startup |

### Shared options

Accepted by every mode:

| option | default | meaning |
|---|---|---|
| `--gguf-dir` | `~/.local/share/r2t2/models` | where the weights are |
| `-l, --language` | `Chinese` | language hint |
| `--auto-language` | | let the model detect the language |
| `-c, --context` | | hotwords or context, e.g. names and terminology |
| `--n-ctx` | `8192` | context size |
| `--n-batch` | `2048` | batch size |
| `--n-threads` | `16` | CPU threads |
| `--cpu-only` | | do not use the GPU |
| `-y, --yes` | | download the model without prompting |
| `--verbose` | | progress and timing on stderr |

`-v` prints the version; verbosity is `--verbose` only.

### Segmentation and quality

| option | default | meaning |
|---|---|---|
| `--vad-sensitivity` | `aggressive` | `quality`, `lowbitrate`, `aggressive`, `veryaggressive` |
| `--vad-min-silence-ms` | `400` | silence that ends a segment |
| `--vad-min-speech-ms` | `160` | speech that starts one |
| `--max-chars` | `24` | longest subtitle line |
| `--max-seconds` | `8` | longest subtitle duration |
| `--repeat-threshold` | `5` | repeats before output counts as stuck |
| `--keep-hallucinations` | | keep output the guards would discard |
| `--unfixed-token-num` | `1` | tokens left revisable at the streaming boundary |
| `--chunk-ms` | `160` | streaming chunk size |
| `--lookahead-ms` | `160` | streaming lookahead |

Timings come from **voice activity detection, not the model**: the model
supplies words, the detector supplies when they were said. Segments are split
further on punctuation and length, and a split never lands inside a Latin word.

### WebSocket ingest

`r2t2 serve` also exposes `ws://<host>:8272/asr_stream_api_v1`, the same protocol
as the reference Python server:

1. The client sends a JSON header; `requestId` is required.
2. Binary frames carry raw little-endian `int16` mono PCM at 16 kHz. The first
   frame may instead be a whole WAV file.
3. `msg.text` in each reply is the **new** text since the last message.
4. A text frame equal to `YOUDAO_ONETIME_ASR_STREAM_EOS` ends the stream.

```sh
cargo run --release --example ws_client -- --audio resources/test.wav
```

### Checking the live path without OBS

```sh
# terminal 1
cargo run --release --example sub_client

# terminal 2
ffmpeg -re -i resources/test.wav -c:a aac -f flv rtmp://127.0.0.1:1935/live
```

---

## Environment variables

| variable | effect |
|---|---|
| `HF_ENDPOINT` | model download mirror, e.g. `https://hf-mirror.com` |
| `XDG_DATA_HOME` | relocates the data directory (Linux, macOS) |
| `RUST_LOG` | log filter; `--verbose` sets `debug` |

Build-time variables are listed under [Building](#building).

---

## Known limits

**Speech only, not music.** On songs the model misrecognises lyrics, voice
activity detection cannot find the gaps between phrases because a backing track
never falls silent, and instrumental passages attract repeated hallucinated
text. Hotwords change the output but do not make it correct, and once the
timings are wrong no post-processing recovers them. Subtitles for sung material
need forced alignment against a known lyric sheet, which is a different tool.

**One stream at a time.** A single GPU holds one model, and the process shares
it. File transcription, WebSocket ingest and RTMP ingest take turns; a second
RTMP publisher is rejected rather than mixed into the first transcript.

**The RTMP ingest is for a local network.** It accepts one publisher, has no
authentication, and does not encrypt.

**The work directory grows.** Every upload is kept, files and results alike,
under `--work-dir`, one directory per job, and nothing removes them. Large
inputs accumulate; delete the directory when the results have been collected.
This is worth knowing before pointing the interface at a long video.

---

## Layout

```
src/
  audio.rs       resampling and WAV decoding
  cli/           the command-line surface, one file per mode
  config.rs      the configuration file
  engine.rs      FFI, and the RAII wrappers around llama.cpp
  lazy_engine.rs the model, loaded on first use
  live.rs        live subtitles: a stream to published text
  media.rs       ffmpeg and ffprobe
  model.rs       locating the GGUF pair
  paths.rs       data directory, and downloading the weights
  prompt.rs      the prompt the model is given
  quality.rs     repetition repair and hallucination detection
  rtmp.rs        RTMP ingest
  stream.rs      the streaming algorithm
  subtitle.rs    segmentation and SRT generation
  vad.rs         voice activity detection
  web.rs         the web interface and its API
ui/              frontend (Bun + Vite + Vue 3); dist/ is embedded at build time
  index.html     the console
  live.html      the caption overlay
examples/
  ws_client.rs   protocol-level integration client
  live_sim.rs    paced streaming client
  sub_client.rs  subscribes to /ws/subtitles and prints captions
third_party/     llama.cpp checkout, cloned on first build (not tracked)
```

---

## Testing

```sh
cargo test
```

Covers prompt construction byte-for-byte against the reference, output parsing,
the VAD state machine, cue splitting, and the quality guards.

---

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
and downloading them means accepting its terms. Two of its conditions matter
before deploying anything:

- **Commercial scale.** More than 100 million monthly active users, or RMB 1
  billion in annual revenue, requires a separate licence from NetEase Youdao.
- **No distillation.** The model may not be used to improve another AI model,
  except a non-commercial one.

It also disallows high-risk deployments such as medical diagnosis, autonomous
driving, military use, and large-scale biometric surveillance.

llama.cpp is compiled at build time under the MIT licence; its full text is
reproduced in NOTICE.

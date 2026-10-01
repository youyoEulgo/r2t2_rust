//! `r2t2 transcribe` — turn an audio file into text.

use std::path::PathBuf;

use anyhow::{bail, Context as _, Result};
use clap::Args;

use crate::audio;
use crate::cli::CommonArgs;
use crate::engine::{Engine, EngineConfig};
use crate::prompt;
use crate::stream::StreamEngine;

#[derive(Debug, Args)]
pub struct TranscribeArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    /// Input audio file (WAV; any sample rate or channel count).
    #[arg(short = 'i', long = "input", value_name = "FILE")]
    pub input: PathBuf,

    /// Maximum tokens to generate.
    #[arg(long = "max-tokens", value_name = "N", default_value_t = 256)]
    pub max_tokens: i32,

    /// Decode in streaming mode, one chunk at a time, instead of in one shot.
    ///
    /// The final text is the same; use `--show-updates` to watch it build.
    #[arg(long = "stream")]
    pub stream: bool,

    /// Streaming chunk size in milliseconds.
    #[arg(long = "chunk-ms", value_name = "MS", default_value_t = 160)]
    pub chunk_ms: u32,

    /// Streaming lookahead for the first chunk, in milliseconds.
    #[arg(long = "lookahead-ms", value_name = "MS", default_value_t = 160)]
    pub lookahead_ms: u32,

    /// Trailing tokens left unfixed when prompting (rollback window).
    #[arg(long = "unfixed-token-num", value_name = "N", default_value_t = 1)]
    pub unfixed_token_num: usize,

    /// Print each incremental update to stderr as it is produced.
    #[arg(long = "show-updates")]
    pub show_updates: bool,
}

pub fn run(args: &TranscribeArgs) -> Result<()> {
    let (model, mmproj) = args.common.resolve_model()?;

    let samples = audio::load_wav_16k_mono(&args.input)?;
    if samples.is_empty() {
        bail!("no audio samples found in {}", args.input.display());
    }
    if args.common.verbose {
        let secs = samples.len() as f64 / audio::TARGET_SAMPLE_RATE as f64;
        eprintln!("audio  : {} samples ({secs:.2}s)", samples.len());
    }

    let cfg = args.common.engine_config(model, mmproj);
    if args.common.verbose {
        eprintln!("loading model...");
    }

    let text = if args.stream {
        streaming(args, &cfg, &samples)?
    } else {
        one_shot(args, &cfg, &samples)?
    };

    // The transcript is the program's only stdout output, so a redirect or a
    // pipe captures exactly the result and nothing else.
    println!("{}", text.trim());
    Ok(())
}

/// One-shot: the whole file in a single decode.
fn one_shot(args: &TranscribeArgs, cfg: &EngineConfig, samples: &[f32]) -> Result<String> {
    let prompt = prompt::build(
        &args.common.context,
        args.common.forced_language(),
        "",
    );
    let engine = Engine::load(cfg).context("failed to initialise the llama.cpp engine")?;

    if args.common.verbose {
        eprintln!("transcribing...");
    }
    let result = engine
        .transcribe(samples, &prompt, args.max_tokens)
        .context("transcription failed")?;

    if args.common.verbose {
        eprintln!("finish : {}", result.finish_reason);
        eprintln!("tokens : {}", result.token_ids.len());
    }
    Ok(result.text)
}

/// Streaming: feed the audio in fixed-size steps.
///
/// The first step carries an extra `--lookahead-ms` of audio and the chunk size
/// is widened to match, so the opening decode has more context than later ones.
/// The reference driver does the same.
fn streaming(args: &TranscribeArgs, cfg: &EngineConfig, samples: &[f32]) -> Result<String> {
    let engine = StreamEngine::load(cfg, args.max_tokens)
        .context("failed to initialise the streaming engine")?;

    let sr = audio::TARGET_SAMPLE_RATE;
    let chunk_size_sec = args.chunk_ms as f32 / 1000.0;
    let mut state = engine.init_state(
        &args.common.context,
        args.common.forced_language(),
        // 0 means the prefix is used from the very first chunk.
        0,
        args.unfixed_token_num,
        chunk_size_sec,
    );

    let step = ((args.chunk_ms as f32 / 1000.0) * sr as f32).round() as usize;
    let lookahead = ((args.lookahead_ms as f32 / 1000.0) * sr as f32).round() as usize;

    let mut pos = 0usize;
    let mut first = true;
    let mut updates = 0usize;

    while pos < samples.len() {
        let (seg, new_chunk_sec) = if first {
            let end = (pos + step + lookahead).min(samples.len());
            (&samples[pos..end], (step + lookahead) as f32 / sr as f32)
        } else {
            let end = (pos + step).min(samples.len());
            (&samples[pos..end], chunk_size_sec)
        };
        pos += seg.len();
        first = false;

        state.chunk_size_sec = new_chunk_sec;
        state.chunk_size_samples = ((new_chunk_sec * sr as f32).round() as usize).max(1);

        match engine.push(seg, &mut state)? {
            Some((_text, fixed)) => {
                updates += 1;
                if args.show_updates {
                    eprintln!("text={fixed}");
                }
            }
            None => {
                if args.show_updates {
                    eprintln!("text=");
                }
            }
        }
    }

    let final_text = engine.finish(&mut state)?;
    if args.common.verbose {
        eprintln!("chunks : {}", state.chunk_id);
        eprintln!("updates: {updates}");
    }
    Ok(final_text)
}

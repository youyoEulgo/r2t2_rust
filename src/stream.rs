// Copyright 2026 youyoEulgo
// SPDX-License-Identifier: Apache-2.0
//
// This file ports the streaming algorithm from `r2t2/r2t2_asr.py`,
// and `parse_asr_output` from Qwen3-ASR's `qwen_asr.inference.utils`
//
// Modifications are described in NOTICE; they are not endorsed by
// the original authors.

//! Streaming (chunked) transcription: the Longest-Stable-Prefix loop.
//!
//! This is a port of `R2T2ASRModel.streaming_transcribe()` from the Python
//! project's `r2t2/r2t2_asr.py`. The algorithm is:
//!
//! 1. Buffer incoming PCM; whenever a full chunk is available, consume it.
//! 2. Append the chunk to everything heard so far and **re-feed the whole
//!    audio** to the model. Nothing is trimmed away, which is what makes the
//!    already-emitted text stay valid.
//! 3. Prompt with the previously decoded text as a prefix -- but roll back the
//!    last `unfixed_token_num` tokens so the model can still revise the
//!    boundary.
//! 4. Decode, then emit `fixed_text`: the decode truncated by the same rollback
//!    window, i.e. only the part considered stable.
//!
//! Two details are load-bearing and easy to get wrong:
//!
//! * **Tokenizer**: the reference uses the HuggingFace tokenizer, not
//!   llama.cpp's. Token boundaries differ between the two, and the rollback
//!   count is in tokens, so boundaries change behaviour. This port uses
//!   llama.cpp's tokenizer (the whole point of the Rust rewrite is to drop the
//!   Python dependency); transcripts match on the sample audio, but the
//!   rollback granularity is not bit-identical to the reference.
//! * **UTF-8 safety**: decoding `cur_ids[..len-k]` can split a multi-byte
//!   character and produce U+FFFD. The reference loops, increasing `k` until
//!   the result is clean. That loop is reproduced faithfully here.
//!
//! Known behaviour: `fixed_text` is *not* strictly append-only. On the sample
//! audio the reference emits one regression in 32 non-empty updates, and this
//! port inherits that characteristic. The instability comes from the model
//! revising a position before the rollback window, which the window cannot
//! cover -- it is inherent to the approach, not a porting defect.

use anyhow::Result;

use crate::engine::{Engine, EngineConfig};

/// Per-stream state. Mirrors `ASRStreamingState`.
#[derive(Debug, Clone)]
pub struct StreamState {
    /// Prefix is reset to "" for the first `unfixed_chunk_num` chunks.
    pub unfixed_chunk_num: usize,
    /// How many trailing tokens to roll back when building the prefix.
    pub unfixed_token_num: usize,
    pub chunk_size_sec: f32,
    pub chunk_size_samples: usize,

    pub chunk_id: usize,
    /// PCM not yet consumed into a full chunk.
    buffer: Vec<f32>,
    /// Everything consumed so far; re-fed in full on every step.
    audio_accum: Vec<f32>,

    /// Chat-templated prompt with no prefix appended.
    prompt_raw: String,
    language: Option<String>,

    /// Latest parsed language.
    pub language_out: String,
    /// Latest full transcript (includes the not-yet-stable tail).
    pub text: String,
    /// Raw decoded text, before `<asr_text>` parsing. Used for rollback.
    raw_decoded: String,
    /// Latest stable (rolled-back) text -- what a UI should display.
    pub fixed_text: String,

    /// Per-chunk increments, used by the `no_reset` variant as its prefix.
    ///
    /// Parallel to the audio in `audio_accum`: when old audio is discarded
    /// from the front, the matching leading entries are dropped too, so the
    /// two stay aligned. See [`StreamEngine::push_no_reset`].
    chunk_text: Vec<String>,
    /// Text accumulated across the whole session by `no_reset`.
    pub last_fixed_text: String,
    /// Whether the first (double-length) audio discard has happened.
    first_chunk_discarded: bool,
}

/// Streaming wrapper around [`Engine`].
pub struct StreamEngine {
    engine: Engine,
    max_new_tokens: i32,
}

impl StreamEngine {
    pub fn load(cfg: &EngineConfig, max_new_tokens: i32) -> Result<Self> {
        Ok(Self {
            engine: Engine::load(cfg)?,
            max_new_tokens,
        })
    }

    /// Transcribe a whole clip in one pass.
    ///
    /// Streaming and one-shot are two ways of driving the same model, and the
    /// process must hold exactly one copy of it: the weights alone are several
    /// gigabytes, and a second copy does not fit on a consumer GPU. So the
    /// one-shot paths reach the model through here rather than loading their
    /// own.
    pub fn transcribe(
        &self,
        audio: &[f32],
        prompt: &str,
        max_tokens: i32,
    ) -> Result<crate::engine::TranscribeResult> {
        self.engine.transcribe(audio, prompt, max_tokens)
    }

    /// Create state for one stream. Mirrors `init_streaming_state()`.
    pub fn init_state(
        &self,
        context: &str,
        language: Option<&str>,
        unfixed_chunk_num: usize,
        unfixed_token_num: usize,
        chunk_size_sec: f32,
    ) -> StreamState {
        let chunk_size_samples = ((chunk_size_sec * crate::audio::TARGET_SAMPLE_RATE as f32)
            .round() as usize)
            .max(1);
        StreamState {
            unfixed_chunk_num,
            unfixed_token_num,
            chunk_size_sec,
            chunk_size_samples,
            chunk_id: 0,
            buffer: Vec::new(),
            audio_accum: Vec::new(),
            prompt_raw: crate::prompt::build(context, language, ""),
            language: language.map(|s| s.to_string()),
            language_out: String::new(),
            text: String::new(),
            raw_decoded: String::new(),
            fixed_text: String::new(),
            chunk_text: Vec::new(),
            last_fixed_text: String::new(),
            first_chunk_discarded: false,
        }
    }

    /// Feed audio. Returns `Some((text, fixed_text))` if at least one chunk was
    /// decoded, otherwise `None` (the audio is still buffering).
    ///
    /// An arbitrary amount of audio may be passed; every complete chunk is
    /// consumed. Mirrors `streaming_transcribe()`.
    pub fn push(&self, pcm: &[f32], state: &mut StreamState) -> Result<Option<(String, String)>> {
        if pcm.is_empty() && state.buffer.len() < state.chunk_size_samples {
            return Ok(None);
        }
        state.buffer.extend_from_slice(pcm);

        let mut decoded_any = false;
        while state.buffer.len() >= state.chunk_size_samples {
            let chunk: Vec<f32> = state.buffer.drain(..state.chunk_size_samples).collect();
            state.audio_accum.extend_from_slice(&chunk);

            // --- build the prefix, rolling back the unstable tail ---------
            let prefix = if state.chunk_id < state.unfixed_chunk_num {
                String::new()
            } else {
                // `|` acts as a hard terminator in this model's output.
                state.raw_decoded = state
                    .raw_decoded
                    .split('|')
                    .next()
                    .unwrap_or("")
                    .to_string();

                let cur_ids = self.engine.tokenize(&state.raw_decoded, false, true)?;
                let mut k = state.unfixed_token_num;
                // Reassigned on every iteration of the loop below.
                #[allow(unused_assignments)]
                let mut prefix = String::new();
                loop {
                    let end_idx = cur_ids.len().saturating_sub(k);
                    prefix = if end_idx > 0 {
                        self.engine.detokenize(&cur_ids[..end_idx])?
                    } else {
                        String::new()
                    };
                    // A replacement char means the cut split a multi-byte
                    // character; widen the rollback and try again.
                    if !prefix.contains('\u{fffd}') {
                        break;
                    }
                    if end_idx == 0 {
                        prefix = String::new();
                        break;
                    }
                    k += 1;
                }
                prefix
            };
            let prefix = prefix.split('|').next().unwrap_or("").to_string();

            // --- decode ---------------------------------------------------
            let prompt = format!("{}{}", state.prompt_raw, prefix);
            let gen_text = self.engine.generate(&state.audio_accum, &prompt, self.max_new_tokens)?;
            let gen_text = normalize_punct_by_context(&gen_text);
            let gen_text = gen_text.replace('\u{fffd}', "");

            state.raw_decoded = format!("{prefix}{gen_text}");
            state.raw_decoded = state.raw_decoded.split('|').next().unwrap_or("").to_string();

            let (lang, txt) = parse_asr_output(&state.raw_decoded, state.language.as_deref());

            // Chinese output sometimes carries spaces between characters, but
            // only after `parse_asr_output` has decided what the text is; the
            // collapse therefore runs on the reconstructed raw string below.
            if state.raw_decoded.contains("<asr_text>") {
                let head = state
                    .raw_decoded
                    .split("<asr_text>")
                    .next()
                    .unwrap_or("")
                    .to_string();
                state.raw_decoded = format!("{head}<asr_text>{txt}");
            } else {
                state.raw_decoded = txt.clone();
            }
            state.raw_decoded = state.raw_decoded.split('|').next().unwrap_or("").to_string();

            if state.language.as_deref() == Some("Chinese") || lang == "Chinese" {
                state.raw_decoded = collapse_cjk_spaces(&state.raw_decoded);
            }

            // --- compute fixed_text (same rollback, for display) ----------
            let cur_ids = self.engine.tokenize(&state.raw_decoded, false, true)?;
            let mut k = if state.raw_decoded.contains("<asr_text>")
                && state.raw_decoded.split("<asr_text>").nth(1) == Some("")
            {
                0
            } else {
                state.unfixed_token_num
            };

            // Reassigned on every iteration of the loop below.
            #[allow(unused_assignments)]
            let mut fixed_text = String::new();
            loop {
                let end_idx = cur_ids.len().saturating_sub(k);
                fixed_text = if end_idx > 0 {
                    self.engine.detokenize(&cur_ids[..end_idx])?
                } else {
                    String::new()
                };
                if !fixed_text.contains('\u{fffd}') {
                    break;
                }
                if end_idx == 0 {
                    fixed_text = String::new();
                    break;
                }
                k += 1;
            }
            let fixed_text = if fixed_text.contains("<asr_text>") {
                fixed_text
                    .split("<asr_text>")
                    .nth(1)
                    .unwrap_or("")
                    .to_string()
            } else {
                fixed_text
            };
            let fixed_text = fixed_text.split('|').next().unwrap_or("").to_string();

            // Without an <asr_text> tag and without a forced language, the
            // model has not committed to anything yet.
            if !state.raw_decoded.contains("<asr_text>") && state.language.is_none() {
                state.text.clear();
                state.chunk_id += 1;
                decoded_any = true;
                continue;
            }

            state.language_out = lang;
            state.text = txt.split('|').next().unwrap_or("").to_string();
            state.fixed_text = fixed_text;
            state.chunk_id += 1;
            decoded_any = true;
        }

        if decoded_any {
            Ok(Some((state.text.clone(), state.fixed_text.clone())))
        } else {
            Ok(None)
        }
    }

    /// Flush any remaining buffered audio and decode one final time.
    /// Mirrors `finish_streaming_transcribe()`.
    pub fn finish(&self, state: &mut StreamState) -> Result<String> {
        if state.buffer.is_empty() {
            return Ok(state.text.clone());
        }

        let tail: Vec<f32> = std::mem::take(&mut state.buffer);
        state.audio_accum.extend_from_slice(&tail);

        let prefix = if state.chunk_id < state.unfixed_chunk_num {
            String::new()
        } else {
            let cur_ids = self.engine.tokenize(&state.raw_decoded, false, true)?;
            let end_idx = cur_ids.len().saturating_sub(state.unfixed_token_num).max(1);
            let p = if end_idx > 0 {
                self.engine.detokenize(&cur_ids[..end_idx])?
            } else {
                String::new()
            };
            p.split('|').next().unwrap_or("").to_string()
        };

        let prompt = format!("{}{}", state.prompt_raw, prefix);
        let gen_text = self.engine.generate(&state.audio_accum, &prompt, self.max_new_tokens)?;
        let gen_text = normalize_punct_by_context(&gen_text).replace('\u{fffd}', "");

        state.raw_decoded = format!("{prefix}{gen_text}");
        state.raw_decoded = state.raw_decoded.split('|').next().unwrap_or("").to_string();

        let (lang, txt) = parse_asr_output(&state.raw_decoded, state.language.as_deref());
        state.language_out = lang;
        state.text = txt.split('|').next().unwrap_or("").to_string();
        state.fixed_text = state.text.clone();
        state.chunk_id += 1;
        Ok(state.text.clone())
    }

    // ----------------------------------------------------------------------- //
    // no_reset variant -- the one the WebSocket service uses
    // ----------------------------------------------------------------------- //

    /// Feed audio using the rolling-window ("no reset") strategy.
    ///
    /// Port of `streaming_transcribe_no_reset()`. It differs from [`Self::push`]
    /// in three ways that matter for a long-lived connection:
    ///
    /// 1. **The audio window is capped.** Once `audio_accum` exceeds 16 s the
    ///    oldest 8 s are dropped, so memory and prompt length stay bounded on a
    ///    stream that runs for hours. The matching leading `chunk_text` entries
    ///    are dropped with it.
    /// 2. **The prefix comes from `chunk_text`**, not from re-decoding the whole
    ///    transcript. Each step appends only its own increment, so the prefix
    ///    stays consistent with the audio that is actually still in the window.
    /// 3. **Text accumulates into `last_fixed_text`** instead of being
    ///    recomputed, which is what lets the caller emit increments.
    ///
    /// Returns `(text, last_fixed_text)` when at least one chunk was decoded.
    pub fn push_no_reset(
        &self,
        pcm: &[f32],
        state: &mut StreamState,
    ) -> Result<Option<(String, String)>> {
        const SAMPLE_RATE: usize = crate::audio::TARGET_SAMPLE_RATE as usize;
        const MAX_SAMPLES: usize = 16 * SAMPLE_RATE;
        const DISCARD_SAMPLES: usize = 8 * SAMPLE_RATE;
        const NORMAL_CHUNK: usize = 2560; // 160 ms
        const FIRST_CHUNK: usize = 5120; // 320 ms (first chunk is double length)

        state.buffer.extend_from_slice(pcm);
        let mut decoded_any = false;

        while state.buffer.len() >= state.chunk_size_samples {
            let chunk: Vec<f32> = state.buffer.drain(..state.chunk_size_samples).collect();
            state.audio_accum.extend_from_slice(&chunk);

            // --- trim the rolling window ----------------------------------
            if state.audio_accum.len() > MAX_SAMPLES {
                // How many chunk_text entries correspond to the dropped audio.
                // The first chunk is twice as long as the rest, so the very
                // first discard drops one extra entry.
                let discard_chunks = if !state.first_chunk_discarded {
                    state.first_chunk_discarded = true;
                    1 + DISCARD_SAMPLES.saturating_sub(FIRST_CHUNK) / NORMAL_CHUNK
                } else {
                    DISCARD_SAMPLES / NORMAL_CHUNK
                };
                state.audio_accum.drain(..DISCARD_SAMPLES);
                let n = discard_chunks.min(state.chunk_text.len());
                state.chunk_text.drain(..n);
            }

            // --- prefix from accumulated increments -----------------------
            let prefix_text: String = state.chunk_text.concat();
            let prefix = if state.language.is_none() && !state.language_out.is_empty() {
                format!("language {}<asr_text>{}", state.language_out, prefix_text)
            } else {
                prefix_text.clone()
            };
            let prefix = prefix.split('|').next().unwrap_or("").to_string();

            // --- decode ---------------------------------------------------
            let prompt = format!("{}{}", state.prompt_raw, prefix);
            let gen_text = self.engine.generate(&state.audio_accum, &prompt, self.max_new_tokens)?;
            let gen_text = normalize_punct_by_context(&gen_text).replace('\u{fffd}', "");

            state.raw_decoded = format!("{prefix}{gen_text}");
            state.raw_decoded = state.raw_decoded.split('|').next().unwrap_or("").to_string();

            let (lang, txt) = parse_asr_output(&state.raw_decoded, state.language.as_deref());

            if state.language.as_deref() == Some("Chinese") || lang == "Chinese" {
                state.raw_decoded = collapse_cjk_spaces(&state.raw_decoded);
            }

            if state.raw_decoded.contains("<asr_text>") {
                let head = state
                    .raw_decoded
                    .split("<asr_text>")
                    .next()
                    .unwrap_or("")
                    .to_string();
                state.raw_decoded = format!("{head}<asr_text>{txt}");
            } else {
                state.raw_decoded = txt.clone();
            }
            state.raw_decoded = state.raw_decoded.split('|').next().unwrap_or("").to_string();

            // --- roll back for fixed_text ---------------------------------
            let cur_ids = self.engine.tokenize(&state.raw_decoded, false, true)?;
            let mut k = if state.raw_decoded.contains("<asr_text>")
                && state.raw_decoded.split("<asr_text>").nth(1) == Some("")
            {
                0
            } else {
                state.unfixed_token_num
            };

            // Reassigned on every iteration of the loop below.
            #[allow(unused_assignments)]
            let mut fixed_text = String::new();
            loop {
                let end_idx = cur_ids.len().saturating_sub(k);
                fixed_text = if end_idx > 0 {
                    self.engine.detokenize(&cur_ids[..end_idx])?
                } else {
                    String::new()
                };
                if !fixed_text.contains('\u{fffd}') {
                    break;
                }
                if end_idx == 0 {
                    fixed_text = String::new();
                    break;
                }
                k += 1;
            }
            let fixed_text = if fixed_text.contains("<asr_text>") {
                fixed_text
                    .split("<asr_text>")
                    .nth(1)
                    .unwrap_or("")
                    .to_string()
            } else {
                fixed_text
            };

            if !state.raw_decoded.contains("<asr_text>") && state.language.is_none() {
                state.text.clear();
                state.chunk_id += 1;
                decoded_any = true;
                continue;
            }

            state.language_out = lang;
            state.text = txt.split('|').next().unwrap_or("").to_string();
            state.chunk_id += 1;

            // The increment is whatever `fixed_text` adds on top of the prefix
            // already accumulated. If the model rewrote the prefix instead of
            // extending it, there is no safe increment to emit.
            let prefix_stripped = prefix_text.trim();
            let fixed_stripped = fixed_text.trim();
            let new_asr_text = fixed_stripped
                .strip_prefix(prefix_stripped)
                .unwrap_or("")
                .split('|')
                .next()
                .unwrap_or("")
                .to_string();

            state.chunk_text.push(new_asr_text.clone());
            state.last_fixed_text.push_str(&new_asr_text);
            decoded_any = true;
        }

        if decoded_any {
            Ok(Some((state.text.clone(), state.last_fixed_text.clone())))
        } else {
            Ok(None)
        }
    }

    /// Flush the tail for the `no_reset` path. Mirrors
    /// `finish_streaming_transcribe_no_reset()`.
    pub fn finish_no_reset(&self, state: &mut StreamState) -> Result<String> {
        if state.buffer.is_empty() {
            return Ok(state.last_fixed_text.clone());
        }

        let tail: Vec<f32> = std::mem::take(&mut state.buffer);
        state.audio_accum.extend_from_slice(&tail);

        let prefix_text: String = state.chunk_text.concat();
        let prefix = if state.language.is_none() && !state.language_out.is_empty() {
            format!("language {}<asr_text>{}", state.language_out, prefix_text)
        } else {
            prefix_text.clone()
        };
        let prefix = prefix.split('|').next().unwrap_or("").to_string();

        let prompt = format!("{}{}", state.prompt_raw, prefix);
        let gen_text = self.engine.generate(&state.audio_accum, &prompt, self.max_new_tokens)?;
        let gen_text = normalize_punct_by_context(&gen_text).replace('\u{fffd}', "");

        state.raw_decoded = format!("{prefix}{gen_text}");
        state.raw_decoded = state.raw_decoded.split('|').next().unwrap_or("").to_string();

        let (lang, txt) = parse_asr_output(&state.raw_decoded, state.language.as_deref());
        state.language_out = lang;
        state.text = txt.split('|').next().unwrap_or("").to_string();

        let fixed = state.raw_decoded.clone();
        let prefix_stripped = prefix_text.trim();
        let fixed_stripped = if fixed.contains("<asr_text>") {
            fixed.split("<asr_text>").nth(1).unwrap_or("").trim()
        } else {
            fixed.trim()
        };
        let new_asr_text = fixed_stripped
            .strip_prefix(prefix_stripped)
            .unwrap_or("")
            .to_string();
        state.last_fixed_text.push_str(&new_asr_text);
        state.chunk_id += 1;
        Ok(state.last_fixed_text.clone())
    }
}

// --------------------------------------------------------------------------- //
// output parsing
// --------------------------------------------------------------------------- //

/// Split a raw decode into `(language, text)`.
///
/// Port of `parse_asr_output()` from `qwen_asr.inference.utils`. The cases are:
///
/// * **Language forced** (`user_language` set): the model is expected to emit
///   plain text with no metadata, so the whole string is the transcript and the
///   language is echoed back unchanged.
/// * **`<asr_text>` present**: everything before the tag is metadata, everything
///   after is the transcript. The language is read from a `language X` line.
/// * **No tag**: the whole string is the transcript, language unknown.
///
/// Note the asymmetry that is easy to get wrong: when a language is forced the
/// text is taken **verbatim, tag included**, whereas the untagged path strips
/// whitespace. Both are reproduced here.
pub fn parse_asr_output(raw: &str, user_language: Option<&str>) -> (String, String) {
    let s = raw.trim();
    if s.is_empty() {
        return (String::new(), String::new());
    }

    // Forced language: treat the entire output as text.
    if let Some(forced) = user_language {
        return (forced.to_string(), s.to_string());
    }

    let Some(idx) = s.find("<asr_text>") else {
        // No tag => pure text, language unknown.
        return (String::new(), s.to_string());
    };

    let meta_part = &s[..idx];
    let text_part = s[idx + "<asr_text>".len()..].trim();

    // Empty-audio heuristic: the model reports `language None`.
    if meta_part.to_lowercase().contains("language none") {
        return (String::new(), text_part.to_string());
    }

    // Language is the value of a `language X` line in the metadata.
    let mut lang = String::new();
    for line in meta_part.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.to_lowercase().starts_with("language") {
            let val = line["language".len()..].trim();
            if !val.is_empty() {
                lang = val.to_string();
            }
            break;
        }
    }

    (lang, text_part.to_string())
}

/// Remove spaces that sit between two CJK characters.
fn collapse_cjk_spaces(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == ' ' {
            let prev_cjk = out.chars().last().is_some_and(is_cjk);
            let next_cjk = chars.get(i + 1).copied().is_some_and(is_cjk);
            if prev_cjk && next_cjk {
                i += 1; // drop the space
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

fn is_cjk(c: char) -> bool {
    matches!(c, '\u{4e00}'..='\u{9fff}')
}

/// Normalise punctuation to match the preceding character's script.
///
/// Port of `_normalize_punct_by_context`. The rule is deliberately asymmetric:
/// only the **preceding non-space character** is consulted.
///
/// * CJK before the mark -> Chinese punctuation (`你好,` becomes `你好，`)
/// * ASCII alphanumeric or a quote before the mark -> English punctuation
/// * anything else (including nothing at all) -> the mark is left alone
///
/// Checking the *following* character instead -- a natural-looking variation --
/// changes results on real output: a transcript ending in Chinese followed by
/// `.` would be left as ASCII `.` rather than becoming `。`, because there is
/// no following character to inspect.
fn normalize_punct_by_context(text: &str) -> String {
    const EN2ZH: [(char, char); 8] = [
        (',', '，'),
        ('.', '。'),
        ('!', '！'),
        ('?', '？'),
        (';', '；'),
        (':', '：'),
        ('(', '（'),
        (')', '）'),
    ];
    let zh2en: Vec<(char, char)> = EN2ZH.iter().map(|(e, z)| (*z, *e)).collect();
    let is_punct = |c: char| EN2ZH.iter().any(|(e, _)| *e == c) || zh2en.iter().any(|(z, _)| *z == c);

    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());

    for (i, &c) in chars.iter().enumerate() {
        if !is_punct(c) {
            out.push(c);
            continue;
        }
        // Nearest preceding character that is not whitespace.
        let prev = chars[..i].iter().rev().find(|ch| !ch.is_whitespace()).copied();

        let Some(prev) = prev else {
            out.push(c);
            continue;
        };

        if is_cjk(prev) {
            out.push(EN2ZH.iter().find(|(e, _)| *e == c).map_or(c, |(_, z)| *z));
        } else if prev.is_ascii_alphanumeric() || prev == '"' || prev == '\'' {
            out.push(zh2en.iter().find(|(z, _)| *z == c).map_or(c, |(_, e)| *e));
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_forced_language_output() {
        // With a forced language the reference treats the whole string as text,
        // tag and all -- the caller reconstructs the raw form from it.
        let (lang, text) = parse_asr_output("language Chinese<asr_text>你好", Some("Chinese"));
        assert_eq!(lang, "Chinese");
        assert_eq!(text, "language Chinese<asr_text>你好");
    }

    #[test]
    fn parses_reported_language() {
        let (lang, text) = parse_asr_output("language English<asr_text>hello", None);
        assert_eq!(lang, "English");
        assert_eq!(text, "hello");
    }

    #[test]
    fn no_tag_yields_whole_string_as_text() {
        let (lang, text) = parse_asr_output("language Chinese", None);
        assert_eq!(lang, "");
        assert_eq!(text, "language Chinese");
    }

    #[test]
    fn language_none_means_silence() {
        let (lang, text) = parse_asr_output("language None<asr_text>", None);
        assert_eq!(lang, "");
        assert_eq!(text, "");
    }

    #[test]
    fn reads_language_from_its_own_line() {
        let (lang, text) = parse_asr_output("language Chinese\n<asr_text>你好", None);
        assert_eq!(lang, "Chinese");
        assert_eq!(text, "你好");
    }

    #[test]
    fn collapses_spaces_between_cjk() {
        assert_eq!(collapse_cjk_spaces("你 好 世 界"), "你好世界");
        // Spaces around Latin text must survive.
        assert_eq!(collapse_cjk_spaces("你 good 好"), "你 good 好");
    }

    #[test]
    fn punctuation_follows_preceding_script() {
        assert_eq!(normalize_punct_by_context("你好,世界"), "你好，世界");
        assert_eq!(normalize_punct_by_context("hello, world"), "hello, world");
        // A trailing mark has no following character; only the preceding one
        // counts, so ASCII '.' after CJK still becomes '。'.
        assert_eq!(normalize_punct_by_context("你好."), "你好。");
        assert_eq!(normalize_punct_by_context("ok."), "ok.");
        // Chinese punctuation next to ASCII text is converted the other way.
        assert_eq!(normalize_punct_by_context("ok。"), "ok.");
        // Whitespace between the text and the mark is skipped over.
        assert_eq!(normalize_punct_by_context("你好 ,"), "你好 ，");
    }
}

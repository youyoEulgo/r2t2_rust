//! Subtitle generation: media in, SRT out.
//!
//! The pipeline is deliberately **offline** rather than streaming, because a
//! finished file allows a strategy a live stream cannot use:
//!
//! 1. **VAD segments the whole audio first.** The detector reports where speech
//!    starts and stops, which gives every cue a precise timestamp. This is the
//!    key move: the ASR model is not asked for timings (it cannot provide them
//!    -- the streaming path explicitly drops the forced aligner), so *the
//!    detector supplies the timing and the model supplies only the words*.
//! 2. **Each segment is transcribed independently**, one-shot. No rolling
//!    window, no incremental state, no cross-segment contamination.
//! 3. **Long segments are split again** on punctuation and length, so a cue
//!    stays readable instead of running for twenty seconds.
//!
//! Timestamps are computed from frame indices, so they are exact rather than
//! inferred from the text.

use std::path::Path;

use anyhow::{Context, Result};

use crate::engine::{Engine, EngineConfig};
use crate::prompt;
use crate::vad::{Segment, Segmenter, VadConfig, FRAME_MS, FRAME_SAMPLES};

/// One subtitle cue.
#[derive(Debug, Clone, PartialEq)]
pub struct Cue {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// How to split a long segment into readable cues.
#[derive(Debug, Clone)]
pub struct SplitConfig {
    /// Hard limit: a cue longer than this is split.
    pub max_chars: usize,
    /// A cue shorter than this is merged with its neighbour when possible.
    pub min_chars: usize,
    /// Upper bound on a cue's duration, in seconds.
    pub max_seconds: f64,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            // Roughly two lines of Chinese at typical subtitle widths.
            // Note this counts characters, so a Latin-heavy line of the same
            // length will read wider.
            max_chars: 24,
            min_chars: 4,
            max_seconds: 8.0,
        }
    }
}

/// Tuning for the output quality guards.
///
/// Defaults mirror the reference Python server so behaviour is comparable.
#[derive(Debug, Clone)]
pub struct QualityConfig {
    /// Repeats needed before a run or loop counts as "stuck".
    pub repeat_threshold: usize,
    /// Longest repeated phrase considered, in characters.
    pub max_pattern_len: usize,
    /// How much of the tail to examine for a loop.
    pub tail_check_len: usize,
    /// Emit segments flagged as hallucinated instead of dropping them.
    ///
    /// Useful for auditing the detector: run once with this set and compare
    /// against a normal run to see exactly what is being discarded.
    pub keep_hallucinations: bool,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            repeat_threshold: 5,
            max_pattern_len: 50,
            tail_check_len: 256,
            keep_hallucinations: false,
        }
    }
}

/// Result of processing one file.
#[derive(Debug)]
pub struct SubtitleResult {
    pub cues: Vec<Cue>,
    /// Segments the detector found.
    pub segments: usize,
    /// Segments dropped: empty, degenerate, or flagged as hallucinated.
    pub empty_segments: usize,
}

/// Build subtitles for a decoded waveform.
///
/// `samples` is mono 16 kHz. `engine` is borrowed so the caller controls its
/// lifetime across multiple files.
pub fn build(
    engine: &Engine,
    samples: &[f32],
    language: Option<&str>,
    context: &str,
    vad: VadConfig,
    split: &SplitConfig,
    quality: &QualityConfig,
    max_tokens: i32,
) -> Result<SubtitleResult> {
    let segments = detect_segments(samples, vad)?;
    let mut cues = Vec::new();
    let mut empty = 0usize;

    for seg in &segments {
        let text = transcribe_segment(engine, samples, seg, language, context, max_tokens)?;
        // Collapse decoder loops before anything else looks at the text; a
        // stuck phrase otherwise becomes a string of identical cues.
        let text = crate::quality::fix_repetitions(text.trim(), quality.repeat_threshold);

        if crate::quality::is_degenerate(&text) {
            empty += 1;
            continue;
        }
        // A hallucination running to the end of a segment means the model
        // invented this span rather than transcribing it. Dropping it is
        // correct: a wrong cue is worse than a missing one.
        if !quality.keep_hallucinations
            && crate::quality::detect_hallucination(
                &text,
                quality.repeat_threshold,
                quality.max_pattern_len,
                quality.tail_check_len,
            )
            .is_some()
        {
            empty += 1;
            continue;
        }

        let (start, end) = segment_times(seg);
        cues.extend(split_cue(start, end, &text, split));
    }

    // Merge cues that are too short to read on their own.
    let cues = merge_short(cues, split);

    Ok(SubtitleResult {
        cues,
        segments: segments.len(),
        empty_segments: empty,
    })
}

/// Run the detector over the whole waveform and collect closed segments.
pub fn detect_segments(samples: &[f32], config: VadConfig) -> Result<Vec<Segment>> {
    let mut segmenter = Segmenter::new(config)?;
    let mut out = Vec::new();

    for frame in samples.chunks_exact(FRAME_SAMPLES) {
        let frame_i16: Vec<i16> = frame
            .iter()
            .map(|s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect();
        if let Some(seg) = segmenter.push_frame(&frame_i16)? {
            out.push(seg);
        }
    }
    if let Some(seg) = segmenter.flush() {
        out.push(seg);
    }
    Ok(out)
}

/// Convert a segment's frame span to `(start, end)` seconds.
pub fn segment_times(seg: &Segment) -> (f64, f64) {
    let fps = 1000.0 / FRAME_MS as f64;
    let start = seg.start_frame as f64 / fps;
    let end = (seg.start_frame + seg.frames) as f64 / fps;
    (start, end)
}

/// Transcribe one detected segment.
fn transcribe_segment(
    engine: &Engine,
    samples: &[f32],
    seg: &Segment,
    language: Option<&str>,
    context: &str,
    max_tokens: i32,
) -> Result<String> {
    let (start_sample, end_sample) = segment_sample_range(seg, samples.len());
    if end_sample <= start_sample {
        return Ok(String::new());
    }
    let audio = &samples[start_sample..end_sample];
    let p = prompt::build(context, language, "");
    let result = engine
        .transcribe(audio, &p, max_tokens)
        .context("transcription of a segment failed")?;

    // The one-shot path returns `language X<asr_text>...`; keep only the text.
    let (_, text) = crate::stream::parse_asr_output(&result.text, language);
    Ok(text.trim().to_string())
}

fn segment_sample_range(seg: &Segment, total: usize) -> (usize, usize) {
    let start = (seg.start_frame as usize) * FRAME_SAMPLES;
    let end = start + (seg.frames as usize) * FRAME_SAMPLES;
    (start.min(total), end.min(total))
}

/// Break one segment's text into readable cues, interpolating timings by
/// character count.
///
/// Exact per-word timings are not available, so a cue's share of the segment's
/// duration is proportional to its share of the characters. For subtitles this
/// is accurate enough, and it keeps cues from overlapping.
pub fn split_cue(start: f64, end: f64, text: &str, cfg: &SplitConfig) -> Vec<Cue> {
    let pieces = split_text(text, cfg);
    if pieces.is_empty() {
        return Vec::new();
    }
    if pieces.len() == 1 {
        return vec![Cue {
            start,
            end,
            text: pieces[0].clone(),
        }];
    }

    let total_chars: usize = pieces.iter().map(|p| p.chars().count()).sum();
    let duration = (end - start).max(0.0);
    let mut cues = Vec::with_capacity(pieces.len());
    let mut cursor = start;

    for (i, piece) in pieces.iter().enumerate() {
        let chars = piece.chars().count();
        // Last piece absorbs any rounding drift so cues tile exactly.
        let share = if i + 1 == pieces.len() {
            (start + duration) - cursor
        } else if total_chars == 0 {
            duration / pieces.len() as f64
        } else {
            duration * chars as f64 / total_chars as f64
        };
        let piece_end = (cursor + share).min(end);
        cues.push(Cue {
            start: cursor,
            end: piece_end,
            text: piece.clone(),
        });
        cursor = piece_end;
    }
    cues
}

/// Split text into cue-sized pieces, preferring punctuation boundaries.
///
/// The critical constraint is that a break must never land inside a Latin word:
/// splitting `Linux` into `L` / `inux` is a visible defect that no amount of
/// correct timing can excuse. Breaks are therefore only allowed at a position
/// where the previous character is not part of an unbroken ASCII word run, and
/// punctuation is preferred when one is available nearby.
fn split_text(text: &str, cfg: &SplitConfig) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }

    let chars: Vec<char> = text.chars().collect();
    let mut pieces: Vec<String> = Vec::new();
    let mut start = 0usize;

    while start < chars.len() {
        let remaining = chars.len() - start;
        if remaining <= cfg.max_chars {
            pieces.push(chars[start..].iter().collect());
            break;
        }

        // Candidate cut offsets within the window, best first.
        let limit = start + cfg.max_chars;
        let cut = best_break(&chars, start, limit, cfg);

        pieces.push(chars[start..cut].iter().collect());
        start = cut;
    }

    // A trailing scrap that is too small to read is rebalanced with its
    // neighbour; see the note in `rebalance_tail`.
    rebalance_tail(&mut pieces, cfg);
    pieces
}

/// Choose where to cut, given a window `[start, limit)`.
///
/// Preference order:
/// 1. Just after sentence-final punctuation.
/// 2. Just after a clause separator.
/// 3. At a space, which is by definition a word boundary.
/// 4. At the window limit, but pushed forward to the end of the current ASCII
///    word so a word is never halved.
fn best_break(chars: &[char], start: usize, limit: usize, cfg: &SplitConfig) -> usize {
    // Weights: sentence enders beat clause separators.
    let mut best: Option<(usize, usize)> = None; // (position, score)
    for i in start..limit {
        let c = chars[i];
        let pos = i + 1;
        if pos - start < cfg.min_chars {
            continue;
        }
        let score = match c {
            '。' | '！' | '？' | '!' | '?' => 3,
            '；' | ';' | '，' | ',' | '、' | '：' | ':' => 2,
            _ => continue,
        };
        // Later breaks are better when scores tie, to fill each cue.
        if best.is_none_or(|(_, s)| score >= s) {
            best = Some((pos, score));
        }
    }
    if let Some((pos, _)) = best {
        return pos;
    }

    // No punctuation: prefer the last space in the window.
    if let Some(i) = (start..limit).rev().find(|&i| chars[i].is_whitespace()) {
        if i + 1 - start >= cfg.min_chars {
            return i + 1;
        }
    }

    // Otherwise cut at the limit -- unless that lands inside an ASCII word, in
    // which case pull back to the start of that word rather than overshooting.
    //
    // Extending forward instead would be simpler but unbounded: one long token
    // (a URL, a German compound) would push the cue arbitrarily past the limit.
    // Backing up costs a slightly short cue and keeps the bound meaningful.
    let mut cut = limit;
    let is_word = |c: char| c.is_ascii_alphanumeric();
    if cut > start
        && cut < chars.len()
        && is_word(chars[cut - 1])
        && is_word(chars[cut])
    {
        let mut back = cut;
        while back > start && is_word(chars[back - 1]) {
            back -= 1;
        }
        // Only accept the backed-up cut if it still leaves a readable cue;
        // otherwise the word is longer than the whole limit and splitting it
        // is the lesser evil.
        if back > start && back - start >= cfg.min_chars {
            cut = back;
        }
    }
    cut
}

/// Rebalance a trailing piece that is too short to stand alone.
///
/// Folding it backwards is not always possible: the previous piece may already
/// sit at the length limit, and overshooting is worse than a short final cue.
/// When folding does not fit, characters are instead pulled back from the
/// previous piece so both end up within bounds.
fn rebalance_tail(pieces: &mut Vec<String>, cfg: &SplitConfig) {
    if pieces.len() < 2 {
        return;
    }
    let last_len = pieces.last().unwrap().chars().count();
    if last_len >= cfg.min_chars {
        return;
    }

    let prev_idx = pieces.len() - 2;
    let prev_len = pieces[prev_idx].chars().count();
    if prev_len + last_len <= cfg.max_chars {
        let tail = pieces.pop().unwrap();
        pieces[prev_idx].push_str(&tail);
        return;
    }

    // Move enough characters back to clear the minimum, keeping the previous
    // piece at or above `min_chars` too.
    let need = cfg.min_chars - last_len;
    let movable = prev_len.saturating_sub(cfg.min_chars);
    let take = need.min(movable);
    if take == 0 {
        return;
    }
    let prev: Vec<char> = pieces[prev_idx].chars().collect();
    let split = prev.len() - take;
    let moved: String = prev[split..].iter().collect();
    pieces[prev_idx] = prev[..split].iter().collect();
    let tail = pieces.pop().unwrap();
    pieces.push(format!("{moved}{tail}"));
}

/// Merge cues that are too brief to read, subject to the duration limit.
fn merge_short(cues: Vec<Cue>, cfg: &SplitConfig) -> Vec<Cue> {
    let mut out: Vec<Cue> = Vec::with_capacity(cues.len());
    for cue in cues {
        match out.last_mut() {
            Some(prev)
                if prev.text.chars().count() < cfg.min_chars
                    && cue.end - prev.start <= cfg.max_seconds =>
            {
                prev.text.push_str(&cue.text);
                prev.end = cue.end;
            }
            _ => out.push(cue),
        }
    }
    out
}

/// Render cues as SubRip (`.srt`).
pub fn to_srt(cues: &[Cue]) -> String {
    let mut out = String::with_capacity(cues.len() * 64);
    for (i, cue) in cues.iter().enumerate() {
        out.push_str(&format!("{}\n", i + 1));
        out.push_str(&format!(
            "{} --> {}\n",
            srt_timestamp(cue.start),
            srt_timestamp(cue.end)
        ));
        out.push_str(cue.text.trim());
        out.push_str("\n\n");
    }
    out
}

/// `HH:MM:SS,mmm`, the SubRip timestamp format.
pub fn srt_timestamp(seconds: f64) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

/// Convenience: load a media file, transcribe it, return the cues.
#[allow(clippy::too_many_arguments)]
pub fn transcribe_file(
    engine: &Engine,
    path: &Path,
    language: Option<&str>,
    context: &str,
    vad: VadConfig,
    split: &SplitConfig,
    quality: &QualityConfig,
    max_tokens: i32,
) -> Result<SubtitleResult> {
    let samples = crate::media::load_media_16k_mono(path)?;
    if samples.is_empty() {
        anyhow::bail!("no audio decoded from {}", path.display());
    }
    build(
        engine, &samples, language, context, vad, split, quality, max_tokens,
    )
}

/// Load a model for subtitle work.
pub fn load_engine(cfg: &EngineConfig, max_tokens: i32) -> Result<Engine> {
    let _ = max_tokens;
    Engine::load(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srt_timestamp_formats_correctly() {
        assert_eq!(srt_timestamp(0.0), "00:00:00,000");
        assert_eq!(srt_timestamp(1.5), "00:00:01,500");
        assert_eq!(srt_timestamp(61.25), "00:01:01,250");
        assert_eq!(srt_timestamp(3661.007), "01:01:01,007");
        // Negative input must not produce a malformed timestamp.
        assert_eq!(srt_timestamp(-5.0), "00:00:00,000");
    }

    #[test]
    fn renders_valid_srt() {
        let cues = vec![
            Cue { start: 0.5, end: 2.0, text: "第一句".into() },
            Cue { start: 2.0, end: 4.25, text: "第二句".into() },
        ];
        let srt = to_srt(&cues);
        assert!(srt.starts_with("1\n00:00:00,500 --> 00:00:02,000\n第一句\n\n"));
        assert!(srt.contains("2\n00:00:02,000 --> 00:00:04,250\n第二句\n\n"));
    }

    #[test]
    fn splits_on_punctuation() {
        // Long enough to exceed max_chars, so a break is required.
        let cfg = SplitConfig { max_chars: 12, ..Default::default() };
        let pieces = split_text("今天天气不错我们出去走走吧。明天也是好天气呢。", &cfg);
        assert!(pieces.len() >= 2, "got {pieces:?}");
        // Every piece must respect the limit, allowing for the tail rebalance.
        assert!(pieces.iter().all(|p| p.chars().count() <= 13), "got {pieces:?}");
    }

    #[test]
    fn never_splits_inside_a_latin_word() {
        let cfg = SplitConfig { max_chars: 14, min_chars: 2, ..Default::default() };
        // The window would land inside "Windows"; the cut must move past it.
        let pieces = split_text("用于在 Windows 上运行 Linux 容器", &cfg);
        for p in &pieces {
            assert!(
                !p.ends_with("Window") && !p.ends_with("Wind") && !p.ends_with("Win"),
                "a Latin word was halved: {pieces:?}"
            );
            assert!(
                !p.starts_with("indows") && !p.starts_with("dows"),
                "a Latin word was halved: {pieces:?}"
            );
        }
    }

    #[test]
    fn cue_length_stays_bounded_with_latin_text() {
        // Regression: an earlier version extended past the limit to avoid
        // halving a word, with no upper bound, producing 28-character cues
        // from a 24-character limit.
        let cfg = SplitConfig { max_chars: 24, min_chars: 4, ..Default::default() };
        let text = "微软今日宣布正式发布用于在Windows上支持Linux容器的WSLC。";
        let pieces = split_text(text, &cfg);
        for p in &pieces {
            assert!(
                p.chars().count() <= cfg.max_chars,
                "cue exceeded max_chars ({}): {p:?}",
                p.chars().count()
            );
        }
        // And no word was halved in the process.
        for p in &pieces {
            assert!(!p.ends_with("Window"), "halved a word: {pieces:?}");
            assert!(!p.starts_with("indows"), "halved a word: {pieces:?}");
        }
    }

    #[test]
    fn splits_long_text_without_punctuation() {
        let cfg = SplitConfig { max_chars: 10, ..Default::default() };
        let pieces = split_text("一二三四五六七八九十一二三四五六七八九十一二三四五", &cfg);
        assert!(pieces.len() >= 2, "long text must be broken up");
        assert!(pieces.iter().all(|p| p.chars().count() <= 10));
    }

    #[test]
    fn folds_tiny_trailing_piece() {
        let cfg = SplitConfig { max_chars: 10, min_chars: 4, ..Default::default() };
        let pieces = split_text("一二三四五六七八九十。一", &cfg);
        // The lone "一" is too short to stand alone and is folded back.
        assert!(pieces.iter().all(|p| p.chars().count() >= 4), "got {pieces:?}");
    }

    #[test]
    fn short_text_stays_one_cue() {
        let cues = split_cue(1.0, 3.0, "你好", &SplitConfig::default());
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "你好");
        assert!((cues[0].start - 1.0).abs() < 1e-9);
        assert!((cues[0].end - 3.0).abs() < 1e-9);
    }

    #[test]
    fn split_cues_tile_their_segment_without_gaps() {
        let cfg = SplitConfig { max_chars: 12, ..Default::default() };
        let cues = split_cue(0.0, 6.0, "今天天气不错我们出去走走吧。明天也是好天气呢。", &cfg);
        assert!(cues.len() >= 2, "expected several cues, got {cues:?}");
        assert!((cues[0].start - 0.0).abs() < 1e-9, "starts at the segment start");
        assert!(
            (cues.last().unwrap().end - 6.0).abs() < 1e-9,
            "ends at the segment end"
        );
        for pair in cues.windows(2) {
            assert!(
                (pair[0].end - pair[1].start).abs() < 1e-9,
                "cues must be contiguous"
            );
        }
    }

    #[test]
    fn segment_times_convert_frames_to_seconds() {
        // 20 ms frames: frame 50 is 1.0 s, spanning 25 frames is 0.5 s.
        let (s, e) = segment_times(&Segment { start_frame: 50, frames: 25 });
        assert!((s - 1.0).abs() < 1e-9);
        assert!((e - 1.5).abs() < 1e-9);
    }

    #[test]
    fn merges_very_short_cues() {
        let cues = vec![
            Cue { start: 0.0, end: 0.2, text: "嗯".into() },
            Cue { start: 0.2, end: 1.0, text: "好的".into() },
        ];
        let merged = merge_short(cues, &SplitConfig::default());
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].text, "嗯好的");
    }
}

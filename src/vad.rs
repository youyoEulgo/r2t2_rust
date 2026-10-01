// Copyright 2026 youyoEulgo
// SPDX-License-Identifier: Apache-2.0

//! Voice activity detection, and the segment state machine built on it.
//!
//! [`webrtc_vad`] decides per 10/20/30 ms frame whether speech is present. On
//! its own that is too jittery to segment with -- a single misclassified frame
//! would split or merge segments. [`Segmenter`] adds the hysteresis: speech
//! starts only after `min_speech_frames` consecutive voiced frames, and ends
//! only after `min_silence_frames` consecutive unvoiced ones, with an optional
//! majority-vote smoothing window on top.
//!
//! The parameter names follow the Python service's FireRedVAD configuration so
//! the two are recognisably the same design, but **the values do not transfer**:
//! FireRedVAD emits a speech *probability* compared against
//! `speech_threshold = 0.4`, whereas WebRTC VAD emits a boolean from a
//! different algorithm entirely. The defaults here were picked for this
//! detector and should be tuned per deployment.

use anyhow::{bail, Result};
use webrtc_vad::{SampleRate, Vad, VadMode};

/// Frame length the detector consumes, in milliseconds.
///
/// WebRTC VAD accepts only 10, 20 or 30 ms. 20 ms at 16 kHz is 320 samples.
pub const FRAME_MS: usize = 20;

/// Samples per VAD frame at 16 kHz.
pub const FRAME_SAMPLES: usize = crate::audio::TARGET_SAMPLE_RATE as usize * FRAME_MS / 1000;

/// How sensitive the detector is.
///
/// `Aggressive` and above bias towards rejecting non-speech, which suits noisy
/// material (film scores, traffic, room tone) at the cost of occasionally
/// clipping a quiet word. `Quality` is the better choice for clean close-mic
/// audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Sensitivity {
    /// Fewest false negatives; best on clean audio.
    Quality,
    /// For low-bitrate / narrowband audio.
    LowBitrate,
    /// Biased against non-speech. The default: streaming sources carry music
    /// and noise that a permissive setting would happily transcribe.
    #[default]
    Aggressive,
    /// Most aggressive rejection.
    VeryAggressive,
}

impl Sensitivity {
    fn to_mode(self) -> VadMode {
        match self {
            Self::Quality => VadMode::Quality,
            Self::LowBitrate => VadMode::LowBitrate,
            Self::Aggressive => VadMode::Aggressive,
            Self::VeryAggressive => VadMode::VeryAggressive,
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "quality" => Self::Quality,
            "lowbitrate" | "low-bitrate" => Self::LowBitrate,
            "aggressive" => Self::Aggressive,
            "veryaggressive" | "very-aggressive" => Self::VeryAggressive,
            other => bail!("unknown VAD sensitivity {other:?} (quality|lowbitrate|aggressive|veryaggressive)"),
        })
    }
}

/// Tuning for [`Segmenter`].
#[derive(Debug, Clone)]
pub struct VadConfig {
    pub sensitivity: Sensitivity,
    /// Consecutive voiced frames required to open a segment.
    pub min_speech_frames: usize,
    /// Consecutive unvoiced frames required to close one.
    pub min_silence_frames: usize,
    /// Majority-vote window over the raw per-frame decisions. `1` disables it.
    pub smooth_window: usize,
    /// Hard cap on a single segment, in frames. Guards against a detector that
    /// never reports silence (continuous music, for instance).
    pub max_speech_frames: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            sensitivity: Sensitivity::default(),
            // 8 frames x 20 ms = 160 ms of speech to open. Below that is
            // usually a click or a breath.
            min_speech_frames: 8,
            // 20 frames x 20 ms = 400 ms of silence to close. Longer than the
            // Python service's 200 ms because WebRTC VAD toggles more readily;
            // this keeps ordinary mid-sentence pauses from splitting a segment.
            min_silence_frames: 20,
            smooth_window: 5,
            // 30 s, matching the Python service's max_speech_frame=2000 at a
            // 10 ms frame size.
            max_speech_frames: 1500,
        }
    }
}

/// A detected run of speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    /// Offset of the segment start from the beginning of the stream, in frames.
    pub start_frame: u64,
    /// Number of frames in the segment.
    pub frames: u64,
}

/// Turns a stream of frames into speech segments.
pub struct Segmenter {
    vad: Vad,
    config: VadConfig,
    /// Ring buffer of recent raw decisions, for majority-vote smoothing.
    history: std::collections::VecDeque<bool>,
    /// Smoothed decisions currently inside an open segment (trailing run).
    in_speech: bool,
    /// Consecutive voiced frames seen while outside a segment.
    speech_run: usize,
    /// Consecutive unvoiced frames seen while inside a segment.
    silence_run: usize,
    segment_start: u64,
    segment_frames: u64,
    /// Total frames pushed, used for absolute offsets.
    frame_index: u64,
}

// SAFETY: the only non-`Send` member is `webrtc_vad::Vad`, which wraps an
// opaque `Fvad *` allocated by the C library. That pointer owns no Rust data
// and refers to no thread-local state, so moving the `Segmenter` to another
// thread is sound. The crate simply does not declare the impl itself.
//
// `Sync` is deliberately *not* implemented: `push_frame` takes `&mut self`
// precisely because the detector holds mutable state, and the type system
// should keep enforcing that.
unsafe impl Send for Segmenter {}

impl Segmenter {
    pub fn new(config: VadConfig) -> Result<Self> {
        let vad = Vad::new_with_rate_and_mode(SampleRate::Rate16kHz, config.sensitivity.to_mode());
        Ok(Self {
            vad,
            history: std::collections::VecDeque::with_capacity(config.smooth_window.max(1)),
            config,
            in_speech: false,
            speech_run: 0,
            silence_run: 0,
            segment_start: 0,
            segment_frames: 0,
            frame_index: 0,
        })
    }

    /// Whether a segment is currently open.
    pub fn in_speech(&self) -> bool {
        self.in_speech
    }

    /// Frames elapsed in the open segment.
    pub fn current_frames(&self) -> u64 {
        self.segment_frames
    }

    /// Feed exactly one frame of [`FRAME_SAMPLES`] samples.
    ///
    /// Returns `Some(segment)` when this frame *closes* a segment. The caller
    /// is expected to have been accumulating audio since the previous close.
    pub fn push_frame(&mut self, frame: &[i16]) -> Result<Option<Segment>> {
        if frame.len() != FRAME_SAMPLES {
            bail!(
                "VAD frame must be {FRAME_SAMPLES} samples ({} ms at 16 kHz), got {}",
                FRAME_MS,
                frame.len()
            );
        }

        let raw = self
            .vad
            .is_voice_segment(frame)
            .map_err(|_| anyhow::anyhow!("WebRTC VAD rejected a {FRAME_SAMPLES}-sample frame"))?;

        // Majority vote over the smoothing window.
        self.history.push_back(raw);
        if self.history.len() > self.config.smooth_window.max(1) {
            self.history.pop_front();
        }
        let voiced = if self.config.smooth_window <= 1 {
            raw
        } else {
            let yes = self.history.iter().filter(|v| **v).count();
            yes * 2 > self.history.len()
        };

        self.frame_index += 1;
        let mut closed = None;

        if self.in_speech {
            self.segment_frames += 1;
            if voiced {
                self.silence_run = 0;
            } else {
                self.silence_run += 1;
            }
            // Close on sustained silence, or when the segment hits its cap.
            if self.silence_run >= self.config.min_silence_frames
                || self.segment_frames >= self.config.max_speech_frames as u64
            {
                closed = Some(Segment {
                    start_frame: self.segment_start,
                    frames: self.segment_frames,
                });
                self.in_speech = false;
                self.segment_frames = 0;
                self.silence_run = 0;
                self.speech_run = 0;
            }
        } else if voiced {
            self.speech_run += 1;
            if self.speech_run >= self.config.min_speech_frames {
                // Back-date the start so the leading frames that opened the
                // segment are included, otherwise quiet onsets get clipped.
                self.in_speech = true;
                self.segment_start = self.frame_index - self.speech_run as u64;
                self.segment_frames = self.speech_run as u64;
                self.silence_run = 0;
            }
        } else {
            self.speech_run = 0;
        }

        Ok(closed)
    }

    /// Close any open segment at end of stream.
    ///
    /// Trailing silence is trimmed: the tail that triggered the close is not
    /// speech, and keeping it would pad every final subtitle.
    pub fn flush(&mut self) -> Option<Segment> {
        if !self.in_speech || self.segment_frames == 0 {
            self.in_speech = false;
            self.segment_frames = 0;
            return None;
        }
        let frames = self
            .segment_frames
            .saturating_sub(self.silence_run as u64)
            .max(1);
        let seg = Segment {
            start_frame: self.segment_start,
            frames,
        };
        self.in_speech = false;
        self.segment_frames = 0;
        self.silence_run = 0;
        Some(seg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Broadband noise: WebRTC VAD classifies this as speech.
    fn voiced_frame() -> Vec<i16> {
        (0..FRAME_SAMPLES)
            .map(|i| (((i * 7919) % 20000) as i32 - 10000) as i16)
            .collect()
    }

    /// Digital silence.
    fn silent_frame() -> Vec<i16> {
        vec![0i16; FRAME_SAMPLES]
    }

    /// Push `count` copies of `frame`, returning the first segment closed.
    ///
    /// WebRTC VAD has a built-in hangover: after a voiced frame it keeps
    /// reporting voice for roughly three frames regardless of input. Tests
    /// therefore drive it with enough frames for that to wash out rather than
    /// assuming a one-frame latency.
    fn push_n(seg: &mut Segmenter, frame: &[i16], count: usize) -> Option<Segment> {
        let mut closed = None;
        for _ in 0..count {
            if let Some(s) = seg.push_frame(frame).unwrap() {
                closed = closed.or(Some(s));
            }
        }
        closed
    }

    #[test]
    fn rejects_wrong_frame_length() {
        let mut seg = Segmenter::new(VadConfig::default()).unwrap();
        assert!(seg.push_frame(&[0i16; 100]).is_err());
    }

    #[test]
    fn silence_alone_never_opens_a_segment() {
        let mut seg = Segmenter::new(VadConfig::default()).unwrap();
        let closed = push_n(&mut seg, &silent_frame(), 60);
        assert!(closed.is_none(), "pure silence must not produce a segment");
        assert!(!seg.in_speech());
    }

    #[test]
    fn opens_after_min_speech_frames() {
        let cfg = VadConfig {
            smooth_window: 1,
            min_speech_frames: 3,
            ..Default::default()
        };
        let mut seg = Segmenter::new(cfg).unwrap();
        let v = voiced_frame();
        seg.push_frame(&v).unwrap();
        assert!(!seg.in_speech(), "one frame is below the threshold");
        seg.push_frame(&v).unwrap();
        assert!(!seg.in_speech(), "two frames is below the threshold");
        seg.push_frame(&v).unwrap();
        assert!(seg.in_speech(), "third frame opens the segment");
    }

    #[test]
    fn closes_after_min_silence_frames() {
        let cfg = VadConfig {
            smooth_window: 1,
            min_speech_frames: 2,
            min_silence_frames: 3,
            ..Default::default()
        };
        let mut seg = Segmenter::new(cfg).unwrap();
        // Enough speech to open and to outlast the detector's hangover.
        push_n(&mut seg, &voiced_frame(), 10);
        assert!(seg.in_speech());

        let closed = push_n(&mut seg, &silent_frame(), 12);
        let closed = closed.expect("sustained silence must close the segment");
        assert!(!seg.in_speech());
        assert!(
            closed.frames >= 10,
            "the segment must include the speech frames, got {}",
            closed.frames
        );
    }

    #[test]
    fn brief_pause_does_not_split() {
        let cfg = VadConfig {
            smooth_window: 1,
            min_speech_frames: 2,
            min_silence_frames: 30,
            ..Default::default()
        };
        let mut seg = Segmenter::new(cfg).unwrap();
        push_n(&mut seg, &voiced_frame(), 10);
        assert!(seg.in_speech());
        // A gap far shorter than min_silence_frames.
        push_n(&mut seg, &silent_frame(), 3);
        assert!(seg.in_speech(), "a short pause is still the same segment");
    }

    #[test]
    fn flush_trims_trailing_silence() {
        let cfg = VadConfig {
            smooth_window: 1,
            min_speech_frames: 2,
            min_silence_frames: 999, // never auto-close
            ..Default::default()
        };
        let mut seg = Segmenter::new(cfg).unwrap();
        push_n(&mut seg, &voiced_frame(), 10);
        // Long enough to outlast the detector's hangover (~3 frames), so the
        // tail really is counted as silence.
        push_n(&mut seg, &silent_frame(), 12);
        let during = seg.current_frames();

        let closed = seg.flush().expect("flush closes the open segment");
        assert!(
            closed.frames < during,
            "trailing silence must be trimmed: {} !< {}",
            closed.frames,
            during
        );
        assert!(!seg.in_speech());
    }

    #[test]
    fn flush_without_segment_is_none() {
        let mut seg = Segmenter::new(VadConfig::default()).unwrap();
        assert!(seg.flush().is_none());
    }

    #[test]
    fn max_speech_frames_caps_a_segment() {
        let cfg = VadConfig {
            smooth_window: 1,
            min_speech_frames: 1,
            min_silence_frames: 999,
            max_speech_frames: 20,
            ..Default::default()
        };
        let mut seg = Segmenter::new(cfg).unwrap();
        let closed = push_n(&mut seg, &voiced_frame(), 40)
            .expect("the cap must force a close even with no silence");
        assert!(closed.frames <= 21, "segment exceeded the cap: {}", closed.frames);
        assert!(!seg.in_speech());
    }

    #[test]
    fn sensitivity_parses_all_levels() {
        assert_eq!(Sensitivity::parse("quality").unwrap(), Sensitivity::Quality);
        assert_eq!(Sensitivity::parse("Aggressive").unwrap(), Sensitivity::Aggressive);
        assert_eq!(
            Sensitivity::parse("very-aggressive").unwrap(),
            Sensitivity::VeryAggressive
        );
        assert!(Sensitivity::parse("nonsense").is_err());
    }
}

//! Output quality guards: repetition repair and hallucination detection.
//!
//! ASR models fail in two characteristic ways on difficult audio, and both are
//! worse than they sound -- a subtitle file cannot be silently wrong, and a
//! looping stream never recovers on its own.
//!
//! * **Repetition.** Decoding can get stuck emitting the same character or
//!   phrase (`的的的的的的`, `Okay. Okay. Okay. Okay.`). [`fix_repetitions`]
//!   collapses these.
//! * **Hallucination.** Given silence, music, or noise, the model may invent
//!   fluent text with no relation to the audio. [`detect_hallucination`]
//!   catches the common shape of this failure -- a phrase looped to the end of
//!   the output -- so the caller can discard that span.
//!
//! Both are ports of the reference implementation in the Python server
//! (`ws_server.py`), kept deliberately close to it: these are heuristics tuned
//! against real failures, and "improving" them without the same test material
//! is more likely to introduce false positives than to help.
//!
//! A third failure mode is not covered here and cannot be: confidently wrong
//! but fluent text (`会话容器` heard as `绘画容器`). That needs a language model
//! or a hotword list, not a heuristic.

/// Punctuation ignored when normalising text for pattern matching.
const PUNCT: &str = "，。！？、；：,.!?;:~…·\"'()（）《》—-";

/// Collapse runs of a repeated character, and then loops of a repeated phrase.
///
/// `threshold` is the number of repeats that counts as "stuck": a run must be
/// *longer* than this to be collapsed, so ordinary Chinese (`看看`, `刚刚`) and
/// deliberate emphasis survive.
pub fn fix_repetitions(text: &str, threshold: usize) -> String {
    let chars_fixed = fix_char_repeats(text, threshold);
    fix_pattern_repeats(&chars_fixed, threshold, 20)
}

/// `的的的的的的` (6 repeats) becomes `的` when the threshold is 5.
///
/// Runs at or below the threshold are left alone, so `的的` and `好好好` pass
/// through untouched.
fn fix_char_repeats(s: &str, threshold: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;

    while i < chars.len() {
        let mut count = 1;
        while i + count < chars.len() && chars[i + count] == chars[i] {
            count += 1;
        }
        if count > threshold {
            out.push(chars[i]);
        } else {
            for _ in 0..count {
                out.push(chars[i]);
            }
        }
        i += count;
    }
    out
}

/// Collapse a phrase repeated `threshold` or more times.
///
/// Scans for the first position where any pattern of length 1..=`max_len`
/// repeats `threshold` times consecutively, emits that pattern once, and
/// recurses on the remainder. This catches `Okay. Okay. Okay. Okay. Okay.`
/// without needing to know the phrase in advance.
fn fix_pattern_repeats(s: &str, threshold: usize, max_len: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let min_repeat_chars = threshold * 2;
    if n < min_repeat_chars {
        return s.to_string();
    }

    let mut i = 0;
    let mut result = String::with_capacity(s.len());

    while i + min_repeat_chars <= n {
        let mut found = false;

        for k in 1..=max_len {
            if i + k * threshold > n {
                break;
            }
            let pattern = &chars[i..i + k];

            let matches = (1..threshold).all(|rep| {
                let start = i + rep * k;
                &chars[start..start + k] == pattern
            });
            if !matches {
                continue;
            }

            // Extend past the required count to absorb the whole loop.
            let mut end = i + threshold * k;
            while end + k <= n && &chars[end..end + k] == pattern {
                end += k;
            }

            result.extend(pattern.iter());
            let rest: String = chars[end..].iter().collect();
            result.push_str(&fix_pattern_repeats(&rest, threshold, max_len));
            i = n;
            found = true;
            break;
        }

        if found {
            break;
        }
        result.push(chars[i]);
        i += 1;
    }

    if i < n {
        result.extend(chars[i..].iter());
    }
    result
}

/// Whether the tail of `text` looks like a hallucination loop.
///
/// Returns the reason when it does, for logging.
///
/// Two passes, matching the reference:
/// 1. **Exact** -- the last `k` characters repeat `repeat_threshold` times.
/// 2. **Normalised** -- same, after stripping punctuation and whitespace, which
///    catches noisy variants such as `Okay. O kay, okay.`
///
/// Only the final `tail_len` characters are examined: a single repeated phrase
/// somewhere in the middle of an hour of speech is not evidence of a loop,
/// whereas one running to the end is.
pub fn detect_hallucination(
    text: &str,
    repeat_threshold: usize,
    max_pattern_len: usize,
    tail_len: usize,
) -> Option<String> {
    if text.is_empty() {
        return None;
    }
    let chars: Vec<char> = text.chars().collect();
    let tail: Vec<char> = chars[chars.len().saturating_sub(tail_len)..].to_vec();
    let n = tail.len();

    // Pass 1: exact.
    for k in 1..=max_pattern_len {
        if n < k * repeat_threshold {
            continue;
        }
        let pattern = &tail[n - k..];
        if is_blank(pattern) {
            continue; // pure punctuation; the normalised pass covers it
        }
        let repeats = (1..repeat_threshold).all(|r| {
            let end = n - r * k;
            &tail[end - k..end] == pattern
        });
        if repeats {
            return Some(format!(
                "tail_pattern:'{}'x{}",
                pattern.iter().collect::<String>(),
                repeat_threshold
            ));
        }
    }

    // Pass 2: punctuation- and whitespace-insensitive. Starts at k = 3 because
    // one- and two-character patterns false-positive on ordinary English.
    let normalised = normalise_for_pattern(&tail);
    let n2 = normalised.len();
    for k in 3..=max_pattern_len {
        if n2 < k * repeat_threshold {
            continue;
        }
        let pattern = &normalised[n2 - k..];
        let repeats = (1..repeat_threshold).all(|r| {
            let end = n2 - r * k;
            &normalised[end - k..end] == pattern
        });
        if repeats {
            return Some(format!(
                "tail_pattern_norm:'{}'x{}",
                pattern.iter().collect::<String>(),
                repeat_threshold
            ));
        }
    }

    None
}

/// True when every character is punctuation or whitespace.
fn is_blank(chars: &[char]) -> bool {
    chars
        .iter()
        .all(|c| PUNCT.contains(*c) || c.is_whitespace())
}

/// Lowercase and drop punctuation and whitespace.
fn normalise_for_pattern(chars: &[char]) -> Vec<char> {
    chars
        .iter()
        .filter(|c| !PUNCT.contains(**c) && !c.is_whitespace())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// Whether a decoded segment is worth keeping as a subtitle cue.
///
/// Empty, punctuation-only, and character-loop outputs all mean the model had
/// nothing to say -- usually a segment the detector opened around a noise
/// rather than speech.
///
/// A single character is *not* degenerate: one short syllable can be a real
/// word (`好`, `嗯`), and dropping it would lose content. Only a repeated run
/// signals a stuck decoder.
pub fn is_degenerate(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }
    let chars: Vec<char> = trimmed.chars().collect();
    if is_blank(&chars) {
        return true;
    }
    // A single character repeated three or more times, e.g. "。。。。" or "啊啊啊".
    if chars.len() >= 3 && chars.iter().all(|c| *c == chars[0]) {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- repetition repair ------------------------------------------------

    #[test]
    fn collapses_long_character_runs() {
        assert_eq!(fix_repetitions("的的的的的的", 5), "的");
        assert_eq!(fix_repetitions("好好好好好好好", 5), "好");
    }

    #[test]
    fn keeps_short_runs() {
        // Ordinary doubled characters are not repetition.
        assert_eq!(fix_repetitions("看看这个", 5), "看看这个");
        assert_eq!(fix_repetitions("好好好好好", 5), "好好好好好");
    }

    #[test]
    fn collapses_phrase_loops() {
        let out = fix_repetitions("Okay. Okay. Okay. Okay. Okay. ", 5);
        assert_eq!(out.matches("Okay").count(), 1, "got {out:?}");
    }

    #[test]
    fn collapses_longer_phrase_loops() {
        let out = fix_repetitions("感谢观看感谢观看感谢观看感谢观看感谢观看", 5);
        assert_eq!(out, "感谢观看", "got {out:?}");
    }

    #[test]
    fn leaves_normal_text_untouched() {
        let text = "微软宣布推出WSLC，支持在Windows上运行Linux容器。";
        assert_eq!(fix_repetitions(text, 5), text);
    }

    // ---- hallucination detection -----------------------------------------

    #[test]
    fn flags_exact_tail_loop() {
        let text = "Okay. Okay. Okay. Okay. Okay.";
        assert!(detect_hallucination(text, 5, 50, 256).is_some());
    }

    #[test]
    fn flags_normalised_tail_loop() {
        // Same phrase with inconsistent punctuation and spacing.
        let text = "okay. O kay, okay. okay! okay";
        assert!(
            detect_hallucination(text, 5, 50, 256).is_some(),
            "noisy repetition should still be caught"
        );
    }

    #[test]
    fn passes_ordinary_speech() {
        let text = "微软宣布推出WSLC，支持在Windows上运行Linux容器。\
                    作者Michael Larebell，继发布了WSL上的Linux容器公开预览版后。";
        assert!(detect_hallucination(text, 5, 50, 256).is_none());
    }

    #[test]
    fn ignores_repetition_that_does_not_reach_the_end() {
        // A loop in the middle is not a runaway; speech resumed afterwards.
        let text = "Okay. Okay. Okay. Okay. Okay. 然后我们继续讨论下一个话题。";
        assert!(
            detect_hallucination(text, 5, 50, 256).is_none(),
            "only a loop running to the end should trigger"
        );
    }

    #[test]
    fn ignores_empty_and_punctuation_only() {
        assert!(detect_hallucination("", 5, 50, 256).is_none());
        // Pure punctuation repeats are handled by the degenerate check instead.
        assert!(detect_hallucination(".....", 5, 50, 256).is_none());
    }

    // ---- degeneracy -------------------------------------------------------

    #[test]
    fn detects_degenerate_outputs() {
        assert!(is_degenerate(""));
        assert!(is_degenerate("   "));
        assert!(is_degenerate("。。。"));
        assert!(is_degenerate("啊啊啊啊"));
        // A single character is a real word, not a stuck decoder.
        assert!(!is_degenerate("好"));
        assert!(!is_degenerate("嗯"));
        assert!(!is_degenerate("微软宣布"));
    }
}

// Copyright 2026 youyoEulgo
// SPDX-License-Identifier: Apache-2.0
//
// This file reproduces `build_asr_prompt` from Confucius4-R2T2's
// `r2t2_llama/llama_native_backend.py`
//
// Modifications are described in NOTICE; they are not endorsed by
// the original authors.

//! Prompt construction.
//!
//! This reproduces `build_asr_prompt()` from the Python project's
//! `r2t2_llama/llama_native_backend.py` exactly -- including the newlines and
//! the trailing `language ...<asr_text>` suffix. A single stray character
//! changes what the model emits, so the literal lives in one place and is
//! covered by a test.

/// Build the chat-templated prompt for one transcription.
///
/// `context` is an optional hotword / topic hint; `language` is the language
/// hint, or `None` to let the model decide.
pub fn build(context: &str, language: Option<&str>, assistant_prefix: &str) -> String {
    let mut prompt = String::new();
    prompt.push_str("<|im_start|>system\n");
    prompt.push_str(context);
    prompt.push_str("<|im_end|>\n");
    prompt.push_str("<|im_start|>user\n");
    prompt.push_str("<|audio_start|><|audio_pad|><|audio_end|>");
    prompt.push_str("<|im_end|>\n");
    prompt.push_str("<|im_start|>assistant\n");
    if let Some(lang) = language {
        prompt.push_str(&format!("language {lang}<asr_text>"));
    }
    prompt.push_str(assistant_prefix);
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python_reference_byte_for_byte() {
        // Copied from the Python implementation.
        let expected = "<|im_start|>system\n\
                        <|im_end|>\n\
                        <|im_start|>user\n\
                        <|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n\
                        <|im_start|>assistant\n\
                        language Chinese<asr_text>";
        assert_eq!(build("", Some("Chinese"), ""), expected);
    }

    #[test]
    fn omits_language_suffix_when_absent() {
        let p = build("", None, "");
        assert!(p.ends_with("<|im_start|>assistant\n"));
        assert!(!p.contains("<asr_text>"));
    }

    #[test]
    fn includes_context_when_given() {
        let p = build("热词：有道", Some("Chinese"), "");
        assert!(p.starts_with("<|im_start|>system\n热词：有道<|im_end|>\n"));
    }
}

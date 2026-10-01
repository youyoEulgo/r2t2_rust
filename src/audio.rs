// Copyright 2026 youyoEulgo
// SPDX-License-Identifier: Apache-2.0

//! Audio loading: any WAV in, mono 16 kHz `f32` out.
//!
//! llama.cpp's `mtmd_bitmap_init_from_audio` expects mono 16 kHz samples, so
//! anything else has to be converted here. Resampling is linear interpolation,
//! which matches `_resample_to_16k()` in the Python project's `example.py` --
//! keeping the two routes numerically identical matters more here than
//! resampler quality, since the goal is to reproduce the same transcript.

use std::path::Path;

use anyhow::{bail, Context, Result};

/// Sample rate the model expects.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Read a WAV file and return mono 16 kHz `f32` samples scaled to `[-1, 1]`.
pub fn load_wav_16k_mono(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    let path = path.as_ref();
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("could not open WAV file: {}", path.display()))?;
    let spec = reader.spec();

    if spec.channels == 0 {
        bail!("WAV file reports zero channels: {}", path.display());
    }

    // Decode every sample to f32 in [-1, 1].
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .context("failed to read WAV samples")?,
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max = (1i64 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .collect::<Result<Vec<_>, _>>()
                .context("failed to read WAV samples")?
                .into_iter()
                .map(|s| s as f32 / max)
                .collect()
        }
    };

    let mono = downmix(&samples, spec.channels as usize);
    resample(&mono, spec.sample_rate, TARGET_SAMPLE_RATE)
}

/// Average interleaved channels down to mono.
fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Linear-interpolation resampler, matching the Python reference.
fn resample(input: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    if from == to || input.is_empty() {
        return Ok(input.to_vec());
    }
    if from == 0 {
        bail!("source sample rate is zero");
    }

    let duration = input.len() as f64 / from as f64;
    let out_len = (duration * to as f64).round() as usize;
    if out_len == 0 {
        return Ok(Vec::new());
    }

    // Position of each input sample and each output sample on a common time
    // axis, then interpolate. This is what np.interp does in the Python code.
    let step_in = duration / input.len() as f64;
    let step_out = duration / out_len as f64;

    let mut out = Vec::with_capacity(out_len);
    let mut j = 0usize; // index of the input sample at or before the cursor
    for i in 0..out_len {
        let t = i as f64 * step_out;
        while j + 1 < input.len() && (j + 1) as f64 * step_in <= t {
            j += 1;
        }
        if j + 1 >= input.len() {
            out.push(input[input.len() - 1]);
            continue;
        }
        let t0 = j as f64 * step_in;
        let t1 = (j + 1) as f64 * step_in;
        let frac = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
        out.push(input[j] + (input[j + 1] - input[j]) * frac as f32);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_averages_channels() {
        // Two stereo frames: (1.0, 0.0) and (0.0, 1.0).
        let got = downmix(&[1.0, 0.0, 0.0, 1.0], 2);
        assert_eq!(got, vec![0.5, 0.5]);
    }

    #[test]
    fn mono_passes_through() {
        let got = downmix(&[0.1, 0.2, 0.3], 1);
        assert_eq!(got, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn resample_is_identity_at_same_rate() {
        let input = vec![0.1, 0.2, 0.3];
        assert_eq!(resample(&input, 16_000, 16_000).unwrap(), input);
    }

    #[test]
    fn resample_doubles_length_when_uprating() {
        let input: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let got = resample(&input, 8_000, 16_000).unwrap();
        assert_eq!(got.len(), 200);
        // Endpoints should be preserved.
        assert!((got[0] - 0.0).abs() < 1e-6);
        assert!((got[got.len() - 1] - 0.99).abs() < 1e-3);
    }

    #[test]
    fn resample_halves_length_when_downrating() {
        let input: Vec<f32> = (0..200).map(|i| i as f32 / 200.0).collect();
        let got = resample(&input, 16_000, 8_000).unwrap();
        assert_eq!(got.len(), 100);
    }
}

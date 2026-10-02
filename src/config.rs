// Copyright 2026 youyoEulgo
// SPDX-License-Identifier: Apache-2.0

//! The configuration file.
//!
//! A single TOML file at `~/.local/share/r2t2/config.toml`, holding the
//! settings that outlive one run. Everything here has a default, and the file
//! is created on first use, so the program runs without it.
//!
//! # Why a file rather than command-line flags
//!
//! These are appearance settings for the caption overlay, which OBS loads as a
//! browser source. A flag would have to be repeated on every start, and a query
//! parameter would have to be typed into OBS by hand and retyped whenever it
//! changed. A file is written once, read by both the console and the overlay,
//! and can be edited in a text editor by anyone who prefers that.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// How the caption overlay is drawn.
///
/// Every field has a `Default`, so a partial file works and a missing file
/// means "all defaults".
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CaptionConfig {
    /// How many lines stay on screen, including the one being spoken.
    pub lines: usize,

    /// How many characters fit on a line before it is pushed up.
    ///
    /// Counted in full-width units, so two Latin letters equal one CJK
    /// character. That is what makes a line look the same length either way,
    /// which counting raw characters does not.
    pub chars: usize,

    /// Font size in pixels, relative to a 1080p frame.
    pub size: usize,

    /// Text colour.
    pub color: String,

    /// Background behind each line, as a CSS colour. Set `transparent` to draw
    /// no background at all.
    pub background: String,

    /// Distance from the bottom of the frame, as a CSS length.
    pub bottom: String,

    /// Drop the background bar, for compositing directly onto video.
    pub transparent: bool,
}

impl Default for CaptionConfig {
    fn default() -> Self {
        Self {
            lines: 2,
            chars: 20,
            size: 48,
            color: "#ffffff".to_string(),
            background: "rgba(0, 0, 0, 0.62)".to_string(),
            bottom: "6%".to_string(),
            transparent: false,
        }
    }
}

impl CaptionConfig {
    /// Clamp to values the overlay can actually render.
    ///
    /// A config file is meant to be edited by hand, so it will contain nonsense
    /// sooner or later. Clamping beats refusing to start: the caption still
    /// appears, and the console shows what was used.
    pub fn sanitized(mut self) -> Self {
        self.lines = self.lines.clamp(1, 5);
        self.chars = self.chars.clamp(4, 120);
        self.size = self.size.clamp(8, 200);
        self
    }
}

/// The whole configuration file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub caption: CaptionConfig,
}

impl Config {
    /// Read the configuration, falling back to defaults.
    ///
    /// A missing file is not an error — it is the normal first run. A file that
    /// exists but cannot be parsed *is* reported, because silently ignoring it
    /// would leave someone editing a file that has no effect.
    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => match toml::from_str::<Config>(&text) {
                Ok(cfg) => cfg.sanitized(),
                Err(err) => {
                    warn!(
                        path = %path.display(),
                        error = %err,
                        "ignoring the configuration file; it could not be parsed"
                    );
                    Config::default()
                }
            },
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Config::default(),
            Err(err) => {
                warn!(path = %path.display(), error = %err, "could not read the configuration");
                Config::default()
            }
        }
    }

    fn sanitized(mut self) -> Self {
        self.caption = self.caption.sanitized();
        self
    }

    /// Write the configuration, creating the directory if needed.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("could not create {}", parent.display()))?;
        }
        let text = toml::to_string_pretty(self).context("could not serialise the config")?;

        // Write to a neighbouring file and rename, so an interrupted write
        // cannot leave a half-written config behind.
        let tmp = path.with_extension("toml.tmp");
        std::fs::write(&tmp, &text)
            .with_context(|| format!("could not write {}", tmp.display()))?;
        std::fs::rename(&tmp, path)
            .with_context(|| format!("could not replace {}", path.display()))?;
        Ok(())
    }

    /// Load the configuration from the standard location.
    pub fn load_default() -> Self {
        Self::load(&crate::paths::config_file())
    }

    /// Write the file if it is not there yet, so there is something to edit.
    ///
    /// Called at startup: a file that does not exist is easy to forget about,
    /// and the console offers no way to discover it.
    pub fn ensure_file(path: &Path) -> Result<()> {
        if path.exists() {
            return Ok(());
        }
        let cfg = Config::default();
        cfg.save(path)?;
        info!(path = %path.display(), "wrote a default configuration file");
        Ok(())
    }
}

/// Where the configuration lives, for messages.
pub fn default_path() -> PathBuf {
    crate::paths::config_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_usable() {
        let c = CaptionConfig::default();
        assert_eq!(c.lines, 2);
        assert_eq!(c.chars, 20);
        assert!(c.size > 0);
    }

    #[test]
    fn a_partial_file_keeps_the_other_defaults() {
        let cfg: Config = toml::from_str("[caption]\nlines = 3\n").unwrap();
        let cfg = cfg.sanitized();
        assert_eq!(cfg.caption.lines, 3);
        // Untouched fields keep their defaults.
        assert_eq!(cfg.caption.chars, 20);
        assert_eq!(cfg.caption.size, 48);
    }

    #[test]
    fn nonsense_is_clamped_rather_than_rejected() {
        let cfg: Config = toml::from_str("[caption]\nlines = 999\nchars = 0\nsize = 1\n").unwrap();
        let cfg = cfg.sanitized();
        assert_eq!(cfg.caption.lines, 5);
        assert_eq!(cfg.caption.chars, 4);
        assert_eq!(cfg.caption.size, 8);
    }

    #[test]
    fn a_round_trip_preserves_everything() {
        let dir = std::env::temp_dir().join("r2t2-config-test");
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("config.toml");

        let mut cfg = Config::default();
        cfg.caption.lines = 4;
        cfg.caption.transparent = true;
        cfg.save(&path).unwrap();

        let back = Config::load(&path);
        assert_eq!(back.caption.lines, 4);
        assert!(back.caption.transparent);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_file_is_not_an_error() {
        let cfg = Config::load(Path::new("/nonexistent/r2t2/config.toml"));
        assert_eq!(cfg.caption.lines, 2);
    }

    #[test]
    fn a_broken_file_falls_back_to_defaults() {
        let dir = std::env::temp_dir().join("r2t2-config-broken");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(&path, "this is not toml {{{").unwrap();

        let cfg = Config::load(&path);
        assert_eq!(cfg.caption.lines, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

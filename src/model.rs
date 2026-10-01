// Copyright 2026 youyoEulgo
// SPDX-License-Identifier: Apache-2.0

//! Locating the model's GGUF pair.
//!
//! The language model and the audio projector are separate GGUF files that must
//! be loaded together. `mmproj*` names the projector and the other `.gguf` names
//! the language model, matching the convention the model repository uses.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

/// Find the paired GGUF files in `dir`, returning `(model, projector)`.
///
/// Exactly one of each is required. That strictness is the point: dropping
/// several quantisations into one directory is an easy mistake, and picking
/// whichever happens to sort first would silently transcribe with the wrong
/// model rather than failing.
pub fn resolve_gguf(dir: &Path) -> Result<(PathBuf, PathBuf)> {
    if !dir.is_dir() {
        bail!("gguf directory not found: {}", dir.display());
    }

    let mut models = Vec::new();
    let mut projectors = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("could not read {}", dir.display()))? {
        let path = entry?.path();
        if !path.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
            continue;
        }
        let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        if name.starts_with("mmproj") {
            projectors.push(path);
        } else {
            models.push(path);
        }
    }
    models.sort();
    projectors.sort();

    match (models.len(), projectors.len()) {
        (1, 1) => Ok((models.remove(0), projectors.remove(0))),
        (0, 0) => bail!("no GGUF files found in {}", dir.display()),
        (0, _) => bail!(
            "no language-model GGUF in {} (found {} projector(s) but no plain *.gguf)",
            dir.display(),
            projectors.len()
        ),
        (_, 0) => bail!(
            "no projector GGUF in {} (expected one 'mmproj*.gguf')",
            dir.display()
        ),
        (m, p) => bail!(
            "expected exactly one mmproj*.gguf and one other *.gguf in {}, found {m} model(s) and {p} projector(s); \
             keep one quantisation per directory",
            dir.display()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a throwaway directory holding the named files.
    struct TmpDir(PathBuf);

    impl TmpDir {
        fn new(tag: &str, files: &[&str]) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "r2t2-gguf-{tag}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            for f in files {
                std::fs::write(dir.join(f), b"x").unwrap();
            }
            Self(dir)
        }
    }

    impl Drop for TmpDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn accepts_one_pair() {
        let d = TmpDir::new("pair", &["Confucius4-R2T2-Q8_0.gguf", "mmproj-Confucius4-R2T2-Q8_0.gguf"]);
        let (m, p) = resolve_gguf(&d.0).unwrap();
        assert!(m.to_string_lossy().contains("Confucius4"));
        assert!(!m.to_string_lossy().contains("mmproj"));
        assert!(p.to_string_lossy().contains("mmproj"));
    }

    #[test]
    fn ignores_non_gguf_files() {
        let d = TmpDir::new("extra", &["model.gguf", "mmproj-x.gguf", "README.md", "config.json"]);
        assert!(resolve_gguf(&d.0).is_ok());
    }

    #[test]
    fn rejects_two_models() {
        let d = TmpDir::new("two", &["a.gguf", "b.gguf", "mmproj-x.gguf"]);
        let err = resolve_gguf(&d.0).unwrap_err().to_string();
        assert!(err.contains("exactly one"), "got {err}");
    }

    #[test]
    fn rejects_two_projectors() {
        let d = TmpDir::new("twoproj", &["a.gguf", "mmproj-x.gguf", "mmproj-y.gguf"]);
        assert!(resolve_gguf(&d.0).is_err());
    }

    #[test]
    fn rejects_projector_without_model() {
        let d = TmpDir::new("noproj", &["mmproj-x.gguf"]);
        let err = resolve_gguf(&d.0).unwrap_err().to_string();
        assert!(err.contains("language-model"), "got {err}");
    }

    #[test]
    fn rejects_empty_dir() {
        let d = TmpDir::new("empty", &[]);
        assert!(resolve_gguf(&d.0).is_err());
    }

    #[test]
    fn rejects_missing_dir() {
        let missing = std::env::temp_dir().join("r2t2-definitely-not-here");
        let err = resolve_gguf(&missing).unwrap_err().to_string();
        assert!(err.contains("not found"), "got {err}");
    }
}

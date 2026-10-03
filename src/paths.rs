// Copyright 2026 youyoEulgo
// SPDX-License-Identifier: Apache-2.0

//! Where the program keeps its data, and how it obtains a model.
//!
//! Everything lives under one directory, following the XDG base directory
//! specification:
//!
//! ```text
//! ~/.local/share/r2t2/
//!   models/     the GGUF pair
//!   work/       uploads and their results (the web interface)
//! ```
//!
//! `$XDG_DATA_HOME` overrides the root, so a user who has relocated their data
//! directory gets the program's files there too.
//!
//! # Why downloading is interactive
//!
//! The model is about 2 GB. Fetching it without being asked would be rude on a
//! metered connection, and a silent 2 GB download that fails halfway is worse
//! than no download at all. So a missing model prompts, and only proceeds on an
//! explicit yes — which also keeps every non-interactive path (`--gguf-dir`,
//! CI, a pipe) predictable.

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context as _, Result};

/// Quantisation to fetch when the model is missing.
///
/// Q8_0 is within a fraction of a percent of the full-precision model on this
/// task while being less than half the size, so it is the default rather than
/// f16.
pub const DEFAULT_MODEL_FILE: &str = "Confucius4-R2T2-Q8_0.gguf";
pub const DEFAULT_MMPROJ_FILE: &str = "mmproj-Confucius4-R2T2-Q8_0.gguf";

/// HuggingFace repository holding the GGUF weights.
pub const MODEL_REPO: &str = "netease-youdao/Confucius4-R2T2-GGUF";

/// Root of the program's data directory.
pub fn data_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("XDG_DATA_HOME") {
        if !dir.is_empty() {
            return PathBuf::from(dir).join("r2t2");
        }
    }
    home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".local/share/r2t2")
}

/// Where the GGUF pair lives.
pub fn models_dir() -> PathBuf {
    data_dir().join("models")
}

/// Where the web interface keeps uploads and results.
pub fn work_dir() -> PathBuf {
    data_dir().join("work")
}

/// The configuration file.
///
/// One file rather than a directory: the settings are few, they belong to the
/// person running the program rather than to any one stream, and a file is
/// easier to edit by hand or copy between machines than a database would be.
pub fn config_file() -> PathBuf {
    data_dir().join("config.toml")
}

/// `$HOME`, or `None` when it is not set.
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|h| !h.is_empty())
        .map(PathBuf::from)
}

/// Resolve a model directory, downloading into the default location if needed.
///
/// * `explicit` — a `--gguf-dir` the user gave. Used as-is; a missing or
///   incomplete directory is an error rather than a prompt, because an explicit
///   path means the user believes the model is there.
/// * `assume_yes` — skip the prompt (for scripts).
/// * otherwise the default directory is checked, and if it does not hold a
///   usable pair the user is asked before anything is downloaded.
pub fn resolve_models(explicit: Option<&Path>, assume_yes: bool) -> Result<PathBuf> {
    if let Some(dir) = explicit {
        if !dir.is_dir() {
            bail!(
                "model directory not found: {}\n\
                 (given with --gguf-dir; omit it to use {})",
                dir.display(),
                models_dir().display()
            );
        }
        return Ok(dir.to_path_buf());
    }

    let dir = models_dir();
    if has_model_pair(&dir) {
        return Ok(dir);
    }

    // Nothing usable in the default place.
    if !prompt_to_download(&dir, assume_yes)? {
        // Printed rather than executed, and using only tools every platform
        // already has. `curl` ships with macOS, Linux and Windows, whereas the
        // `hf` CLI this used to suggest needs a Python install that the rest
        // of the program deliberately avoids.
        let base = endpoint();
        bail!(
            "no model found in {dir}\n\
             Download it manually with:\n\
             \x20 curl -L --create-dirs -o \"{dir}/{model}\" \\\n\
             \x20     {base}/{MODEL_REPO}/resolve/main/{model}\n\
             \x20 curl -L --create-dirs -o \"{dir}/{mmproj}\" \\\n\
             \x20     {base}/{MODEL_REPO}/resolve/main/{mmproj}\n\
             \n\
             Set HF_ENDPOINT to download from a mirror instead.",
            dir = dir.display(),
            model = DEFAULT_MODEL_FILE,
            mmproj = DEFAULT_MMPROJ_FILE,
        );
    }

    download_models(&dir)?;
    Ok(dir)
}

/// Whether `dir` holds at least one `mmproj*.gguf` and one other `.gguf`.
///
/// Deliberately a light check: the caller is deciding whether to offer a
/// download, and the strict "exactly one of each" rule is enforced later when
/// the files are actually opened.
pub fn has_model_pair(dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    let mut model = false;
    let mut projector = false;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.ends_with(".gguf") {
            continue;
        }
        if name.starts_with("mmproj") {
            projector = true;
        } else {
            model = true;
        }
    }
    model && projector
}

/// Ask before downloading, unless the caller already said yes.
///
/// Returns `false` when there is no terminal to ask on, so an unattended run
/// fails with instructions instead of hanging on a prompt nobody can answer.
fn prompt_to_download(dir: &Path, assume_yes: bool) -> Result<bool> {
    if assume_yes {
        return Ok(true);
    }

    let stdin = std::io::stdin();
    if !stdin.is_terminal() {
        return Ok(false);
    }

    eprintln!(
        "No model found in {}.\n\
         Download {} (about 2.1 GB) from HuggingFace now? [y/N] ",
        dir.display(),
        DEFAULT_MODEL_FILE,
    );
    std::io::stderr().flush().ok();

    let mut answer = String::new();
    if stdin.read_line(&mut answer).is_err() {
        return Ok(false);
    }
    Ok(matches!(
        answer.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

/// Base URL for downloads, honouring `HF_ENDPOINT` for mirrors.
fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim_end_matches('/').to_string())
        .unwrap_or_else(|| "https://huggingface.co".to_string())
}

/// Fetch the default model pair into `dir`.
pub fn download_models(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("could not create {}", dir.display()))?;

    let base = endpoint();
    for file in [DEFAULT_MODEL_FILE, DEFAULT_MMPROJ_FILE] {
        let url = format!("{base}/{MODEL_REPO}/resolve/main/{file}");
        let dest = dir.join(file);
        if dest.is_file() {
            eprintln!("{} already present, skipping", dest.display());
            continue;
        }
        eprintln!("downloading {file} from {base} ...");
        fetch(&url, &dest).with_context(|| {
            format!(
                "could not download {file}\n\
                 If huggingface.co is unreachable, set HF_ENDPOINT to a mirror, e.g.\n\
                 \x20 HF_ENDPOINT=https://hf-mirror.com"
            )
        })?;
    }
    Ok(())
}

/// Stream one URL to one file, reporting progress on stderr.
///
/// Downloads to a temporary name and renames on success, so an interrupted
/// transfer never leaves a truncated file that looks complete.
fn fetch(url: &str, dest: &Path) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .user_agent(concat!("r2t2/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("could not build the HTTP client")?;

    let mut response = client.get(url).send().context("request failed")?;
    if !response.status().is_success() {
        bail!("server returned HTTP {}", response.status());
    }
    let total = response.content_length();

    let tmp = dest.with_extension("part");
    let mut file = std::fs::File::create(&tmp)
        .with_context(|| format!("could not create {}", tmp.display()))?;

    let mut written: u64 = 0;
    let mut last_report = std::time::Instant::now();
    let mut buf = vec![0u8; 256 * 1024];

    loop {
        use std::io::Read as _;
        let n = response.read(&mut buf).context("transfer interrupted")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).context("could not write to disk")?;
        written += n as u64;

        // Report at most a few times a second; a line per 256 KB would be
        // thousands of lines for this file.
        if last_report.elapsed().as_millis() >= 250 {
            match total {
                Some(t) if t > 0 => eprint!(
                    "\r  {:>5.1}%  {:.0} / {:.0} MB",
                    written as f64 * 100.0 / t as f64,
                    written as f64 / 1e6,
                    t as f64 / 1e6
                ),
                _ => eprint!("\r  {:.0} MB", written as f64 / 1e6),
            }
            std::io::stderr().flush().ok();
            last_report = std::time::Instant::now();
        }
    }
    eprintln!();

    file.sync_all().ok();
    drop(file);

    if let Some(t) = total {
        if written != t {
            let _ = std::fs::remove_file(&tmp);
            bail!("incomplete download: got {written} of {t} bytes");
        }
    }

    std::fs::rename(&tmp, dest)
        .with_context(|| format!("could not move the download into {}", dest.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_a_complete_pair() {
        let dir = tempdir("pair");
        std::fs::write(dir.join("a.gguf"), b"x").unwrap();
        std::fs::write(dir.join("mmproj-a.gguf"), b"x").unwrap();
        assert!(has_model_pair(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_a_lone_model() {
        let dir = tempdir("lone");
        std::fs::write(dir.join("a.gguf"), b"x").unwrap();
        assert!(!has_model_pair(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_a_lone_projector() {
        let dir = tempdir("proj");
        std::fs::write(dir.join("mmproj-a.gguf"), b"x").unwrap();
        assert!(!has_model_pair(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ignores_non_gguf_and_directories() {
        let dir = tempdir("mixed");
        std::fs::write(dir.join("README.md"), b"x").unwrap();
        std::fs::create_dir_all(dir.join("sub.gguf")).unwrap();
        assert!(!has_model_pair(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_directory_is_not_a_pair() {
        assert!(!has_model_pair(Path::new("/nonexistent/r2t2-test")));
    }

    #[test]
    fn data_dir_honours_xdg_data_home() {
        // The variable is process-wide, so this asserts the documented rule
        // rather than mutating it: with XDG_DATA_HOME unset the default is
        // under $HOME.
        if std::env::var_os("XDG_DATA_HOME").is_none() {
            if let Some(home) = home_dir() {
                assert_eq!(data_dir(), home.join(".local/share/r2t2"));
            }
        }
    }

    fn tempdir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "r2t2-paths-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}

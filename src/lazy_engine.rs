// Copyright 2026 youyoEulgo
// SPDX-License-Identifier: Apache-2.0

//! The process's one recognition engine, loaded on demand.
//!
//! # Why this is lazy
//!
//! The server does not need the model to start. It needs it to transcribe —
//! which is to say, only once there is something to transcribe. Starting the
//! interface first means a missing or half-downloaded model is reported *in the
//! interface*, with a way to fix it, instead of as an error on a console that
//! someone who double-clicked the program may never read.
//!
//! # Why there is exactly one
//!
//! The weights are several gigabytes and a consumer GPU holds one copy. Every
//! path — file uploads, WebSocket ingest, RTMP captions — shares this engine
//! and takes turns, because a llama.cpp context holds a single sequence and two
//! could not decode at once anyway.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context as _, Result};
use tokio::sync::{Mutex, OnceCell};
use tracing::{info, warn};

use crate::engine::EngineConfig;
use crate::stream::StreamEngine;

/// Tokens produced per decode step. Shared by every path.
pub const MAX_NEW_TOKENS: i32 = 10;

/// The engine, created on first use from settings that may not have been
/// usable at startup.
pub struct LazyEngine {
    /// How to build the engine, and where the model should be.
    config: EngineConfig,
    /// Directory the model is expected in, for the download button.
    models_dir: PathBuf,
    /// Set once the engine exists. The `OnceCell` serialises the load, so two
    /// simultaneous first requests cannot each load several gigabytes.
    cell: OnceCell<Arc<Mutex<StreamEngine>>>,
    /// Why the last *load* failed, when the files were there.
    ///
    /// A missing file is not an error the user needs explained; a file that is
    /// present and cannot be opened is.
    load_error: Mutex<Option<String>>,
}

/// What the interface needs to know about the engine's availability.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EngineStatus {
    /// Whether the engine has been loaded and can be used right now.
    pub ready: bool,
    /// Whether the weights are present on disk.
    pub model_present: bool,
    /// Whether the weights can be used.
    ///
    /// Distinct from `ready`, and this is the field the interface should judge
    /// by. The engine loads lazily, so `ready` stays false until something is
    /// actually transcribed — reporting that as a fault would show an error on
    /// a perfectly healthy installation that simply has not been used yet.
    pub usable: bool,
    /// Where the weights are expected, so the interface can show the path.
    pub models_dir: String,
    /// Why the engine could not be loaded, if an attempt was made and failed.
    pub error: Option<String>,
}

impl LazyEngine {
    pub fn new(config: EngineConfig, models_dir: PathBuf) -> Self {
        Self {
            config,
            models_dir,
            cell: OnceCell::new(),
            load_error: Mutex::new(None),
        }
    }

    /// The models directory, for the interface and the download button.
    pub fn models_dir(&self) -> PathBuf {
        self.models_dir.clone()
    }

    /// Whether the weights look usable right now.
    pub fn model_present(&self) -> bool {
        crate::paths::has_model_pair(&self.models_dir)
    }

    /// Whether the engine is already loaded.
    pub fn is_ready(&self) -> bool {
        self.cell.initialized()
    }

    /// The engine, loading it if this is the first successful call.
    pub async fn get(&self) -> Result<Arc<Mutex<StreamEngine>>> {
        let cell = &self.cell;
        let result = cell
            .get_or_try_init(|| async {
                // The weights may have appeared since the last attempt — the
                // download button exists precisely so they can — so this is
                // resolved at load time rather than captured at startup.
                let (model, mmproj) = crate::model::resolve_gguf(&self.models_dir)
                    .context("the model files are not usable")?;

                info!(
                    model = %model.display(),
                    mmproj = %mmproj.display(),
                    "loading model"
                );

                let mut config = self.config.clone();
                config.model = model.display().to_string();
                config.mmproj = mmproj.display().to_string();

                let engine = tokio::task::spawn_blocking(move || {
                    StreamEngine::load(&config, MAX_NEW_TOKENS)
                })
                .await
                .context("the model load task failed")??;

                info!("model ready");
                Ok::<_, anyhow::Error>(Arc::new(Mutex::new(engine)))
            })
            .await;

        match &result {
            Ok(_) => *self.load_error.lock().await = None,
            Err(err) => {
                let text = format!("{err:#}");
                warn!(error = %text, "could not load the model");
                *self.load_error.lock().await = Some(text);
            }
        }
        result.cloned()
    }

    /// What the interface should show.
    ///
    /// Answered from the filesystem rather than from any remembered outcome.
    /// The files are the fact: they can appear while the process runs, and a
    /// cached verdict would go stale the moment a download finished.
    pub async fn status(&self) -> EngineStatus {
        let present = self.model_present();
        EngineStatus {
            ready: self.is_ready(),
            model_present: present,
            usable: present,
            models_dir: self.models_dir.display().to_string(),
            // A missing-file error is stale as soon as the files appear. The
            // next recognition attempt will report a genuinely invalid model;
            // the status endpoint should not keep showing the old failure.
            error: if present {
                None
            } else {
                self.load_error.lock().await.clone()
            },
        }
    }

    }

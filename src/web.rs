//! Web interface: static assets and the HTTP API behind them.
//!
//! # Why jobs rather than one request
//!
//! Transcribing a file takes seconds to minutes, far longer than an HTTP
//! request should stay open. An upload is therefore accepted, given an id, and
//! processed in the background; the browser polls for progress. That also makes
//! the work survive a page reload.
//!
//! # Concurrency
//!
//! One GPU, one llama.cpp context, one KV cache — the same constraint the
//! WebSocket service has. Jobs are queued and run one at a time behind the
//! engine mutex. A queued job reports its position so the interface can say so
//! rather than appearing hung.
//!
//! # Artifacts
//!
//! Audio input yields a transcript and a `.txt`. Video input additionally
//! yields an `.srt`, and a Matroska file with the subtitles muxed in — video
//! and audio are stream-copied, so that step is fast and lossless.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use anyhow::{bail, Context as _, Result};
use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Multipart, Path as AxumPath, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::cli::serve::AppState;
use crate::engine::Engine;
use crate::media;
use crate::subtitle::{self, QualityConfig, SplitConfig};
use crate::vad::{Sensitivity, VadConfig};

// --------------------------------------------------------------------------- //
// jobs
// --------------------------------------------------------------------------- //

/// Where a job has got to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Stage {
    Queued,
    Decoding,
    Transcribing,
    Muxing,
    Done,
    Failed,
}

/// Everything the browser needs to render progress and results.
#[derive(Debug, Clone, Serialize)]
pub struct JobStatus {
    pub id: String,
    pub stage: Stage,
    /// Jobs ahead of this one, when queued.
    pub queued_behind: usize,
    /// Segments decoded so far, for a progress indication.
    pub segments_done: usize,
    /// Total segments, once the detector has run.
    pub segments_total: Option<usize>,
    pub duration_secs: Option<f64>,
    pub error: Option<String>,
    /// Human-readable result text.
    pub transcript: Option<String>,
    /// Downloadable artifacts, by kind.
    pub artifacts: Vec<Artifact>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Artifact {
    /// `txt`, `srt`, or `mkv`.
    pub kind: String,
    /// Filename to offer on download.
    pub filename: String,
    pub bytes: u64,
    /// URL path to fetch it from.
    pub url: String,
}

/// What the caller asked for.
#[derive(Debug, Clone, Deserialize)]
pub struct JobOptions {
    #[serde(default = "default_language")]
    pub language: String,
    #[serde(default)]
    pub auto_language: bool,
    #[serde(default)]
    pub context: String,
    #[serde(default = "default_vad_sensitivity")]
    pub vad_sensitivity: String,
    #[serde(default = "default_min_silence")]
    pub vad_min_silence_ms: u64,
    #[serde(default = "default_min_speech")]
    pub vad_min_speech_ms: u64,
    #[serde(default = "default_max_chars")]
    pub max_chars: usize,
    #[serde(default = "default_max_seconds")]
    pub max_seconds: f64,
    #[serde(default = "default_repeat_threshold")]
    pub repeat_threshold: usize,
    #[serde(default)]
    pub keep_hallucinations: bool,
    /// Produce an MKV with the subtitles muxed in. Video only.
    #[serde(default = "default_true")]
    pub make_mkv: bool,
}

fn default_language() -> String {
    "Chinese".to_string()
}
fn default_vad_sensitivity() -> String {
    "aggressive".to_string()
}
fn default_min_silence() -> u64 {
    300
}
fn default_min_speech() -> u64 {
    120
}
fn default_max_chars() -> usize {
    24
}
fn default_max_seconds() -> f64 {
    8.0
}
fn default_repeat_threshold() -> usize {
    5
}
fn default_true() -> bool {
    true
}

impl JobOptions {
    fn language(&self) -> Option<&str> {
        (!self.auto_language && !self.language.is_empty()).then(|| self.language.as_str())
    }

    fn vad(&self) -> Result<VadConfig> {
        Ok(VadConfig {
            sensitivity: Sensitivity::parse(&self.vad_sensitivity)?,
            min_silence_frames: (self.vad_min_silence_ms / 20).max(1) as usize,
            min_speech_frames: (self.vad_min_speech_ms / 20).max(1) as usize,
            ..Default::default()
        })
    }

    fn split(&self) -> SplitConfig {
        SplitConfig {
            max_chars: self.max_chars,
            max_seconds: self.max_seconds,
            ..Default::default()
        }
    }

    fn quality(&self) -> QualityConfig {
        QualityConfig {
            repeat_threshold: self.repeat_threshold,
            keep_hallucinations: self.keep_hallucinations,
            ..Default::default()
        }
    }
}

/// A job's mutable record.
struct Job {
    status: JobStatus,
    /// Directory holding this job's inputs and artifacts.
    dir: PathBuf,
}

/// Shared job table.
#[derive(Default)]
pub struct Jobs {
    map: StdMutex<HashMap<String, Job>>,
}

impl Jobs {
    fn insert(&self, id: String, job: Job) {
        self.map.lock().unwrap().insert(id, job);
    }

    fn update<F: FnOnce(&mut Job)>(&self, id: &str, f: F) {
        if let Some(job) = self.map.lock().unwrap().get_mut(id) {
            f(job);
        }
    }

    fn get_status(&self, id: &str) -> Option<JobStatus> {
        self.map.lock().unwrap().get(id).map(|j| j.status.clone())
    }

    fn get_dir(&self, id: &str) -> Option<PathBuf> {
        self.map.lock().unwrap().get(id).map(|j| j.dir.clone())
    }

    /// How many jobs are not finished yet, excluding `id` itself.
    fn pending_before(&self, id: &str) -> usize {
        let map = self.map.lock().unwrap();
        let mut created: Vec<(&String, &Job)> = map.iter().collect();
        // Insertion order is not tracked, so fall back to creation time
        // encoded in the id, which is monotonic.
        created.sort_by(|a, b| a.0.cmp(b.0));
        let mut ahead = 0;
        for (key, job) in created {
            if key == id {
                break;
            }
            if !matches!(job.status.stage, Stage::Done | Stage::Failed) {
                ahead += 1;
            }
        }
        ahead
    }
}

/// State shared by the HTTP handlers.
pub struct WebState {
    pub jobs: Arc<Jobs>,
    pub work_dir: PathBuf,
    /// Engine settings, used to build the engine on first use.
    engine_config: crate::engine::EngineConfig,
    /// The engine, created on first upload.
    ///
    /// Deferred because loading the model takes about a second and allocates
    /// its share of the GPU: a server started only for live audio should not
    /// pay that until a file is actually uploaded.
    engine: tokio::sync::OnceCell<Mutex<Engine>>,
    pub next_id: AtomicU64,
}

impl WebState {
    pub fn new(work_dir: PathBuf, engine_config: crate::engine::EngineConfig) -> Self {
        Self {
            jobs: Arc::new(Jobs::default()),
            work_dir,
            engine_config,
            engine: tokio::sync::OnceCell::new(),
            next_id: AtomicU64::new(1),
        }
    }

    /// The engine, loading it if this is the first call.
    async fn engine(&self) -> Result<&Mutex<Engine>> {
        self.engine
            .get_or_try_init(|| async {
                let engine = Engine::load(&self.engine_config)
                    .context("failed to load the model for the web interface")?;
                Ok::<_, anyhow::Error>(Mutex::new(engine))
            })
            .await
    }
}

/// Routes for the interface and its API.
pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/", get(index))
        .route("/_ui/{*path}", get(asset))
        .route("/api/jobs", post(create_job))
        .route("/api/jobs/{id}", get(job_status))
        .route("/api/jobs/{id}/files/{kind}", get(job_file))
        // axum caps request bodies at 2 MB by default, well under a video.
        .layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES))
}

// --------------------------------------------------------------------------- //
// static assets
// --------------------------------------------------------------------------- //

#[derive(rust_embed::RustEmbed)]
#[folder = "ui/dist"]
struct UiAssets;

async fn index() -> Response {
    match UiAssets::get("index.html") {
        Some(a) => html(a.data.into_owned()),
        None => (
            StatusCode::NOT_FOUND,
            "web interface not built: run `bun run build` in ui/",
        )
            .into_response(),
    }
}

async fn asset(AxumPath(path): AxumPath<String>) -> Response {
    match UiAssets::get(&path) {
        Some(a) => {
            let ct = content_type(&path);
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, ct)
                .body(Body::from(a.data.into_owned()))
                .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
        }
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

fn html(body: Vec<u8>) -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
        .body(Body::from(body))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json") => "application/json; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("ico") => "image/x-icon",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("woff2") => "font/woff2",
        _ => "application/octet-stream",
    }
}

// --------------------------------------------------------------------------- //
// handlers
// --------------------------------------------------------------------------- //

/// Largest upload we accept, so a mistake cannot fill the disk.
const MAX_UPLOAD_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Accept a file and its options, return a job id immediately.
async fn create_job(
    State(app): State<Arc<AppState>>,
    mut form: Multipart,
) -> Result<Json<JobStatus>, ApiError> {
    let web = app.web.clone().ok_or_else(|| {
        ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "web interface is not enabled")
    })?;

    let id = format!(
        "{:08x}",
        web.next_id.fetch_add(1, Ordering::Relaxed)
    );
    let dir = web.work_dir.join(&id);
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| ApiError::internal(format!("could not create work directory: {e}")))?;

    let mut options = JobOptions {
        language: default_language(),
        auto_language: false,
        context: String::new(),
        vad_sensitivity: default_vad_sensitivity(),
        vad_min_silence_ms: default_min_silence(),
        vad_min_speech_ms: default_min_speech(),
        max_chars: default_max_chars(),
        max_seconds: default_max_seconds(),
        repeat_threshold: default_repeat_threshold(),
        keep_hallucinations: false,
        make_mkv: true,
    };
    let mut saved: Option<(PathBuf, String)> = None;

    while let Some(field) = form.next_field().await.map_err(|e| {
        ApiError::new(StatusCode::BAD_REQUEST, format!("malformed upload: {e}"))
    })? {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                let filename = field
                    .file_name()
                    .map(sanitize_filename)
                    .unwrap_or_else(|| "input.bin".to_string());
                let data = field.bytes().await.map_err(|e| {
                    ApiError::new(StatusCode::BAD_REQUEST, format!("could not read upload: {e}"))
                })?;
                if data.len() > MAX_UPLOAD_BYTES {
                    bail_upload_too_large()?;
                }
                if data.is_empty() {
                    return Err(ApiError::new(StatusCode::BAD_REQUEST, "uploaded file is empty"));
                }
                let path = dir.join(&filename);
                tokio::fs::write(&path, &data)
                    .await
                    .map_err(|e| ApiError::internal(format!("could not save upload: {e}")))?;
                saved = Some((path, filename));
            }
            "options" => {
                let raw = field.text().await.map_err(|e| {
                    ApiError::new(StatusCode::BAD_REQUEST, format!("bad options: {e}"))
                })?;
                options = serde_json::from_str(&raw).map_err(|e| {
                    ApiError::new(StatusCode::BAD_REQUEST, format!("bad options JSON: {e}"))
                })?;
            }
            _ => {}
        }
    }

    let Some((path, filename)) = saved else {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "no file in the request",
        ));
    };

    let status = JobStatus {
        id: id.clone(),
        stage: Stage::Queued,
        queued_behind: 0,
        segments_done: 0,
        segments_total: None,
        duration_secs: None,
        error: None,
        transcript: None,
        artifacts: Vec::new(),
    };
    web.jobs.insert(
        id.clone(),
        Job {
            status: status.clone(),
            dir: dir.clone(),
        },
    );

    // Run in the background; the browser polls.
    let web2 = web.clone();
    let job_id = id.clone();
    tokio::spawn(async move {
        let outcome = process(&web2, &job_id, &path, &filename, &options).await;
        if let Err(err) = outcome {
            let message = format!("{err:#}");
            web2.jobs.update(&job_id, |job| {
                job.status.stage = Stage::Failed;
                job.status.error = Some(message);
            });
        }
    });

    Ok(Json(status))
}

fn bail_upload_too_large() -> Result<(), ApiError> {
    Err(ApiError::new(
        StatusCode::PAYLOAD_TOO_LARGE,
        format!("upload exceeds the {} GB limit", MAX_UPLOAD_BYTES / 1024 / 1024 / 1024),
    ))
}

fn sanitize_filename(name: &str) -> String {
    let base = Path::new(name)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "input.bin".to_string());
    base.chars()
        .map(|c| if c.is_control() || c == '/' || c == '\\' { '_' } else { c })
        .collect()
}

/// The actual work, run off the request path.
async fn process(
    web: &Arc<WebState>,
    id: &str,
    input: &Path,
    original_name: &str,
    options: &JobOptions,
) -> Result<()> {
    let set = |stage: Stage| {
        web.jobs.update(id, |job| job.status.stage = stage);
    };

    // ---- inspect the input ------------------------------------------------
    let is_video = media::has_video_stream(input).unwrap_or(false);
    let stem = Path::new(original_name)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());

    // ---- decode -----------------------------------------------------------
    set(Stage::Decoding);
    let samples = media::load_media_16k_mono(input)?;
    if samples.is_empty() {
        bail!("no audio could be decoded from {original_name}");
    }
    let duration = samples.len() as f64 / crate::audio::TARGET_SAMPLE_RATE as f64;
    web.jobs.update(id, |job| job.status.duration_secs = Some(duration));

    // ---- segment (cheap, and tells the UI how much work there is) ---------
    let vad = options.vad()?;
    let segments = subtitle::detect_segments(&samples, vad)?;
    let total = segments.len();
    web.jobs.update(id, |job| job.status.segments_total = Some(total));

    // ---- transcribe -------------------------------------------------------
    set(Stage::Transcribing);
    let engine = web.engine().await?;
    let engine = engine.lock().await;
    let mut cues = Vec::new();
    let mut transcript_parts = Vec::new();

    for (i, seg) in segments.iter().enumerate() {
        let (text, new_cues) = transcribe_one(&engine, &samples, seg, options)?;
        transcript_parts.push(text);
        cues.extend(new_cues);
        web.jobs.update(id, |job| job.status.segments_done = i + 1);
    }
    drop(engine);

    let cues = subtitle::merge_adjacent(cues, &options.split());
    let transcript = join_transcript(&transcript_parts);

    // ---- write artifacts --------------------------------------------------
    let srt = subtitle::to_srt(&cues);
    let srt_path = web.jobs.get_dir(id).unwrap_or_else(|| PathBuf::from(".")).join(format!("{stem}.srt"));
    std::fs::write(&srt_path, &srt).context("could not write the subtitle file")?;

    let txt_path = srt_path.with_extension("txt");
    std::fs::write(&txt_path, format!("{transcript}\n")).context("could not write the transcript")?;

    let mut artifacts = vec![
        artifact("txt", &txt_path, format!("{stem}.txt"))?,
        artifact("srt", &srt_path, format!("{stem}.srt"))?,
    ];

    // ---- mux --------------------------------------------------------------
    if is_video && options.make_mkv {
        set(Stage::Muxing);
        let mkv_path = srt_path.with_extension("mkv");
        let lang = options.language().unwrap_or("und");
        media::mux_subtitles(input, &srt_path, &mkv_path, lang)?;
        artifacts.push(artifact("mkv", &mkv_path, format!("{stem}.mkv"))?);
    }

    web.jobs.update(id, |job| {
        job.status.stage = Stage::Done;
        job.status.transcript = Some(transcript);
        job.status.artifacts = artifacts;
    });
    Ok(())
}

/// Transcribe one detected segment, applying the quality guards.
fn transcribe_one(
    engine: &Engine,
    samples: &[f32],
    seg: &crate::vad::Segment,
    options: &JobOptions,
) -> Result<(String, Vec<subtitle::Cue>)> {
    use crate::vad::{FRAME_MS, FRAME_SAMPLES};

    let start = (seg.start_frame as usize * FRAME_SAMPLES).min(samples.len());
    let end = (start + seg.frames as usize * FRAME_SAMPLES).min(samples.len());
    if end <= start {
        return Ok((String::new(), Vec::new()));
    }

    let p = crate::prompt::build(&options.context, options.language(), "");
    let result = engine.transcribe(&samples[start..end], &p, 512)?;
    let (_, text) = crate::stream::parse_asr_output(&result.text, options.language());

    let quality = options.quality();
    let text = crate::quality::fix_repetitions(text.trim(), quality.repeat_threshold);
    if crate::quality::is_degenerate(&text) {
        return Ok((String::new(), Vec::new()));
    }
    if !quality.keep_hallucinations
        && crate::quality::detect_hallucination(
            &text,
            quality.repeat_threshold,
            quality.max_pattern_len,
            quality.tail_check_len,
        )
        .is_some()
    {
        return Ok((String::new(), Vec::new()));
    }

    let fps = 1000.0 / FRAME_MS as f64;
    let t0 = seg.start_frame as f64 / fps;
    let t1 = (seg.start_frame + seg.frames) as f64 / fps;
    let cues = subtitle::split_cue(t0, t1, &text, &options.split());
    Ok((text, cues))
}

/// Glue segment texts into a readable transcript.
///
/// Segments arrive without trailing punctuation when the speaker runs on, so a
/// bare concatenation reads as one long word. A space is inserted between two
/// Latin runs; CJK is left alone, since Chinese does not space between
/// characters.
fn join_transcript(parts: &[String]) -> String {
    let mut out = String::new();
    for part in parts {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if out.is_empty() {
            out.push_str(part);
            continue;
        }
        let prev_latin = out.chars().last().is_some_and(|c| c.is_ascii_alphanumeric());
        let next_latin = part.chars().next().is_some_and(|c| c.is_ascii_alphanumeric());
        if prev_latin && next_latin {
            out.push(' ');
        }
        out.push_str(part);
    }
    out
}

fn artifact(kind: &str, path: &Path, filename: String) -> Result<Artifact> {
    let bytes = std::fs::metadata(path)
        .with_context(|| format!("artifact missing: {}", path.display()))?
        .len();
    Ok(Artifact {
        kind: kind.to_string(),
        filename,
        bytes,
        url: String::new(), // filled in by the caller, which knows the job id
    })
}

async fn job_status(
    State(app): State<Arc<AppState>>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<JobStatus>, ApiError> {
    let web = app.web.clone().ok_or_else(|| ApiError::not_found())?;
    let mut status = web
        .jobs
        .get_status(&id)
        .ok_or_else(|| ApiError::not_found())?;

    if status.stage == Stage::Queued {
        status.queued_behind = web.jobs.pending_before(&id);
    }
    for a in &mut status.artifacts {
        a.url = format!("/api/jobs/{id}/files/{}", a.kind);
    }
    Ok(Json(status))
}

async fn job_file(
    State(app): State<Arc<AppState>>,
    AxumPath((id, kind)): AxumPath<(String, String)>,
) -> Result<Response, ApiError> {
    let web = app.web.clone().ok_or_else(|| ApiError::not_found())?;
    let status = web
        .jobs
        .get_status(&id)
        .ok_or_else(|| ApiError::not_found())?;
    let artifact = status
        .artifacts
        .iter()
        .find(|a| a.kind == kind)
        .ok_or_else(|| ApiError::not_found())?;

    let dir = web.jobs.get_dir(&id).ok_or_else(|| ApiError::not_found())?;
    let path = dir.join(&artifact.filename);
    let data = tokio::fs::read(&path)
        .await
        .map_err(|_| ApiError::not_found())?;

    let ct = content_type(&artifact.filename);
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, ct)
        .header(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{}\"", artifact.filename),
        )
        .body(Body::from(data))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response()))
}

// --------------------------------------------------------------------------- //
// errors
// --------------------------------------------------------------------------- //

/// An error rendered as JSON, so the interface can show it.
pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, message)
    }

    fn not_found() -> Self {
        Self::new(StatusCode::NOT_FOUND, "not found")
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        #[derive(Serialize)]
        struct Body_ {
            error: String,
        }
        (
            self.status,
            Json(Body_ {
                error: self.message,
            }),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitises_path_traversal_in_filenames() {
        assert_eq!(sanitize_filename("../../etc/passwd"), "passwd");
        assert_eq!(sanitize_filename("/abs/path/movie.mp4"), "movie.mp4");
        assert_eq!(sanitize_filename("normal.wav"), "normal.wav");
        assert_eq!(sanitize_filename("a\u{0}b.wav"), "a_b.wav");
    }

    #[test]
    fn joins_latin_segments_with_a_space() {
        let parts = vec!["hello".to_string(), "world".to_string()];
        assert_eq!(join_transcript(&parts), "hello world");
    }

    #[test]
    fn does_not_space_between_cjk() {
        let parts = vec!["你好".to_string(), "世界".to_string()];
        assert_eq!(join_transcript(&parts), "你好世界");
    }

    #[test]
    fn skips_empty_segments_when_joining() {
        let parts = vec![
            "第一句".to_string(),
            String::new(),
            "第二句".to_string(),
        ];
        assert_eq!(join_transcript(&parts), "第一句第二句");
    }

    #[test]
    fn maps_language_names_to_iso_codes() {
        assert_eq!(media::iso639_2_for_test("Chinese"), "chi");
        assert_eq!(media::iso639_2_for_test("english"), "eng");
        assert_eq!(media::iso639_2_for_test("Klingon"), "und");
    }
}

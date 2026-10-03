// Types mirroring the Rust API in src/web.rs, and the calls that use them.

export type Stage =
  | 'queued'
  | 'decoding'
  | 'transcribing'
  | 'muxing'
  | 'done'
  | 'failed'

export interface Artifact {
  kind: 'txt' | 'srt' | 'mkv'
  filename: string
  bytes: number
  url: string
}

export interface JobStatus {
  id: string
  stage: Stage
  queued_behind: number
  segments_done: number
  segments_total: number | null
  duration_secs: number | null
  error: string | null
  transcript: string | null
  artifacts: Artifact[]
}

/** Options sent alongside the upload; field names match the Rust struct. */
export interface JobOptions {
  language: string
  auto_language: boolean
  context: string
  vad_sensitivity: string
  vad_min_silence_ms: number
  vad_min_speech_ms: number
  max_chars: number
  max_seconds: number
  repeat_threshold: number
  keep_hallucinations: boolean
}

export const LANGUAGES = [
  'Chinese',
  'English',
  'Japanese',
  'Korean',
  'French',
  'German',
  'Spanish',
  'Portuguese',
  'Russian',
  'Italian',
  'Arabic',
] as const

export const VAD_SENSITIVITIES = [
  { value: 'quality', label: 'Quality — 干净音频' },
  { value: 'lowbitrate', label: 'LowBitrate — 窄带音频' },
  { value: 'aggressive', label: 'Aggressive — 嘈杂素材（默认）' },
  { value: 'veryaggressive', label: 'VeryAggressive — 最严格' },
] as const

export const defaultOptions = (): JobOptions => ({
  language: 'Chinese',
  auto_language: false,
  context: '',
  vad_sensitivity: 'aggressive',
  vad_min_silence_ms: 300,
  vad_min_speech_ms: 120,
  max_chars: 24,
  max_seconds: 8,
  repeat_threshold: 5,
  keep_hallucinations: false,
})

/** Human-readable progress line for a job. */
export function describe(job: JobStatus): string {
  switch (job.stage) {
    case 'queued':
      return job.queued_behind > 0
        ? `排队中，前面还有 ${job.queued_behind} 个任务`
        : '排队中'
    case 'decoding':
      return '正在解码音频…'
    case 'transcribing': {
      const total = job.segments_total
      return total
        ? `正在转写 ${job.segments_done}/${total} 段…`
        : '正在转写…'
    }
    case 'muxing':
      return '正在封装字幕到 MKV…'
    case 'done':
      return '完成'
    case 'failed':
      return '失败'
  }
}

/** 0..1 progress, or null when the total is not known yet. */
export function progress(job: JobStatus): number | null {
  if (job.stage === 'done') return 1
  if (job.stage !== 'transcribing' || !job.segments_total) return null
  return job.segments_done / job.segments_total
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`
}

async function readError(res: Response): Promise<string> {
  try {
    const body = await res.json()
    if (body && typeof body.error === 'string') return body.error
  } catch {
    // fall through to the status text
  }
  return `${res.status} ${res.statusText}`
}

/** Upload a file and its options; resolves with the created job. */
export async function submit(
  file: File,
  options: JobOptions,
): Promise<JobStatus> {
  const form = new FormData()
  form.append('file', file)
  form.append('options', JSON.stringify(options))

  const res = await fetch('/api/jobs', { method: 'POST', body: form })
  if (!res.ok) throw new Error(await readError(res))
  return (await res.json()) as JobStatus
}

/** Live-stream state, as reported by the ingest. */
export interface LiveStatus {
  /** True when the server was started with RTMP ingest enabled. */
  rtmp_enabled: boolean
  /** Whether subtitles are being produced at all. */
  subtitles_enabled: boolean
  /** True while a publisher is connected. */
  publishing: boolean
  /** The stream key the publisher used. */
  stream_key: string
  /** Port the RTMP listener is on. */
  rtmp_port: number
  /** Most recent caption text. */
  latest: string
}

/** How the caption overlay is drawn. Mirrors the server's configuration. */
export interface CaptionConfig {
  lines: number
  chars: number
  size: number
  color: string
  background: string
  bottom: string
  transparent: boolean
}

export async function captionConfig(): Promise<CaptionConfig> {
  const res = await fetch('/api/live/caption')
  if (!res.ok) throw new Error(await readError(res))
  return (await res.json()) as CaptionConfig
}

export async function saveCaptionConfig(
  cfg: CaptionConfig,
): Promise<CaptionConfig> {
  const res = await fetch('/api/live/caption', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(cfg),
  })
  if (!res.ok) throw new Error(await readError(res))
  return (await res.json()) as CaptionConfig
}

/** Whether the recognition model is available. */
export interface ModelStatus {
  ready: boolean
  model_present: boolean
  models_dir: string
  error: string | null
}

export async function modelStatus(): Promise<ModelStatus> {
  const res = await fetch('/api/model')
  if (!res.ok) throw new Error(await readError(res))
  return (await res.json()) as ModelStatus
}

/** One progress line from the model download. */
export interface DownloadProgress {
  stage: 'downloading' | 'done' | 'failed'
  written?: number
  total?: number | null
  error?: string
}

/**
 * Download the weights, reporting progress as it goes.
 *
 * The response is newline-delimited JSON rather than one object, because the
 * transfer is 2.1 GB and takes minutes: a single response would leave the
 * interface with nothing to show until it finished.
 */
export async function downloadModel(
  onProgress: (p: DownloadProgress) => void,
): Promise<void> {
  const res = await fetch('/api/model/download', { method: 'POST' })
  if (!res.ok) throw new Error(await readError(res))
  if (!res.body) throw new Error('the server sent no progress stream')

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffered = ''

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffered += decoder.decode(value, { stream: true })

    // A chunk may end mid-line, so only complete lines are parsed.
    let newline: number
    while ((newline = buffered.indexOf('\n')) !== -1) {
      const line = buffered.slice(0, newline).trim()
      buffered = buffered.slice(newline + 1)
      if (line) onProgress(JSON.parse(line) as DownloadProgress)
    }
  }
  if (buffered.trim()) onProgress(JSON.parse(buffered.trim()) as DownloadProgress)
}

export async function liveStatus(): Promise<LiveStatus> {
  const res = await fetch('/api/live')
  if (!res.ok) throw new Error(await readError(res))
  return (await res.json()) as LiveStatus
}

/** Combine a video and a subtitle file into one MKV. */
export async function mux(
  video: File,
  subtitle: File,
  language: string,
): Promise<JobStatus> {
  const form = new FormData()
  form.append('video', video)
  form.append('subtitle', subtitle)
  form.append('language', language)

  const res = await fetch('/api/mux', { method: 'POST', body: form })
  if (!res.ok) throw new Error(await readError(res))
  return (await res.json()) as JobStatus
}

export async function poll(id: string): Promise<JobStatus> {
  const res = await fetch(`/api/jobs/${id}`)
  if (!res.ok) throw new Error(await readError(res))
  return (await res.json()) as JobStatus
}

/**
 * Poll until the job finishes.
 *
 * Stops when `signal` aborts, so navigating away does not leave a timer
 * running. The interval is short because a finished job should feel immediate.
 */
export async function watch(
  id: string,
  onUpdate: (job: JobStatus) => void,
  signal: AbortSignal,
): Promise<JobStatus> {
  let delay = 400
  for (;;) {
    const job = await poll(id)
    onUpdate(job)
    if (job.stage === 'done' || job.stage === 'failed') return job
    await new Promise((resolve) => setTimeout(resolve, delay))
    if (signal.aborted) return job
    // Back off a little, but stay responsive.
    delay = Math.min(delay + 200, 1500)
  }
}

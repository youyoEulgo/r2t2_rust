// The caption overlay.
//
// Deliberately not a Vue app: OBS loads this as a browser source and it only
// ever displays text, so a framework would add weight and startup time for
// nothing. It connects to /ws/subtitles, keeps the last few lines on screen,
// and reconnects on its own if the server restarts.
//
// Appearance comes from the server's configuration file, both at load and as
// changes arrive over the same connection, so editing it in the console takes
// effect in OBS without touching the URL.

// Type only: erased at build time, so hls.js is not pulled into the bundle
// until `startVideo` actually asks for it.
import type Hls from 'hls.js'

import './live.css'

interface CaptionConfig {
  lines: number
  chars: number
  size: number
  color: string
  background: string
  bottom: string
  transparent: boolean
}

type Message =
  | {
      type: 'subtitle'
      text: string
      delta: string
      reset: boolean
      at_ms: number
    }
  | ({ type: 'caption' } & CaptionConfig)
  | { type: 'status'; enabled: boolean; active: boolean; latest: string }

/** Upper bound on rendered lines, matching the server's clamp. */
const MAX_SLOTS = 5

const root = document.getElementById('caption') as HTMLElement
const stage = document.getElementById('stage') as HTMLElement | null
const video = document.getElementById('video') as HTMLVideoElement | null

/**
 * Whether the video path is on.
 *
 * The server decides: if `/hls/stream.m3u8` is there, there is a picture to
 * show. The caption layer works either way, which is what makes it usable as a
 * browser source in OBS where only the text is wanted.
 */
let hls: Hls | null = null

async function startVideo() {
  if (!video) return
  const src = '/hls/stream.m3u8'

  // Only pull in hls.js when there is a picture to play. It is by far the
  // largest part of this page, and the common use — an OBS browser source
  // showing captions only — never needs it.
  const hasVideo = await fetch(src, { method: 'HEAD' })
    .then((r) => r.ok)
    .catch(() => false)
  if (!hasVideo) {
    stage?.classList.add('no-video')
    return
  }

  const { default: HlsPlayer } = await import('hls.js')

  if (HlsPlayer.isSupported()) {
    hls = new HlsPlayer({
      // Deliberately conservative: a live stream that keeps up matters more
      // than one that starts a second sooner, and a small buffer causes
      // constant stalling on an unstable source.
      liveDurationInfinity: true,
      // Start playback near the live edge.
      liveSyncDurationCount: 2,
      // Give up on a segment rather than waiting forever; a partly buffered
      // stream should recover on its own.
      fragLoadingMaxRetry: 6,
      manifestLoadingMaxRetry: 4,
    })
    hls.loadSource(src)
    hls.attachMedia(video)
    hls.on(HlsPlayer.Events.ERROR, (_e, data) => {
      // A missing playlist just means the video path is off, or no publisher
      // has connected yet. Retry quietly; anything else is worth a line.
      if (data.fatal) {
        setTimeout(() => hls?.startLoad(), 2000)
      }
      if (data.details === 'manifestLoadError') {
        stage?.classList.add('no-video')
      }
    })
    hls.on(HlsPlayer.Events.MANIFEST_PARSED, () => {
      stage?.classList.remove('no-video')
      video.play().catch(() => {
        // Autoplay may be blocked until the page is interacted with; OBS does
        // not block it, a plain browser tab might.
      })
    })
  } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
    // Safari plays HLS by itself.
    video.src = src
  }
}

/**
 * One element per possible line, created once and only ever re-texted.
 *
 * Rebuilding the DOM on every update is what made the captions flicker: the
 * browser re-lays-out, and any entry animation replays. Reusing nodes avoids
 * both.
 */
const slots: HTMLElement[] = []
for (let i = 0; i < MAX_SLOTS; i++) {
  const div = document.createElement('div')
  div.className = 'caption-line'
  div.hidden = true
  root.appendChild(div)
  slots.push(div)
}

let config: CaptionConfig = {
  lines: 2,
  chars: 20,
  size: 48,
  color: '#ffffff',
  background: 'rgba(0, 0, 0, 0.62)',
  bottom: '6%',
  transparent: false,
}

/** Finished lines, oldest first. Bounded by `config.lines - 1`. */
const history: string[] = []
/** The line being spoken. */
let current = ''
/** Text of each slot as last drawn, to avoid redundant DOM writes. */
const drawn: string[] = []

let ws: WebSocket | null = null
let retry = 0

/**
 * Width of a string in full-width units.
 *
 * Two Latin letters count as one CJK character, so a line of mixed text looks
 * the same length either way. Counting raw characters does not do that, and a
 * line of English ends up visibly longer than a line of Chinese.
 */
function width(s: string): number {
  let w = 0
  for (const ch of s) {
    const code = ch.codePointAt(0) ?? 0
    // ASCII and Latin-1 are half width; everything else, including CJK and
    // full-width punctuation, counts as one.
    w += code <= 0xff ? 0.5 : 1
  }
  return w
}

/** Characters that may end a line, preferred over cutting mid-phrase. */
const BREAKS = '，。！？、；：,.!?;:）)」』】》'

/** A Latin letter or digit: never split a word made of these. */
const WORD = /[A-Za-z0-9'\u2019-]/

/**
 * Choose where to split off `current`, given it has reached the limit.
 *
 * Three preferences, in order: break after punctuation if there is any near
 * the limit, else break at a space rather than inside a word, else cut at the
 * limit. All three searches are bounded so a line is not left far shorter than
 * it could be.
 */
function breakAt(text: string, limit: number): number {
  let w = 0
  let hardCut = text.length
  for (let i = 0; i < text.length; i++) {
    w += width(text[i])
    if (w > limit) {
      hardCut = i
      break
    }
  }
  if (hardCut === 0) return 1

  // 1. Punctuation reads as a natural end, so it beats an exact fit.
  const punctuationFloor = Math.max(0, hardCut - 6);
  for (let i = hardCut - 1; i >= punctuationFloor; i--) {
    if (BREAKS.includes(text[i])) return i + 1
  }

  // 2. Do not cut a Latin word in half. The search reaches back to the start of
  //    the word rather than a fixed distance, or a long word would be split
  //    anyway; the cost is a shorter line, which reads better than a broken
  //    word.
  if (WORD.test(text[hardCut - 1]) && WORD.test(text[hardCut] ?? '')) {
    let start = hardCut
    while (start > 0 && WORD.test(text[start - 1])) start--
    // Only worth doing if some of the word fits; otherwise the line would be
    // nearly empty and the word would simply overflow.
    if (start > 0 && width(text.slice(0, start)) >= limit / 3) return start
  }

  return hardCut
}

/** Move text that no longer fits out of `current` and into `history`. */
function reflow() {
  const keep = Math.max(1, config.lines) - 1

  while (width(current) > config.chars) {
    const cut = breakAt(current, config.chars)
    const head = current.slice(0, cut)
    const tail = current.slice(cut)
    // Guard against a split that makes no progress, which would loop forever.
    if (head.length === 0) break
    history.push(head)
    current = tail
  }

  while (history.length > keep) history.shift()
}

/** Draw the visible lines into their slots. */
function render() {
  const visible = [...history, current].filter((l) => l.length > 0)
  const shown = visible.slice(-config.lines)

  for (let i = 0; i < MAX_SLOTS; i++) {
    const slot = slots[i]
    const text = shown[i] ?? ''
    if (drawn[i] !== text) {
      slot.textContent = text
      drawn[i] = text
    }
    slot.hidden = text.length === 0
    const isCurrent = i === shown.length - 1 && text.length > 0
    slot.classList.toggle('caption-current', isCurrent)
  }
}

/** Apply the appearance from the configuration. */
function applyConfig(next: CaptionConfig) {
  config = next
  const s = root.style
  s.setProperty('--caption-size', `${config.size}px`)
  s.setProperty('--caption-color', config.color)
  s.setProperty(
    '--caption-bg',
    config.transparent ? 'transparent' : config.background,
  )
  s.setProperty('--caption-bottom', config.bottom)
  s.setProperty(
    '--caption-shadow',
    config.transparent
      ? '0 2px 6px rgba(0,0,0,0.95)'
      : '0 1px 3px rgba(0,0,0,0.8)',
  )
  // A narrower limit can retroactively push text out of the current line.
  reflow()
  render()
}

function showNotice(text: string) {
  drawn.fill('')
  slots[0].textContent = text
  drawn[0] = text
  slots[0].hidden = false
  slots[0].classList.add('caption-notice')
  for (let i = 1; i < MAX_SLOTS; i++) slots[i].hidden = true
}

function onMessage(raw: string) {
  let msg: Message
  try {
    msg = JSON.parse(raw)
  } catch {
    return
  }

  if (msg.type === 'status') {
    if (msg.latest) {
      current = msg.latest
      reflow()
      render()
    }
    if (msg.enabled === false) {
      showNotice('直播字幕未启用（服务端以 --no-subtitles 启动）')
    }
    return
  }

  if (msg.type === 'caption') {
    applyConfig(msg)
    return
  }

  if (msg.type !== 'subtitle') return

  if (msg.reset && current) {
    history.push(current)
    current = ''
  }
  // `text` is authoritative when present; a partial update may carry only a
  // delta.
  if (typeof msg.text === 'string' && msg.text.length > 0) {
    current = msg.text
  } else if (msg.delta) {
    current += msg.delta
  }
  reflow()
  render()
}

async function loadConfig() {
  try {
    const res = await fetch('/api/live/caption')
    if (res.ok) applyConfig((await res.json()) as CaptionConfig)
  } catch {
    // Keep the defaults; the WebSocket will deliver any change later.
  }
}

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  ws = new WebSocket(`${proto}://${location.host}/ws/subtitles`)

  ws.onopen = () => {
    retry = 0
  }
  ws.onmessage = (e) => onMessage(String(e.data))
  ws.onclose = () => {
    retry = Math.min(retry + 1, 10)
    setTimeout(connect, 500 * retry)
  }
  ws.onerror = () => ws?.close()
}

render()
void loadConfig()
connect()
void startVideo()

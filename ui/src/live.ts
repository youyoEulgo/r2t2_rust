// The caption overlay.
//
// Deliberately not a Vue app: OBS loads this as a browser source and it only
// ever displays text, so a framework would add weight and startup time for
// nothing. It connects to /ws/subtitles, keeps the last few lines on screen,
// and reconnects on its own if the server restarts.

import './live.css'

interface Update {
  type: 'status' | 'subtitle'
  enabled?: boolean
  active?: boolean
  latest?: string
  text?: string
  delta?: string
  reset?: boolean
  at_ms?: number
}

/** How many recent lines to keep visible. */
const MAX_LINES = 2

const root = document.getElementById('caption') as HTMLElement

const lines: string[] = []
let current = ''
let ws: WebSocket | null = null
let retry = 0

/** Read a setting from the query string, so OBS can tweak the look per source. */
function param(name: string, fallback: string): string {
  const v = new URLSearchParams(location.search).get(name)
  return v === null || v === '' ? fallback : v
}

function applyStyle() {
  const s = root.style
  s.setProperty('--caption-size', `${param('size', '48')}px`)
  s.setProperty('--caption-color', param('color', '#ffffff'))
  s.setProperty('--caption-bg', param('bg', 'rgba(0,0,0,0.62)'))
  s.setProperty('--caption-bottom', `${param('bottom', '6')}%`)
  s.setProperty('--caption-lines', String(MAX_LINES))
  // A fully transparent background suits overlaying onto video in OBS.
  if (param('transparent', '0') === '1') {
    s.setProperty('--caption-bg', 'transparent')
    s.setProperty('--caption-shadow', '0 2px 6px rgba(0,0,0,0.95)')
  }
}

function render() {
  const visible = [...lines.slice(-(MAX_LINES - 1)), current].filter(Boolean)
  root.innerHTML = ''
  for (const line of visible) {
    const div = document.createElement('div')
    div.className = 'caption-line'
    div.textContent = line
    root.appendChild(div)
  }
  // Keep the newest line visually emphasised, like broadcast captions do.
  const last = root.lastElementChild
  if (last) last.classList.add('caption-current')
}

function onMessage(raw: string) {
  let msg: Update
  try {
    msg = JSON.parse(raw)
  } catch {
    return
  }

  if (msg.type === 'status') {
    // A late viewer gets whatever has already been said.
    if (msg.latest) {
      current = msg.latest
      render()
    }
    if (msg.enabled === false) {
      showNotice('直播字幕未启用（服务端以 --no-subtitles 启动）')
    }
    return
  }

  if (msg.type !== 'subtitle') return

  if (msg.reset && current) {
    lines.push(current)
    while (lines.length > MAX_LINES) lines.shift()
    current = ''
  }
  if (msg.text !== undefined) {
    current = msg.text
  } else if (msg.delta) {
    current += msg.delta
  }
  render()
}

function showNotice(text: string) {
  root.innerHTML = ''
  const div = document.createElement('div')
  div.className = 'caption-line caption-notice'
  div.textContent = text
  root.appendChild(div)
}

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  ws = new WebSocket(`${proto}://${location.host}/ws/subtitles`)

  ws.onopen = () => {
    retry = 0
  }

  ws.onmessage = (e) => onMessage(String(e.data))

  ws.onclose = () => {
    // Back off, but keep trying: the server may be restarting.
    retry = Math.min(retry + 1, 10)
    setTimeout(connect, 500 * retry)
  }

  ws.onerror = () => ws?.close()
}

applyStyle()
render()
connect()

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import {
  LANGUAGES,
  liveStatus,
  modelStatus,
  downloadModel,
  cancelModelDownload,
  captionConfig,
  saveCaptionConfig,
  VAD_SENSITIVITIES,
  defaultOptions,
  describe,
  formatBytes,
  mux,
  progress,
  submit,
  watch as watchJob,
  type Artifact,
  type JobOptions,
  type JobStatus,
  type LiveStatus,
  type CaptionConfig,
  type ModelStatus,
  type DownloadProgress,
} from './api'

/** Which panel is open. Transcribing and muxing are independent jobs. */
const tab = ref<'transcribe' | 'mux' | 'live'>('transcribe')

// ---- model availability ----
//
// The server starts without the weights, so every panel has to be able to say
// that recognition is unavailable and offer to fix it.
const model = ref<ModelStatus | null>(null)
const install = reactive({
  running: false,
  written: 0,
  total: 0,
  error: null as string | null,
  cancelled: false,
})

/** Ask the server to stop the transfer in progress. */
async function stopInstall() {
  try {
    await cancelModelDownload()
  } catch (e) {
    install.error = e instanceof Error ? e.message : String(e)
  }
}

const installPercent = computed(() => {
  if (!install.total) return 0
  return Math.min(100, Math.round((install.written / install.total) * 100))
})

async function refreshModel() {
  try {
    model.value = await modelStatus()
  } catch {
    // The status endpoint is best-effort; a failure here should not blank the
    // interface, and the next poll will try again.
  }
}

async function runInstall() {
  install.running = true
  install.written = 0
  install.total = 0
  install.error = null
  install.cancelled = false
  try {
    await downloadModel((p: DownloadProgress) => {
      if (p.stage === 'downloading') {
        install.written = p.written ?? 0
        install.total = p.total ?? 0
      } else if (p.stage === 'failed') {
        install.error = p.error ?? '下载失败'
      } else if (p.stage === 'cancelled') {
        install.cancelled = true
      }
    })
  } catch (e) {
    install.error = e instanceof Error ? e.message : String(e)
  } finally {
    install.running = false
    await refreshModel()
  }
}

onMounted(refreshModel)

// ---- live panel state ----
const live = ref<LiveStatus | null>(null)
const liveError = ref<string | null>(null)
let liveTimer: number | undefined

/** Poll the live status while the panel is open. */
function refreshLive() {
  liveStatus()
    .then((s) => {
      live.value = s
      liveError.value = null
    })
    .catch((e) => {
      liveError.value = e instanceof Error ? e.message : String(e)
    })
}

watch(tab, (t) => {
  if (liveTimer !== undefined) {
    clearInterval(liveTimer)
    liveTimer = undefined
  }
  if (t === 'live') {
    refreshLive()
    void loadCaption()
    liveTimer = window.setInterval(refreshLive, 2000)
  }
})

onBeforeUnmount(() => {
  if (liveTimer !== undefined) clearInterval(liveTimer)
  if (savedTimer !== undefined) clearTimeout(savedTimer)
})

/** The RTMP URL to paste into OBS, built from where the page is served. */
const rtmpUrl = computed(() => {
  const host = location.hostname || 'localhost'
  const port = live.value?.rtmp_port ?? 1935
  return `rtmp://${host}:${port}/live`
})

/** The caption overlay URL, for an OBS browser source. */
const captionUrl = computed(() => `${location.origin}/live`)

// ---- caption appearance ----
const caption = ref<CaptionConfig | null>(null)
const captionSaved = ref(false)
let savedTimer: number | undefined

async function loadCaption() {
  try {
    caption.value = await captionConfig()
  } catch (e) {
    liveError.value = e instanceof Error ? e.message : String(e)
  }
}

/** Persist the caption settings; the overlay updates over its own socket. */
async function saveCaption() {
  if (!caption.value) return
  try {
    caption.value = await saveCaptionConfig({ ...caption.value })
    captionSaved.value = true
    if (savedTimer !== undefined) clearTimeout(savedTimer)
    savedTimer = window.setTimeout(() => (captionSaved.value = false), 1500)
  } catch (e) {
    liveError.value = e instanceof Error ? e.message : String(e)
  }
}

// ---- mux panel state ----
const muxVideo = ref<File | null>(null)
const muxSubtitle = ref<File | null>(null)
/**
 * Language tag written into the subtitle track's metadata.
 *
 * Players show this name in their track menu and use it to auto-select by
 * system language. Defaults to Chinese, since that is what this tool is
 * overwhelmingly used for; anyone subtitling something else can say so.
 */
const muxLanguage = ref('Chinese')
const muxJob = ref<JobStatus | null>(null)
const muxError = ref<string | null>(null)
const muxBusy = ref(false)

const canMux = computed(() => !!muxVideo.value && !!muxSubtitle.value && !muxBusy.value)

function pickMuxVideo(f: File | null) {
  if (f) muxVideo.value = f
}
function pickMuxSubtitle(f: File | null) {
  if (f) muxSubtitle.value = f
}

async function runMux() {
  if (!muxVideo.value || !muxSubtitle.value) return
  muxBusy.value = true
  muxError.value = null
  muxJob.value = null
  try {
    muxJob.value = await mux(muxVideo.value, muxSubtitle.value, muxLanguage.value)
  } catch (e) {
    muxError.value = e instanceof Error ? e.message : String(e)
  } finally {
    muxBusy.value = false
  }
}

function resetMux() {
  muxVideo.value = null
  muxSubtitle.value = null
  muxJob.value = null
  muxError.value = null
}

const options = reactive<JobOptions>(defaultOptions())
const file = ref<File | null>(null)
const job = ref<JobStatus | null>(null)
const error = ref<string | null>(null)
const busy = ref(false)
const dragging = ref(false)
const showAdvanced = ref(false)

let controller: AbortController | null = null

const isVideo = computed(() =>
  file.value ? /\.(mp4|mkv|mov|avi|webm|flv|m4v|ts)$/i.test(file.value.name) : false,
)
const canSubmit = computed(() => file.value !== null && !busy.value)
const progressValue = computed(() => (job.value ? progress(job.value) : null))

function pick(f: File | null) {
  if (!f) return
  file.value = f
  error.value = null
  job.value = null
}

function onDrop(e: DragEvent) {
  dragging.value = false
  const f = e.dataTransfer?.files?.[0]
  if (f) pick(f)
}

function onFileInput(e: Event) {
  const input = e.target as HTMLInputElement
  pick(input.files?.[0] ?? null)
}

async function run() {
  if (!file.value) return
  busy.value = true
  error.value = null
  job.value = null
  controller?.abort()
  controller = new AbortController()

  try {
    const created = await submit(file.value, { ...options })
    job.value = created
    await watchJob(
      created.id,
      (j) => {
        job.value = j
      },
      controller.signal,
    )
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = false
  }
}

function reset() {
  controller?.abort()
  controller = null
  file.value = null
  job.value = null
  error.value = null
  busy.value = false
}

/** Copy text, with a brief acknowledgement on the button. */
function copy(text: string) {
  void navigator.clipboard.writeText(text)
}

function copyTranscript() {
  const text = job.value?.transcript
  if (text) void navigator.clipboard.writeText(text)
}

const artifactLabel: Record<Artifact['kind'], string> = {
  txt: '纯文本',
  srt: '字幕 (SRT)',
  mkv: '内嵌字幕视频 (MKV)',
}

onBeforeUnmount(() => controller?.abort())
</script>

<template>
  <div class="wrap">
    <header>
      <h1>r2t2 <span class="sub">语音转写</span></h1>
      <p class="lead">
        识别模型为
        <a href="https://huggingface.co/netease-youdao/Confucius4-R2T2" target="_blank" rel="noreferrer">
          Confucius4-R2T2</a>，由网易有道开源；推理基于
        <a href="https://github.com/ggml-org/llama.cpp" target="_blank" rel="noreferrer">llama.cpp</a>。
        感谢两个团队的开源工作。所有计算在本机完成。
      </p>
    </header>

    <!-- The model is loaded on first use, so this is the one thing that has to
         be visible before anything else is attempted. -->
    <section v-if="model && !model.model_present" class="panel notice">
      <h3>尚未安装识别模型</h3>
      <p class="hint">
        服务已启动，但还没有下载权重文件。识别功能要等模型就绪后才能使用。
      </p>
      <p class="hint">
        将保存到 <code>{{ model.models_dir }}</code>，约 2.1 GB。
      </p>

      <template v-if="install.running">
        <div class="bar"><div class="bar-fill" :style="{ width: installPercent + '%' }"></div></div>
        <p class="hint">
          正在下载 {{ installPercent }}%
          <span v-if="install.total">
            （{{ formatBytes(install.written) }} / {{ formatBytes(install.total) }}）
          </span>
        </p>
        <div class="actions">
          <button @click="stopInstall">取消下载</button>
        </div>
        <p v-if="install.written > 0" class="hint">
          取消会删除已下载的部分，下次从零开始。已下载
          {{ formatBytes(install.written) }}。
        </p>
      </template>
      <template v-else>
        <div class="actions">
          <button class="primary" @click="runInstall">
            {{ install.cancelled ? '重新下载模型' : '下载模型' }}
          </button>
        </div>
        <p v-if="install.cancelled" class="hint">下载已取消。</p>
      </template>

      <p v-if="install.error" class="error">{{ install.error }}</p>
      <p class="hint">
        也可以自行下载后放入上述目录，文件名必须是
        <code>Confucius4-R2T2-Q8_0.gguf</code> 与
        <code>mmproj-Confucius4-R2T2-Q8_0.gguf</code>。
      </p>
    </section>

    <section v-else-if="model && !model.ready" class="panel notice">
      <h3>模型未能加载</h3>
      <p class="error">{{ model.error }}</p>
      <p class="hint">
        文件位于 <code>{{ model.models_dir }}</code>。修正后重新开始识别即可，
        服务无需重启。
      </p>
    </section>

    <nav class="tabs">
      <button :class="{ active: tab === 'transcribe' }" @click="tab = 'transcribe'">
        转写
      </button>
      <button :class="{ active: tab === 'mux' }" @click="tab = 'mux'">
        合并字幕
      </button>
      <button :class="{ active: tab === 'live' }" @click="tab = 'live'">
        直播字幕
      </button>
    </nav>

    <!-- ---------- transcribe ---------- -->
    <section v-show="tab === 'transcribe'" class="panel">
      <div
        class="drop"
        :class="{ dragging, has: !!file }"
        @dragover.prevent="dragging = true"
        @dragleave.prevent="dragging = false"
        @drop.prevent="onDrop"
      >
        <input
          id="file"
          type="file"
          accept="audio/*,video/*,.wav,.mp3,.flac,.m4a,.mp4,.mkv,.mov,.avi,.webm"
          @change="onFileInput"
        />
        <label for="file" class="drop-label">
          <template v-if="file">
            <strong>{{ file.name }}</strong>
            <span class="muted">{{ formatBytes(file.size) }} · 点击更换</span>
          </template>
          <template v-else>
            <strong>选择文件</strong>
            <span class="muted">或拖放到这里 · 音频、视频均可，输出 TXT 与 SRT</span>
          </template>
        </label>
      </div>

      <!-- ---------- options ---------- -->
      <div class="grid">
        <div>
          <label for="lang">语言</label>
          <select id="lang" v-model="options.language" :disabled="options.auto_language">
            <option v-for="l in LANGUAGES" :key="l" :value="l">{{ l }}</option>
          </select>
        </div>
        <div class="check-row">
          <input id="auto" type="checkbox" v-model="options.auto_language" />
          <label for="auto">自动识别语言</label>
        </div>
        <div class="span-2">
          <label for="ctx">热词 / 上下文</label>
          <input
            id="ctx"
            type="text"
            v-model="options.context"
            placeholder="例如：WSLC WSL 会话容器 微软"
          />
          <p class="hint">
            拼写固定的术语填在这里能显著减少错误，例如把「绘画容器」纠正为「会话容器」。
          </p>
        </div>
      </div>

      <button class="link" @click="showAdvanced = !showAdvanced">
        {{ showAdvanced ? '收起' : '展开' }}高级设置
      </button>

      <div v-if="showAdvanced" class="grid advanced">
        <div>
          <label for="vad">VAD 灵敏度</label>
          <select id="vad" v-model="options.vad_sensitivity">
            <option v-for="v in VAD_SENSITIVITIES" :key="v.value" :value="v.value">
              {{ v.label }}
            </option>
          </select>
        </div>
        <div>
          <label for="silence">断句静音 (ms)</label>
          <input id="silence" type="number" min="20" step="20" v-model.number="options.vad_min_silence_ms" />
        </div>
        <div>
          <label for="speech">起始语音 (ms)</label>
          <input id="speech" type="number" min="20" step="20" v-model.number="options.vad_min_speech_ms" />
        </div>
        <div>
          <label for="chars">单条字幕最大字数</label>
          <input id="chars" type="number" min="4" v-model.number="options.max_chars" />
        </div>
        <div>
          <label for="secs">单条字幕最大时长 (s)</label>
          <input id="secs" type="number" min="1" step="0.5" v-model.number="options.max_seconds" />
        </div>
        <div>
          <label for="repeat">重复判定阈值</label>
          <input id="repeat" type="number" min="2" v-model.number="options.repeat_threshold" />
        </div>
        <div class="check-row">
          <input id="halluc" type="checkbox" v-model="options.keep_hallucinations" />
          <label for="halluc">保留疑似幻觉片段</label>
        </div>
      </div>

      <div class="actions">
        <button class="primary" :disabled="!canSubmit" @click="run">
          {{ busy ? '处理中…' : '开始转写' }}
        </button>
        <button v-if="file || job" :disabled="busy" @click="reset">清空</button>
      </div>

      <p v-if="error" class="error">{{ error }}</p>
    </section>

    <!-- ---------- live ---------- -->
    <template v-if="tab === 'live'">
      <section class="panel">
        <p class="lead tight">
          把 OBS 的推流地址指向这台机器，识别结果会实时叠加成字幕。
          字幕画面本身是一个网页，可以作为浏览器源加入 OBS 场景。
        </p>

        <p v-if="liveError" class="error">{{ liveError }}</p>

        <template v-else-if="live">
          <div class="status">
            <span :class="['dot', live.publishing ? 'done' : 'idle']"></span>
            <span>
              {{ live.publishing ? '正在接收推流' : '等待推流' }}
            </span>
            <span v-if="live.stream_key" class="muted">· 串流密钥 “{{ live.stream_key }}”</span>
          </div>

          <p v-if="!live.rtmp_enabled" class="error">
            服务端未启用 RTMP 接收，请去掉 <code>--no-rtmp</code> 后重启。
          </p>
          <p v-else-if="!live.subtitles_enabled" class="error">
            服务端启用了 <code>--no-subtitles</code>，只接收音频但不产生字幕。
          </p>

          <h3>推流地址</h3>
          <div class="copy-row">
            <code>{{ rtmpUrl }}</code>
            <button @click="copy(rtmpUrl)">复制</button>
          </div>
          <p class="hint">
            OBS → 设置 → 推流 → 服务选「自定义」，服务器填上面的地址，串流密钥留空即可。
          </p>

          <h3>字幕画面</h3>
          <div class="copy-row">
            <code>{{ captionUrl }}</code>
            <button @click="copy(captionUrl)">复制</button>
            <a :href="captionUrl" target="_blank" rel="noreferrer">
              <button>打开预览</button>
            </a>
          </div>
          <p class="hint">
            在 OBS 里添加「浏览器」源并填入该地址，即可把字幕叠加到画面中。
            可用查询参数调整外观，例如
            <code>?size=64&amp;transparent=1</code>。
          </p>

          <h3>字幕外观</h3>
          <div v-if="caption" class="grid advanced">
            <div>
              <label for="cap-lines">显示行数</label>
              <input id="cap-lines" type="number" min="1" max="5" v-model.number="caption.lines" />
            </div>
            <div>
              <label for="cap-chars">每行字数</label>
              <input id="cap-chars" type="number" min="4" max="120" v-model.number="caption.chars" />
              <p class="hint tight">两个英文字母算一个字。</p>
            </div>
            <div>
              <label for="cap-size">字号 (px)</label>
              <input id="cap-size" type="number" min="8" max="200" v-model.number="caption.size" />
            </div>
            <div>
              <label for="cap-color">文字颜色</label>
              <input id="cap-color" type="color" v-model="caption.color" />
            </div>
            <div>
              <label for="cap-bottom">距底部</label>
              <input id="cap-bottom" type="text" v-model="caption.bottom" />
            </div>
            <div class="check-row">
              <input id="cap-transparent" type="checkbox" v-model="caption.transparent" />
              <label for="cap-transparent">不显示背景条</label>
            </div>
          </div>
          <p v-if="caption && !caption.transparent" class="hint">
            背景色：<code>{{ caption.background }}</code>
          </p>

          <div class="actions">
            <button class="primary" @click="saveCaption">保存外观</button>
            <span v-if="captionSaved" class="saved">已保存</span>
          </div>
          <p class="hint">
            保存在
            <code>~/.local/share/r2t2/config.toml</code>。
            已打开的字幕画面会立即应用，无需刷新 OBS。
          </p>

          <h3>当前字幕</h3>
          <pre class="transcript">{{ live.latest || '（尚未收到语音）' }}</pre>
        </template>
      </section>
    </template>

    <!-- ---------- mux ---------- -->
    <template v-if="tab === 'mux'">
      <section class="panel">
        <p class="lead tight">
          把已有的字幕文件内嵌进视频。视频与音频直接复制，不重新编码，
          因此无论多长都只需数秒。
        </p>

        <div class="grid">
          <div>
            <label>视频文件</label>
            <label class="file-slot" :class="{ has: !!muxVideo }">
              <input
                type="file"
                accept="video/*,.mp4,.mkv,.mov,.avi,.webm"
                @change="pickMuxVideo(($event.target as HTMLInputElement).files?.[0] ?? null)"
              />
              <span v-if="muxVideo">{{ muxVideo.name }}</span>
              <span v-else class="muted">选择视频</span>
            </label>
          </div>
          <div>
            <label>字幕文件 (SRT)</label>
            <label class="file-slot" :class="{ has: !!muxSubtitle }">
              <input
                type="file"
                accept=".srt,text/plain"
                @change="pickMuxSubtitle(($event.target as HTMLInputElement).files?.[0] ?? null)"
              />
              <span v-if="muxSubtitle">{{ muxSubtitle.name }}</span>
              <span v-else class="muted">选择 SRT</span>
            </label>
          </div>
          <div>
            <label for="muxlang">字幕语言</label>
            <select id="muxlang" v-model="muxLanguage">
              <option v-for="l in LANGUAGES" :key="l" :value="l">{{ l }}</option>
            </select>
            <p class="hint">写入字幕轨的元数据，供播放器显示轨道名称与自动选轨。</p>
          </div>
        </div>

        <div class="actions">
          <button class="primary" :disabled="!canMux" @click="runMux">
            {{ muxBusy ? '合并中…' : '合并为 MKV' }}
          </button>
          <button v-if="muxVideo || muxJob" :disabled="muxBusy" @click="resetMux">清空</button>
        </div>

        <p v-if="muxError" class="error">{{ muxError }}</p>

        <template v-if="muxJob && muxJob.stage === 'done'">
          <h3>下载</h3>
          <ul class="artifacts">
            <li v-for="a in muxJob.artifacts" :key="a.kind">
              <a :href="a.url" :download="a.filename">
                <span class="kind">{{ artifactLabel[a.kind] ?? a.kind }}</span>
                <span class="name">{{ a.filename }}</span>
                <span class="muted">{{ formatBytes(a.bytes) }}</span>
              </a>
            </li>
          </ul>
        </template>
        <p v-if="muxJob && muxJob.error" class="error">{{ muxJob.error }}</p>
      </section>
    </template>

    <!-- ---------- progress ---------- -->
    <section v-if="tab === 'transcribe' && job" class="panel">
      <div class="status">
        <span :class="['dot', job.stage]"></span>
        <span>{{ describe(job) }}</span>
        <span v-if="job.duration_secs" class="muted">· {{ job.duration_secs.toFixed(1) }}s 音频</span>
      </div>

      <div v-if="progressValue !== null" class="bar">
        <div class="fill" :style="{ width: `${Math.round(progressValue * 100)}%` }"></div>
      </div>

      <p v-if="job.error" class="error">{{ job.error }}</p>

      <!-- ---------- results ---------- -->
      <template v-if="job.stage === 'done'">
        <div class="result-head">
          <h2>结果</h2>
          <button @click="copyTranscript">复制文本</button>
        </div>
        <pre class="transcript">{{ job.transcript }}</pre>

        <h3>下载</h3>
        <ul class="artifacts">
          <li v-for="a in job.artifacts" :key="a.kind">
            <a :href="a.url" :download="a.filename">
              <span class="kind">{{ artifactLabel[a.kind] ?? a.kind }}</span>
              <span class="name">{{ a.filename }}</span>
              <span class="muted">{{ formatBytes(a.bytes) }}</span>
            </a>
          </li>
        </ul>
      </template>
    </section>

    <footer>
      <p class="muted">
        转写输出 TXT 与 SRT，字幕时间来自语音活动检测，因此不会重叠。
        需要成片时用「合并字幕」把 SRT 内嵌进视频。
      </p>
    </footer>
  </div>
</template>

<style scoped>
.wrap { max-width: 880px; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }

header h1 { margin: 0 0 0.25rem; font-size: 1.6rem; letter-spacing: -0.01em; }
header h1 .sub { color: var(--muted); font-weight: 400; font-size: 1.1rem; }
.lead { margin: 0 0 2rem; color: var(--muted); max-width: 66ch; }
.lead a { color: var(--accent); text-decoration: none; }
.lead a:hover { text-decoration: underline; }

/* ---------- tabs ---------- */
.tabs { display: flex; gap: 0.25rem; margin-bottom: 0.75rem; }
.tabs button {
  background: none; border: none; border-bottom: 2px solid transparent;
  border-radius: 0; padding: 0.5rem 0.9rem; color: var(--muted);
}
.tabs button:hover { background: none; color: var(--text); }
.tabs button.active { color: var(--text); border-bottom-color: var(--accent); }

.lead.tight { margin-bottom: 1rem; font-size: 0.9rem; }

/* A file input styled as a slot, for the mux panel. */
.file-slot {
  position: relative; display: block; margin: 0; cursor: pointer;
  border: 1.5px dashed var(--line); border-radius: 8px;
  padding: 0.75rem; text-align: center; font-size: 0.9rem;
  color: var(--text); transition: border-color 0.15s;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.file-slot:hover { border-color: var(--accent); }
.file-slot.has { border-style: solid; border-color: var(--accent-dim); }
.file-slot input[type="file"] { position: absolute; width: 1px; height: 1px; opacity: 0; }

.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 1.25rem;
  margin-bottom: 1.25rem;
}

/* ---------- drop zone ---------- */
.drop {
  position: relative;
  border: 1.5px dashed var(--line);
  border-radius: 10px;
  transition: border-color 0.15s, background 0.15s;
}
.drop.dragging { border-color: var(--accent); background: rgba(79, 140, 255, 0.06); }
.drop.has { border-style: solid; border-color: var(--accent-dim); }
.drop input[type="file"] { position: absolute; width: 1px; height: 1px; opacity: 0; }
.drop-label {
  display: flex; flex-direction: column; gap: 0.2rem;
  align-items: center; justify-content: center;
  padding: 2rem 1rem; margin: 0; cursor: pointer; text-align: center;
}

/* ---------- fields ---------- */
.grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
  margin-top: 1.25rem;
}
.grid.advanced { padding-top: 0.25rem; }
.span-2 { grid-column: span 2; }

.check-row { display: flex; align-items: center; gap: 0.5rem; padding-top: 1.35rem; }
.check-row input { width: auto; }
.check-row label { margin: 0; color: var(--text); font-size: 0.9rem; }

.hint { margin: 0.35rem 0 0; font-size: 0.8rem; color: var(--muted); }
.muted { color: var(--muted); }

button.link {
  margin-top: 1rem; padding: 0; background: none; border: none;
  color: var(--accent); font-size: 0.9rem;
}
button.link:hover { background: none; text-decoration: underline; }

.actions { display: flex; gap: 0.75rem; margin-top: 1.5rem; }

/* ---------- status ---------- */
.status { display: flex; align-items: center; gap: 0.6rem; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); flex: none; }
.dot.transcribing, .dot.decoding, .dot.muxing { background: var(--accent); animation: pulse 1.2s infinite; }
.dot.done { background: var(--ok); }
.dot.idle { background: var(--muted); }
.dot.failed { background: var(--err); }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }

.bar { height: 4px; background: var(--panel-2); border-radius: 2px; margin-top: 0.85rem; overflow: hidden; }
.fill { height: 100%; background: var(--accent); transition: width 0.3s ease; }

/* ---------- results ---------- */
.result-head { display: flex; align-items: center; justify-content: space-between; margin-top: 1.5rem; }
.result-head h2 { margin: 0; font-size: 1.05rem; }
h3 { font-size: 0.9rem; color: var(--muted); font-weight: 600; margin: 1.5rem 0 0.5rem; }

.transcript {
  background: var(--bg); border: 1px solid var(--line); border-radius: 8px;
  padding: 1rem; margin: 0.75rem 0 0; white-space: pre-wrap; word-break: break-word;
  font-family: inherit; font-size: 0.95rem; line-height: 1.7; max-height: 26rem; overflow: auto;
}

.artifacts { list-style: none; padding: 0; margin: 0; }
.artifacts li + li { margin-top: 0.4rem; }
.artifacts a {
  display: flex; align-items: center; gap: 0.75rem;
  padding: 0.6rem 0.8rem; border: 1px solid var(--line); border-radius: 8px;
  color: inherit; text-decoration: none; transition: border-color 0.15s, background 0.15s;
}
.artifacts a:hover { border-color: var(--accent); background: var(--panel-2); }
.artifacts .kind { color: var(--accent); min-width: 9rem; font-size: 0.9rem; }
.artifacts .name { flex: 1; font-size: 0.9rem; }

.error { color: var(--err); margin: 0.75rem 0 0; font-size: 0.9rem; }

.hint.tight { margin: 0.2rem 0 0; font-size: 0.78rem; }
.saved { color: var(--ok); font-size: 0.85rem; }

/* A panel that reports a condition rather than offering a form. */
.notice { border-color: var(--accent); }
.notice h3 { margin-top: 0; }

.bar {
  height: 8px; border-radius: 4px; overflow: hidden;
  background: var(--bg); border: 1px solid var(--line); margin: 0.75rem 0 0.5rem;
}
.bar-fill { height: 100%; background: var(--accent); transition: width 200ms ease-out; }

/* A path or URL with its own copy button. */
.copy-row {
  display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;
  margin-top: 0.5rem;
}
.copy-row code {
  flex: 1; min-width: 16rem;
  background: var(--bg); border: 1px solid var(--line); border-radius: 6px;
  padding: 0.45rem 0.6rem; font-size: 0.9rem;
  overflow-x: auto; white-space: nowrap;
}
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }

footer { margin-top: 2.5rem; font-size: 0.85rem; }

@media (max-width: 620px) {
  .grid { grid-template-columns: 1fr; }
  .span-2 { grid-column: span 1; }
  .artifacts .kind { min-width: auto; }
}
</style>

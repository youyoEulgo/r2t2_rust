<script setup lang="ts">
import { computed, onBeforeUnmount, reactive, ref } from 'vue'
import {
  LANGUAGES,
  VAD_SENSITIVITIES,
  defaultOptions,
  describe,
  formatBytes,
  mux,
  progress,
  submit,
  watch,
  type Artifact,
  type JobOptions,
  type JobStatus,
} from './api'

/** Which panel is open. Transcribing and muxing are independent jobs. */
const tab = ref<'transcribe' | 'mux'>('transcribe')

// ---- mux panel state ----
const muxVideo = ref<File | null>(null)
const muxSubtitle = ref<File | null>(null)
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
    await watch(
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
        上传音频得到文字，上传视频得到字幕与内嵌字幕的视频。
        推理在本机进行，文件不会离开这台机器。
      </p>
    </header>

    <nav class="tabs">
      <button :class="{ active: tab === 'transcribe' }" @click="tab = 'transcribe'">
        转写
      </button>
      <button :class="{ active: tab === 'mux' }" @click="tab = 'mux'">
        合并字幕
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
            <span class="muted">或拖放到这里 · 音频输出 TXT，视频输出 SRT 与 MKV</span>
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
            <label for="muxlang">字幕语言标记</label>
            <select id="muxlang" v-model="muxLanguage">
              <option v-for="l in LANGUAGES" :key="l" :value="l">{{ l }}</option>
            </select>
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
.lead { margin: 0 0 2rem; color: var(--muted); max-width: 62ch; }

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

footer { margin-top: 2.5rem; font-size: 0.85rem; }

@media (max-width: 620px) {
  .grid { grid-template-columns: 1fr; }
  .span-2 { grid-column: span 1; }
  .artifacts .kind { min-width: auto; }
}
</style>

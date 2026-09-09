<script setup>
import { onMounted, ref } from 'vue'
import { CheckCircle2, Database, FilePlus2, FolderOpen, LoaderCircle, RefreshCw, UploadCloud } from 'lucide-vue-next'
import { api, delay } from '../api'

const emit = defineEmits(['updated'])
const knowledgeBase = ref('/app/KnowledgeBase')
const indexedChunks = ref(0)
const category = ref('未分类')
const selectedFile = ref(null)
const fileInput = ref(null)
const uploading = ref(false)
const indexing = ref(false)
const progress = ref(0)
const statusText = ref('就绪')
const errorText = ref('')
const lastResult = ref(null)
let activeTask = ''

async function loadSettings() {
  try {
    const settings = await api.settings()
    knowledgeBase.value = settings.knowledge_base
    indexedChunks.value = settings.indexed_chunks || 0
  } catch (error) {
    errorText.value = error.message
  }
}

function chooseFile(event) {
  selectedFile.value = event.target.files?.[0] || null
  errorText.value = ''
}

async function upload() {
  if (!selectedFile.value || uploading.value) return
  uploading.value = true
  errorText.value = ''
  statusText.value = `正在上传 ${selectedFile.value.name}…`
  try {
    const result = await api.uploadDocument(selectedFile.value, category.value.trim() || '未分类', knowledgeBase.value)
    knowledgeBase.value = result.knowledge_base
    statusText.value = `${result.file} 已上传，准备更新索引`
    selectedFile.value = null
    if (fileInput.value) fileInput.value.value = ''
    await buildIndex()
  } catch (error) {
    errorText.value = error.message
    statusText.value = '上传失败'
  } finally {
    uploading.value = false
  }
}

async function buildIndex() {
  if (indexing.value || !knowledgeBase.value.trim()) return
  indexing.value = true
  progress.value = 0
  lastResult.value = null
  errorText.value = ''
  statusText.value = '正在启动索引任务…'
  try {
    const started = await api.startIndex(knowledgeBase.value.trim())
    activeTask = started.task_id
    while (activeTask === started.task_id) {
      await delay(700)
      const job = await api.indexStatus(started.task_id)
      progress.value = job.percent || 0
      statusText.value = job.detail || '正在处理文档…'
      if (job.status === 'completed') {
        lastResult.value = job.result || {}
        statusText.value = '索引创建完成'
        progress.value = 100
        await loadSettings()
        emit('updated')
        break
      }
      if (job.status === 'failed') throw new Error(job.error || '索引创建失败')
      if (job.status === 'cancelled') {
        statusText.value = '索引任务已取消'
        break
      }
    }
  } catch (error) {
    errorText.value = error.message
    statusText.value = '索引创建失败'
  } finally {
    indexing.value = false
    activeTask = ''
  }
}

onMounted(loadSettings)
</script>

<template>
  <div class="library-view page-stack">
    <section class="page-intro">
      <span class="eyebrow">Knowledge workspace</span>
      <h2>管理本地知识库</h2>
      <p>上传 PDF、Markdown、TXT 或 DOCX，增量构建可检索、可引用的知识空间。</p>
    </section>

    <div class="stats-strip">
      <div><Database :size="20" /><span><strong>{{ indexedChunks.toLocaleString() }}</strong>已索引切片</span></div>
      <div><FolderOpen :size="20" /><span><strong>4</strong>支持的文档格式</span></div>
      <div><CheckCircle2 :size="20" /><span><strong>本地</strong>数据保存方式</span></div>
    </div>

    <div class="library-grid">
      <section class="panel library-card">
        <div class="panel-heading"><div><span class="eyebrow">01 · Library</span><h3>知识库位置</h3></div><Database :size="20" /></div>
        <label class="field-label" for="knowledge-path">Docker 内部挂载路径</label>
        <div class="input-with-icon"><FolderOpen :size="17" /><input id="knowledge-path" v-model="knowledgeBase" /></div>
        <p class="field-help">浏览器不能直接选择服务器文件夹。这里对应宿主机的 `KnowledgeBase` 挂载目录。</p>
        <button class="primary-button wide" :disabled="indexing" @click="buildIndex">
          <LoaderCircle v-if="indexing" class="spin" :size="17" /><RefreshCw v-else :size="17" />
          {{ indexing ? '正在构建索引' : '扫描并创建索引' }}
        </button>
      </section>

      <section class="panel library-card">
        <div class="panel-heading"><div><span class="eyebrow">02 · Upload</span><h3>添加单个文档</h3></div><FilePlus2 :size="20" /></div>
        <label class="upload-zone" for="file-upload">
          <UploadCloud :size="27" />
          <strong>{{ selectedFile?.name || '选择要上传的文档' }}</strong>
          <span>PDF、Markdown、TXT、DOCX · 最大 50 MB</span>
        </label>
        <input id="file-upload" ref="fileInput" class="sr-only" type="file" accept=".pdf,.md,.txt,.docx" @change="chooseFile" />
        <label class="field-label" for="category">分类名称</label>
        <input id="category" v-model="category" placeholder="未分类" />
        <button class="primary-button wide" :disabled="!selectedFile || uploading || indexing" @click="upload">
          <LoaderCircle v-if="uploading" class="spin" :size="17" /><UploadCloud v-else :size="17" />
          {{ uploading ? '正在上传' : '上传并自动更新索引' }}
        </button>
      </section>
    </div>

    <section class="panel progress-card">
      <div class="progress-copy"><div><span class="status-dot" :class="{ busy: indexing || uploading }" /><strong>{{ statusText }}</strong></div><span>{{ progress }}%</span></div>
      <div class="progress-track"><i :style="{ width: `${progress}%` }" /></div>
      <div v-if="lastResult" class="result-grid">
        <span><strong>{{ lastResult.files || 0 }}</strong>文件</span>
        <span><strong>{{ lastResult.chunks || 0 }}</strong>切片</span>
        <span><strong>{{ lastResult.added_or_updated || 0 }}</strong>新增或更新</span>
        <span><strong>{{ lastResult.unchanged || 0 }}</strong>复用</span>
      </div>
      <div v-if="errorText" class="error-banner">{{ errorText }}</div>
    </section>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, ref } from 'vue'
import { ArrowUp, Bot, Check, Copy, FileText, LoaderCircle, RotateCcw, Square, User, X } from 'lucide-vue-next'
import { api, delay } from '../api'

const makeThreadId = () => globalThis.crypto?.randomUUID?.() || `web-${Date.now()}`
const threadId = ref(localStorage.getItem('pka-thread-id') || makeThreadId())
localStorage.setItem('pka-thread-id', threadId.value)

const question = ref('')
const busy = ref(false)
const taskId = ref('')
const statusText = ref('就绪')
const errorText = ref('')
const messages = ref([])
const activeSources = ref([])
const conversation = ref(null)
let alive = true

const hasConversation = computed(() => messages.value.length > 0)

function resolveMedia(items = []) {
  const imageCache = new Map()
  return items.map((item) => {
    const metadata = item.metadata || {}
    const resourceId = String(metadata.resource_id || item.id)
    let encodedImage = item.data_base64 || ''
    if (encodedImage) imageCache.set(resourceId, encodedImage)
    else encodedImage = imageCache.get(resourceId) || ''
    const imageSrc = encodedImage
      ? `data:${item.mime_type || 'image/png'};base64,${encodedImage}`
      : item.source_url || ''
    return { ...item, metadata, imageSrc }
  }).filter((item) => item.imageSrc)
}

function newConversation() {
  if (busy.value) return
  threadId.value = makeThreadId()
  localStorage.setItem('pka-thread-id', threadId.value)
  messages.value = []
  activeSources.value = []
  errorText.value = ''
  statusText.value = '新对话已创建'
}

async function scrollToBottom() {
  await nextTick()
  conversation.value?.scrollTo({ top: conversation.value.scrollHeight, behavior: 'smooth' })
}

async function sendMessage() {
  const text = question.value.trim()
  if (!text || busy.value) return
  errorText.value = ''
  messages.value.push({ role: 'user', content: text })
  question.value = ''
  busy.value = true
  statusText.value = '正在提交问题…'
  activeSources.value = []
  await scrollToBottom()

  try {
    const started = await api.startChat(text, threadId.value)
    taskId.value = started.task_id
    while (alive && taskId.value === started.task_id) {
      await delay(650)
      const job = await api.chatStatus(started.task_id)
      statusText.value = job.detail || 'Agent 正在处理…'
      if (job.status === 'completed') {
        const result = job.result
        messages.value.push({
          role: 'assistant',
          content: result.answer,
          sources: result.sources || [],
          media: resolveMedia(result.media),
          confidence: result.confidence,
          runId: result.run_id,
          requiresReview: result.requires_review,
          reviewReason: result.review_reason,
          reviewed: false,
        })
        activeSources.value = result.sources || []
        statusText.value = `回答完成 · ${result.tool_calls?.length || 0} 次工具调用`
        break
      }
      if (job.status === 'failed') throw new Error(job.error || '回答生成失败')
      if (job.status === 'cancelled') {
        statusText.value = '回答已取消'
        break
      }
    }
  } catch (error) {
    errorText.value = error.message
    statusText.value = '回答失败'
  } finally {
    busy.value = false
    taskId.value = ''
    scrollToBottom()
  }
}

async function cancelChat() {
  if (!taskId.value) return
  try {
    await api.cancelJob(taskId.value)
    statusText.value = '正在取消…'
  } catch (error) {
    errorText.value = error.message
  }
}

async function reviewMessage(message, approved) {
  if (!message.runId || message.reviewed) return
  try {
    await api.resumeRun(message.runId, approved)
    message.reviewed = true
    message.reviewApproved = approved
  } catch (error) {
    errorText.value = error.message
  }
}

function handleKeydown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    sendMessage()
  }
}

async function copyPath(path) {
  await navigator.clipboard.writeText(path)
  statusText.value = '来源路径已复制'
}

function selectSources(message) {
  if (message.sources?.length) activeSources.value = message.sources
}

onBeforeUnmount(() => { alive = false })
</script>

<template>
  <div class="chat-layout">
    <section class="chat-panel panel">
      <div class="panel-heading">
        <div><span class="eyebrow">知识问答</span><h2>对话</h2></div>
        <button class="ghost-button" :disabled="busy" @click="newConversation"><RotateCcw :size="16" />新对话</button>
      </div>

      <div ref="conversation" class="conversation" aria-live="polite">
        <div v-if="!hasConversation" class="empty-chat">
          <div class="empty-symbol"><Bot :size="27" /></div>
          <h3>从你的资料里找到答案</h3>
          <p>Agent 会规划问题、检索知识库并给出带来源的回答。</p>
          <div class="prompt-grid">
            <button @click="question = '概括一下我的知识库中最重要的主题'">概括知识库的主要主题</button>
            <button @click="question = '列出知识库中值得继续研究的问题'">发现值得继续研究的问题</button>
          </div>
        </div>

        <article
          v-for="(message, index) in messages"
          :key="index"
          class="message"
          :class="message.role"
          @click="selectSources(message)"
        >
          <div class="message-avatar"><User v-if="message.role === 'user'" :size="16" /><Bot v-else :size="17" /></div>
          <div class="message-body">
            <div class="message-meta">{{ message.role === 'user' ? '你' : 'Knowledge Agent' }}</div>
            <div class="message-content">{{ message.content }}</div>
            <div v-if="message.media?.length" class="media-grid">
              <figure v-for="media in message.media" :key="media.id" class="glyph-card">
                <div class="glyph-image">
                  <img :src="media.imageSrc" :alt="media.alt || media.title || '回答图片'" />
                </div>
                <figcaption>
                  <strong>{{ media.metadata.character || media.title || '字形' }}</strong>
                  <span>{{ media.metadata.author || '佚名' }} · {{ media.metadata.work || '来源未注明' }}</span>
                </figcaption>
              </figure>
            </div>
            <div v-if="message.sources?.length" class="answer-meta">
              <span>{{ message.sources.length }} 个引用</span>
              <span v-if="Number.isFinite(message.confidence)">置信度 {{ Math.round(message.confidence * 100) }}%</span>
            </div>
            <div v-if="message.requiresReview" class="review-box">
              <p>{{ message.reviewReason || '此回答需要人工确认。' }}</p>
              <div v-if="!message.reviewed">
                <button class="review-accept" @click.stop="reviewMessage(message, true)"><Check :size="15" />接受</button>
                <button @click.stop="reviewMessage(message, false)"><X :size="15" />拒绝</button>
              </div>
              <span v-else>{{ message.reviewApproved ? '已接受回答' : '已拒绝回答' }}</span>
            </div>
          </div>
        </article>

        <div v-if="busy" class="thinking-row"><LoaderCircle class="spin" :size="17" /><span>{{ statusText }}</span></div>
      </div>

      <div v-if="errorText" class="error-banner">{{ errorText }}</div>
      <div class="composer-wrap">
        <textarea
          v-model="question"
          rows="1"
          maxlength="8000"
          placeholder="输入问题，Enter 发送，Shift + Enter 换行…"
          aria-label="输入知识库问题"
          @keydown="handleKeydown"
        />
        <button v-if="busy" class="send-button cancel" aria-label="取消回答" @click="cancelChat"><Square :size="16" /></button>
        <button v-else class="send-button" :disabled="!question.trim()" aria-label="发送问题" @click="sendMessage"><ArrowUp :size="18" /></button>
      </div>
      <div class="composer-foot"><span class="status-dot" :class="{ busy }" />{{ statusText }}<span>Thread {{ threadId.slice(0, 8) }}</span></div>
    </section>

    <aside class="sources-panel panel">
      <div class="panel-heading"><div><span class="eyebrow">Grounding</span><h2>引用来源</h2></div><span class="count-badge">{{ activeSources.length }}</span></div>
      <div v-if="!activeSources.length" class="empty-sources"><FileText :size="25" /><p>回答中使用的文档与页码会显示在这里。</p></div>
      <div v-else class="source-list">
        <article v-for="(source, index) in activeSources" :key="`${source.path}-${source.page}`" class="source-card">
          <div class="source-index">{{ String(index + 1).padStart(2, '0') }}</div>
          <div class="source-detail"><strong>{{ source.file }}</strong><span>第 {{ source.page }} 页</span><p>{{ source.snippet }}</p></div>
          <button class="copy-button" aria-label="复制来源路径" title="复制来源路径" @click="copyPath(source.path)"><Copy :size="14" /></button>
        </article>
      </div>
    </aside>
  </div>
</template>

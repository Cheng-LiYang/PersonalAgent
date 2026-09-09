<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { CheckCircle2, Eye, EyeOff, KeyRound, LoaderCircle, Save, ShieldCheck } from 'lucide-vue-next'
import { api } from '../api'

const providers = [
  { value: 'compatible', label: 'OpenAI 兼容 API', hint: 'DeepSeek 等兼容服务' },
  { value: 'openai', label: 'OpenAI', hint: 'OpenAI 官方 API' },
  { value: 'local', label: '本地兼容服务', hint: 'Ollama、LM Studio 等' },
  { value: 'extractive', label: '离线抽取模式', hint: '无需 API Key' },
]
const form = ref({ provider: 'extractive', model: 'offline-extractive', base_url: 'http://127.0.0.1', api_key: '' })
const hasSavedKey = ref(false)
const showKey = ref(false)
const saving = ref(false)
const statusText = ref('正在读取设置…')
const errorText = ref('')
const activeProvider = computed(() => providers.find((item) => item.value === form.value.provider))

async function loadSettings() {
  try {
    const settings = await api.settings()
    form.value = { ...settings.model, api_key: '' }
    hasSavedKey.value = settings.has_api_key
    statusText.value = `当前：${settings.model.model}`
  } catch (error) {
    errorText.value = error.message
    statusText.value = '读取设置失败'
  }
}

async function saveSettings() {
  if (!form.value.model.trim() || !form.value.base_url.trim()) return
  saving.value = true
  errorText.value = ''
  statusText.value = '正在测试模型连接…'
  try {
    const result = await api.configureModel({
      provider: form.value.provider,
      model: form.value.model.trim(),
      base_url: form.value.base_url.trim(),
      api_key: form.value.api_key,
    })
    statusText.value = `已连接：${result.model}`
    if (form.value.api_key) hasSavedKey.value = true
    form.value.api_key = ''
  } catch (error) {
    errorText.value = error.message
    statusText.value = '模型连接失败'
  } finally {
    saving.value = false
  }
}

watch(() => form.value.provider, (provider, previous) => {
  if (!previous) return
  if (provider === 'extractive') Object.assign(form.value, { model: 'offline-extractive', base_url: 'http://127.0.0.1', api_key: '' })
  if (provider === 'local') Object.assign(form.value, { model: 'qwen2.5:7b', base_url: 'http://host.docker.internal:11434/v1', api_key: '' })
  if (provider === 'openai') Object.assign(form.value, { model: 'gpt-4o-mini', base_url: 'https://api.openai.com/v1' })
  if (provider === 'compatible') Object.assign(form.value, { model: 'deepseek-v4-flash', base_url: 'https://api.deepseek.com' })
})

onMounted(loadSettings)
</script>

<template>
  <div class="settings-view page-stack">
    <section class="page-intro">
      <span class="eyebrow">Model connection</span>
      <h2>连接回答模型</h2>
      <p>选择云端或本地模型。设置会在连接验证通过后保存到 Docker 数据卷。</p>
    </section>

    <section class="panel settings-card">
      <div class="settings-heading"><div class="settings-icon"><KeyRound :size="22" /></div><div><h3>模型服务</h3><p>Agent 会依据检索证据生成回答</p></div></div>

      <div class="provider-grid">
        <button
          v-for="provider in providers"
          :key="provider.value"
          class="provider-option"
          :class="{ active: form.provider === provider.value }"
          @click="form.provider = provider.value"
        >
          <span><strong>{{ provider.label }}</strong><small>{{ provider.hint }}</small></span>
          <i><CheckCircle2 v-if="form.provider === provider.value" :size="17" /></i>
        </button>
      </div>

      <form class="settings-form" @submit.prevent="saveSettings">
        <label><span>模型名称</span><input v-model="form.model" required autocomplete="off" /></label>
        <label><span>API Base URL</span><input v-model="form.base_url" required type="url" /></label>
        <label v-if="form.provider !== 'extractive' && form.provider !== 'local'">
          <span>API Key <small v-if="hasSavedKey">已安全保存，留空可继续使用</small></span>
          <div class="key-field"><input v-model="form.api_key" :type="showKey ? 'text' : 'password'" :placeholder="hasSavedKey ? '••••••••••••••••' : '输入 API Key'" autocomplete="off" /><button type="button" :aria-label="showKey ? '隐藏 API Key' : '显示 API Key'" @click="showKey = !showKey"><EyeOff v-if="showKey" :size="17" /><Eye v-else :size="17" /></button></div>
        </label>

        <div class="privacy-note"><ShieldCheck :size="19" /><div><strong>隐私说明</strong><p>Docker/Linux 环境使用本地编码保存密钥，运行 Trace 会自动脱敏。请保护好宿主机的 data 目录。</p></div></div>
        <div v-if="errorText" class="error-banner">{{ errorText }}</div>
        <div class="settings-actions"><span class="model-state"><i />{{ statusText }}</span><button class="primary-button" :disabled="saving"><LoaderCircle v-if="saving" class="spin" :size="17" /><Save v-else :size="17" />{{ saving ? '正在连接' : '应用模型设置' }}</button></div>
      </form>
    </section>

    <section class="soft-note"><strong>{{ activeProvider?.label }}</strong><span>{{ activeProvider?.hint }} · 配置只作用于当前 Personal Knowledge Agent 实例</span></section>
  </div>
</template>

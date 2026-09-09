<script setup>
import { computed, onMounted, ref } from 'vue'
import { Menu, Moon, Sun } from 'lucide-vue-next'
import { api } from './api'
import AppSidebar from './components/AppSidebar.vue'
import ChatView from './components/ChatView.vue'
import LibraryView from './components/LibraryView.vue'
import SettingsView from './components/SettingsView.vue'

const activeView = ref('chat')
const sidebarOpen = ref(false)
const serviceOnline = ref(false)
const indexedChunks = ref(0)
const storedTheme = localStorage.getItem('pka-theme')
const darkMode = ref(storedTheme ? storedTheme === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches)

const titles = {
  chat: ['智能对话', '和你的知识一起思考'],
  library: ['知识库', '管理你的本地资料'],
  settings: ['模型设置', '连接回答模型'],
}
const currentTitle = computed(() => titles[activeView.value])

function applyTheme() {
  document.documentElement.dataset.theme = darkMode.value ? 'dark' : 'light'
  localStorage.setItem('pka-theme', darkMode.value ? 'dark' : 'light')
}

function toggleTheme() {
  darkMode.value = !darkMode.value
  applyTheme()
}

function navigate(view) {
  activeView.value = view
  sidebarOpen.value = false
}

async function refreshStatus() {
  try {
    const [, settings] = await Promise.all([api.health(), api.settings()])
    serviceOnline.value = true
    indexedChunks.value = settings.indexed_chunks || 0
  } catch {
    serviceOnline.value = false
  }
}

onMounted(() => {
  applyTheme()
  refreshStatus()
})
</script>

<template>
  <div class="app-shell">
    <AppSidebar
      :active-view="activeView"
      :open="sidebarOpen"
      :online="serviceOnline"
      :indexed-chunks="indexedChunks"
      @navigate="navigate"
      @close="sidebarOpen = false"
    />

    <main class="main-shell">
      <header class="topbar">
        <button class="icon-button mobile-menu" aria-label="打开导航" @click="sidebarOpen = true">
          <Menu :size="20" />
        </button>
        <div>
          <p class="topbar-kicker">{{ currentTitle[0] }}</p>
          <h1>{{ currentTitle[1] }}</h1>
        </div>
        <div class="topbar-actions">
          <span class="service-state" :class="{ online: serviceOnline }">
            <i />{{ serviceOnline ? '服务已连接' : '服务未连接' }}
          </span>
          <button class="icon-button" :aria-label="darkMode ? '切换浅色模式' : '切换深色模式'" @click="toggleTheme">
            <Sun v-if="darkMode" :size="18" />
            <Moon v-else :size="18" />
          </button>
        </div>
      </header>

      <section class="workspace">
        <ChatView v-if="activeView === 'chat'" />
        <LibraryView v-else-if="activeView === 'library'" @updated="refreshStatus" />
        <SettingsView v-else />
      </section>
    </main>
  </div>
</template>

<script setup>
import { Bot, Database, MessageSquareText, Settings2, X } from 'lucide-vue-next'

defineProps({
  activeView: { type: String, required: true },
  open: { type: Boolean, default: false },
  online: { type: Boolean, default: false },
  indexedChunks: { type: Number, default: 0 },
})
defineEmits(['navigate', 'close'])

const navigation = [
  { id: 'chat', label: '智能对话', caption: '检索与问答', icon: MessageSquareText },
  { id: 'library', label: '知识库', caption: '文档与索引', icon: Database },
  { id: 'settings', label: '模型设置', caption: '连接与隐私', icon: Settings2 },
]
</script>

<template>
  <div v-if="open" class="sidebar-backdrop" @click="$emit('close')" />
  <aside class="sidebar" :class="{ open }">
    <div class="brand-row">
      <div class="brand-mark"><Bot :size="22" /></div>
      <div>
        <strong>Knowledge Agent</strong>
        <span>Personal workspace</span>
      </div>
      <button class="icon-button sidebar-close" aria-label="关闭导航" @click="$emit('close')"><X :size="18" /></button>
    </div>

    <nav aria-label="主导航">
      <button
        v-for="item in navigation"
        :key="item.id"
        class="nav-item"
        :class="{ active: activeView === item.id }"
        @click="$emit('navigate', item.id)"
      >
        <component :is="item.icon" :size="19" />
        <span><strong>{{ item.label }}</strong><small>{{ item.caption }}</small></span>
      </button>
    </nav>

    <div class="sidebar-summary">
      <div class="summary-label"><span>知识状态</span><i :class="{ online }" /></div>
      <strong>{{ indexedChunks.toLocaleString() }}</strong>
      <p>个切片已建立索引</p>
    </div>
    <p class="sidebar-foot">Local-first · Docker runtime</p>
  </aside>
</template>

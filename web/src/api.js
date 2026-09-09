const API_BASE = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '')

async function apiFetch(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options)
  const contentType = response.headers.get('content-type') || ''
  const payload = contentType.includes('application/json') ? await response.json() : await response.text()
  if (!response.ok) {
    const detail = payload?.detail ?? payload?.error ?? payload
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return payload
}

function jsonOptions(method, body) {
  return {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }
}

export const api = {
  health: () => apiFetch('/health'),
  settings: () => apiFetch('/settings'),
  startChat: (question, threadId) =>
    apiFetch('/chat/start', jsonOptions('POST', { question, thread_id: threadId })),
  chatStatus: (taskId) => apiFetch(`/chat/status/${encodeURIComponent(taskId)}`),
  cancelJob: (taskId) => apiFetch(`/jobs/${encodeURIComponent(taskId)}/cancel`, { method: 'POST' }),
  startIndex: (path) => apiFetch('/index/build/start', jsonOptions('POST', { path })),
  indexStatus: (taskId) => apiFetch(`/index/status/${encodeURIComponent(taskId)}`),
  uploadDocument: (file, category, knowledgeBase) => {
    const params = new URLSearchParams({ category })
    if (knowledgeBase) params.set('knowledge_base', knowledgeBase)
    const form = new FormData()
    form.append('file', file)
    return apiFetch(`/document/upload?${params}`, { method: 'POST', body: form })
  },
  configureModel: (payload) => apiFetch('/settings/model', jsonOptions('POST', payload)),
  resumeRun: (runId, approved, feedback = '') =>
    apiFetch(`/runs/${encodeURIComponent(runId)}/resume`, jsonOptions('POST', { approved, feedback })),
}

export const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds))

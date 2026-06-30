<template>
  <div class="app-container">
    <!-- Sidebar -->
    <aside class="sidebar">
      <div class="sidebar-header">
        <span class="logo-icon">🚀</span>
        <h2>LabRAG Panel</h2>
      </div>

      <div class="info-section">
        <h3>Información del Sistema</h3>
        <div class="info-card">
          <div class="info-row">
            <span class="info-label">Estado:</span>
            <span :class="['status-badge', backendStatus.ready ? 'online' : 'offline']">
              {{ backendStatus.ready ? 'Listo' : (backendStatus.connected ? 'Cargando RAG' : 'Desconectado') }}
            </span>
          </div>
          <div class="info-row">
            <span class="info-label">LLM:</span>
            <span class="info-value">Llama-3.1 8B</span>
          </div>
          <div class="info-row">
            <span class="info-label">Embeddings:</span>
            <span class="info-value">all-MiniLM-L6-v2</span>
          </div>
          <div class="info-row">
            <span class="info-label">Vector Store:</span>
            <span class="info-value">Qdrant</span>
          </div>
        </div>
      </div>

      <div class="info-section">
        <h3>Documentos Cargados</h3>
        <div class="doc-item">
          <span class="doc-icon">📄</span>
          <div class="doc-details">
            <span class="doc-name">{{ documentName }}</span>
            <span class="doc-size">Base de Conocimiento RAG</span>
          </div>
        </div>
      </div>

      <div class="info-section suggestions-section">
        <h3>Preguntas sugeridas</h3>
        <button 
          v-for="sug in suggestions" 
          :key="sug" 
          @click="selectSuggestion(sug)"
          :disabled="loading || !backendStatus.ready"
          class="sug-btn"
        >
          {{ sug }}
        </button>
      </div>

      <div class="sidebar-footer">
        <p>Desarrollado con FastAPI, Vue 3 & LangChain</p>
      </div>
    </aside>

    <!-- Main Chat Area -->
    <main class="chat-area">
      <!-- Chat Header -->
      <header class="chat-header">
        <div class="header-main">
          <h1>Asistente Inteligente de Documentación</h1>
          <p>Realiza preguntas sobre el documento indexado y obtén respuestas precisas con referencias.</p>
        </div>
        <div class="connection-status">
          <span :class="['pulse-dot', backendStatus.ready ? 'green' : (backendStatus.connected ? 'yellow' : 'red')]"></span>
          <span class="connection-text">{{ backendStatus.text }}</span>
        </div>
      </header>

      <!-- Message History -->
      <section class="messages-container" ref="messageContainer">
        <!-- Welcome Screen if no messages -->
        <div v-if="messages.length === 0" class="welcome-screen">
          <div class="welcome-icon">🤖</div>
          <h2>¡Bienvenido al Chat RAG!</h2>
          <p>Estoy listo para analizar la documentación y responder tus preguntas basándome estrictamente en los hechos descritos en el documento.</p>
          <div class="start-hints">
            <div class="hint-card">
              <h4>🔍 Recuperación Vectorial</h4>
              <p>Busca en Qdrant los fragmentos de texto más similares a tu pregunta.</p>
            </div>
            <div class="hint-card">
              <h4>🤖 Generación Aumentada</h4>
              <p>Usa Llama-3.1 en Groq para resumir y responder con precisión en español.</p>
            </div>
            <div class="hint-card">
              <h4>📄 Fuentes Transparentes</h4>
              <p>Muestra exactamente de qué página y qué fragmento proviene cada respuesta.</p>
            </div>
          </div>
        </div>

        <!-- Render Messages -->
        <div 
          v-for="(msg, index) in messages" 
          :key="index" 
          :class="['message-row', msg.sender]"
        >
          <div class="message-bubble-wrapper">
            <div class="avatar">
              {{ msg.sender === 'user' ? '👤' : '🤖' }}
            </div>
            <div class="message-content-wrapper">
              <div class="sender-name">{{ msg.sender === 'user' ? 'Tú' : 'Asistente RAG' }}</div>
              <div class="message-bubble">
                <div v-if="msg.sender === 'rag'" v-html="formatResponse(msg.text)"></div>
                <div v-else>{{ msg.text }}</div>
              </div>

              <!-- Sources (only for RAG responses with sources) -->
              <div v-if="msg.sender === 'rag' && msg.sources && msg.sources.length > 0" class="sources-wrapper">
                <button @click="msg.showSources = !msg.showSources" class="toggle-sources-btn">
                  <span>{{ msg.showSources ? '▼ Ocultar Fuentes' : '▶ Ver Fuentes (' + msg.sources.length + ')' }}</span>
                </button>
                <div v-if="msg.showSources" class="sources-list animate-slide-down">
                  <div v-for="(source, sIdx) in msg.sources" :key="sIdx" class="source-card">
                    <div class="source-card-header">
                      <span class="source-pdf">📄 {{ source.archivo }}</span>
                      <span class="source-page">Pág. {{ source.pagina }}</span>
                    </div>
                    <p class="source-snippet">"{{ source.fragmento }}"</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Loader / Thinking State -->
        <div v-if="loading" class="message-row rag loading-row">
          <div class="message-bubble-wrapper">
            <div class="avatar pulse">🤖</div>
            <div class="message-content-wrapper">
              <div class="sender-name">Asistente RAG</div>
              <div class="message-bubble thinking-bubble">
                <div class="typing-loader">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                <span class="thinking-text">Buscando en la documentación y generando respuesta...</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- Chat Input Area -->
      <footer class="chat-input-area">
        <form @submit.prevent="sendMessage" class="input-form">
          <input 
            type="text" 
            v-model="inputMessage" 
            placeholder="Escribe tu pregunta sobre la documentación..." 
            :disabled="loading || !backendStatus.ready"
            ref="textInput"
            required
          />
          <button 
            type="submit" 
            :disabled="loading || !backendStatus.ready || !inputMessage.trim()" 
            class="send-btn"
          >
            <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
          </button>
        </form>
      </footer>
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import axios from 'axios'

const API_BASE_URL = `http://${window.location.hostname}:8000`

const messages = ref([])
const inputMessage = ref('')
const loading = ref(false)
const messageContainer = ref(null)
const textInput = ref(null)

const suggestions = ref([
  '¿De qué trata este documento?',
  'Resume los puntos clave de la documentación',
  '¿Cuáles son las características principales?',
  '¿Hay algún flujo técnico detallado?'
])

const backendStatus = ref({
  connected: false,
  ready: false,
  text: 'Comprobando conexión...'
})

const documentName = ref('Cargando...')

// Check backend health status
const checkHealth = async () => {
  try {
    const res = await axios.get(`${API_BASE_URL}/api/health`, { timeout: 3000 })
    backendStatus.value.connected = true
    if (res.data) {
      documentName.value = res.data.document || 'documentacion.pdf'
      if (res.data.rag_ready) {
        backendStatus.value.ready = true
        backendStatus.value.text = 'Conectado (RAG Listo)'
      } else {
        backendStatus.value.ready = false
        backendStatus.value.text = 'Inicializando modelo...'
      }
    }
  } catch (error) {
    backendStatus.value.connected = false
    backendStatus.value.ready = false
    backendStatus.value.text = 'Backend desconectado'
  }
}

onMounted(() => {
  checkHealth()
  // Poll health status every 5 seconds
  setInterval(checkHealth, 5000)
  
  // Focus on input
  if (textInput.value) {
    textInput.value.focus()
  }
})

const scrollToBottom = async () => {
  await nextTick()
  if (messageContainer.value) {
    messageContainer.value.scrollTop = messageContainer.value.scrollHeight
  }
}

const selectSuggestion = (sug) => {
  inputMessage.value = sug
  sendMessage()
}

const sendMessage = async () => {
  const query = inputMessage.value.trim()
  if (!query || loading.value || !backendStatus.value.ready) return

  // Add User message
  messages.value.push({
    sender: 'user',
    text: query
  })
  
  inputMessage.value = ''
  loading.value = true
  await scrollToBottom()

  try {
    const response = await axios.post(`${API_BASE_URL}/api/preguntar`, {
      pregunta: query
    })

    // Add RAG response
    messages.value.push({
      sender: 'rag',
      text: response.data.respuesta,
      sources: response.data.fuentes || [],
      showSources: false
    })
  } catch (error) {
    let errorMsg = 'Lo siento, ha ocurrido un error al conectarse con el servidor.'
    if (error.response && error.response.data && error.response.data.detail) {
      errorMsg = `Error: ${error.response.data.detail}`
    }
    messages.value.push({
      sender: 'rag',
      text: errorMsg,
      sources: []
    })
  } finally {
    loading.value = false
    await scrollToBottom()
    // Refocus input
    nextTick(() => {
      if (textInput.value) textInput.value.focus()
    })
  }
}

// Simple HTML formatter for responses to handle bullet points and paragraphs nicely
const formatResponse = (text) => {
  if (!text) return ''
  
  // Clean markdown style line breaks
  let lines = text.split('\n')
  let inList = false
  let html = ''
  
  for (let line of lines) {
    let trimmed = line.trim()
    if (trimmed.startsWith('-') || trimmed.startsWith('*')) {
      if (!inList) {
        html += '<ul class="chat-list">'
        inList = true
      }
      // Remove the dash/asterisk
      let cleanText = trimmed.substring(1).trim()
      html += `<li>${cleanText}</li>`
    } else {
      if (inList) {
        html += '</ul>'
        inList = false
      }
      if (trimmed === '') {
        html += '<div class="spacer"></div>'
      } else {
        html += `<p class="chat-paragraph">${trimmed}</p>`
      }
    }
  }
  if (inList) {
    html += '</ul>'
  }
  return html
}
</script>

<style>
/* Global Resets and Variables */
:root {
  --bg-app: #0b0f19;
  --bg-sidebar: #111827;
  --bg-bubble-user: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
  --bg-bubble-rag: rgba(30, 41, 59, 0.7);
  --border-color: rgba(255, 255, 255, 0.08);
  --text-main: #f1f5f9;
  --text-secondary: #94a3b8;
  --accent-color: #6366f1;
  --success-color: #10b981;
  --warning-color: #f59e0b;
  --error-color: #ef4444;
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: 'Inter', system-ui, sans-serif;
  background-color: var(--bg-app);
  color: var(--text-main);
  overflow: hidden;
  height: 100vh;
}

/* Layout */
.app-container {
  display: flex;
  height: 100vh;
  width: 100vw;
}

/* Sidebar */
.sidebar {
  width: 320px;
  background-color: var(--bg-sidebar);
  border-right: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  padding: 24px;
  flex-shrink: 0;
}

.sidebar-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 32px;
}

.logo-icon {
  font-size: 28px;
}

.sidebar-header h2 {
  font-family: 'Outfit', sans-serif;
  font-size: 22px;
  font-weight: 700;
  background: linear-gradient(120deg, #818cf8, #c084fc);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.info-section {
  margin-bottom: 28px;
}

.info-section h3 {
  font-family: 'Outfit', sans-serif;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-secondary);
  margin-bottom: 12px;
  font-weight: 600;
}

.info-card {
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
}

.info-label {
  color: var(--text-secondary);
}

.info-value {
  font-weight: 500;
}

.status-badge {
  padding: 4px 8px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.02em;
}

.status-badge.online {
  background-color: rgba(16, 185, 129, 0.15);
  color: var(--success-color);
}

.status-badge.offline {
  background-color: rgba(239, 68, 68, 0.15);
  color: var(--error-color);
}

.doc-item {
  display: flex;
  align-items: center;
  gap: 12px;
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 12px;
}

.doc-icon {
  font-size: 24px;
}

.doc-details {
  display: flex;
  flex-direction: column;
}

.doc-name {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-main);
  word-break: break-all;
}

.doc-size {
  font-size: 11px;
  color: var(--text-secondary);
}

.suggestions-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.sug-btn {
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid var(--border-color);
  color: var(--text-main);
  padding: 10px 14px;
  border-radius: 8px;
  font-size: 12px;
  text-align: left;
  cursor: pointer;
  transition: all 0.2s ease;
  line-height: 1.4;
}

.sug-btn:hover:not(:disabled) {
  background: rgba(99, 102, 241, 0.08);
  border-color: rgba(99, 102, 241, 0.3);
  transform: translateY(-1px);
}

.sug-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.sidebar-footer {
  margin-top: auto;
  font-size: 11px;
  color: var(--text-secondary);
  text-align: center;
  border-top: 1px solid var(--border-color);
  padding-top: 16px;
}

/* Chat Area */
.chat-area {
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  background-color: var(--bg-app);
  position: relative;
}

/* Glow decoration */
.chat-area::before {
  content: '';
  position: absolute;
  top: -10%;
  right: -10%;
  width: 500px;
  height: 500px;
  background: radial-gradient(circle, rgba(99, 102, 241, 0.06) 0%, rgba(0, 0, 0, 0) 70%);
  z-index: 0;
  pointer-events: none;
}

.chat-header {
  padding: 20px 32px;
  border-bottom: 1px solid var(--border-color);
  display: flex;
  justify-content: space-between;
  align-items: center;
  z-index: 1;
  background: rgba(11, 15, 25, 0.8);
  backdrop-filter: blur(12px);
}

.chat-header h1 {
  font-family: 'Outfit', sans-serif;
  font-size: 20px;
  font-weight: 600;
  color: var(--text-main);
  margin-bottom: 4px;
}

.chat-header p {
  font-size: 13px;
  color: var(--text-secondary);
}

.connection-status {
  display: flex;
  align-items: center;
  gap: 8px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid var(--border-color);
  padding: 6px 14px;
  border-radius: 30px;
}

.pulse-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}

.pulse-dot.green {
  background-color: var(--success-color);
  box-shadow: 0 0 8px var(--success-color);
  animation: pulse-green 2s infinite;
}

.pulse-dot.yellow {
  background-color: var(--warning-color);
  box-shadow: 0 0 8px var(--warning-color);
  animation: pulse-yellow 2s infinite;
}

.pulse-dot.red {
  background-color: var(--error-color);
  box-shadow: 0 0 8px var(--error-color);
  animation: pulse-red 2s infinite;
}

.connection-text {
  font-size: 12px;
  font-weight: 500;
}

/* Messages List */
.messages-container {
  flex-grow: 1;
  padding: 32px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 24px;
  z-index: 1;
}

.welcome-screen {
  max-width: 680px;
  margin: 60px auto;
  text-align: center;
}

.welcome-icon {
  font-size: 56px;
  margin-bottom: 16px;
  animation: float 4s ease-in-out infinite;
}

.welcome-screen h2 {
  font-family: 'Outfit', sans-serif;
  font-size: 28px;
  margin-bottom: 12px;
  font-weight: 700;
  background: linear-gradient(135deg, #fff 30%, #a5b4fc 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.welcome-screen p {
  color: var(--text-secondary);
  font-size: 15px;
  line-height: 1.6;
  margin-bottom: 36px;
}

.start-hints {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}

.hint-card {
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 18px;
  text-align: left;
}

.hint-card h4 {
  font-family: 'Outfit', sans-serif;
  font-size: 14px;
  color: var(--text-main);
  margin-bottom: 8px;
  font-weight: 600;
}

.hint-card p {
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.5;
  margin-bottom: 0;
}

/* Message Bubble Rows */
.message-row {
  display: flex;
  width: 100%;
}

.message-row.user {
  justify-content: flex-end;
}

.message-row.rag {
  justify-content: flex-start;
}

.message-bubble-wrapper {
  display: flex;
  gap: 16px;
  max-width: 75%;
}

.user .message-bubble-wrapper {
  flex-direction: row-reverse;
}

.avatar {
  width: 38px;
  height: 38px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  flex-shrink: 0;
}

.user .avatar {
  background: var(--accent-color);
}

.message-content-wrapper {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.user .message-content-wrapper {
  align-items: flex-end;
}

.sender-name {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-secondary);
}

.message-bubble {
  padding: 16px 20px;
  border-radius: 16px;
  font-size: 14px;
  line-height: 1.6;
  position: relative;
  word-break: break-word;
}

.user .message-bubble {
  background: var(--bg-bubble-user);
  color: #fff;
  border-bottom-right-radius: 4px;
  box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2);
}

.rag .message-bubble {
  background: var(--bg-bubble-rag);
  color: var(--text-main);
  border-bottom-left-radius: 4px;
  border: 1px solid var(--border-color);
  backdrop-filter: blur(8px);
}

.chat-paragraph {
  margin-bottom: 12px;
}
.chat-paragraph:last-child {
  margin-bottom: 0;
}

.chat-list {
  margin-left: 20px;
  margin-bottom: 12px;
}
.chat-list:last-child {
  margin-bottom: 0;
}
.chat-list li {
  margin-bottom: 6px;
}

.spacer {
  height: 12px;
}

/* Thinking loader animation */
.thinking-bubble {
  display: flex;
  align-items: center;
  gap: 12px;
  color: var(--text-secondary);
}

.typing-loader {
  display: flex;
  gap: 4px;
}

.typing-loader span {
  width: 6px;
  height: 6px;
  background-color: var(--text-secondary);
  border-radius: 50%;
  animation: bounce 1.4s infinite ease-in-out both;
}

.typing-loader span:nth-child(1) { animation-delay: -0.32s; }
.typing-loader span:nth-child(2) { animation-delay: -0.16s; }

.thinking-text {
  font-size: 13px;
  font-style: italic;
}

/* Sources toggle */
.sources-wrapper {
  margin-top: 10px;
  width: 100%;
}

.toggle-sources-btn {
  background: none;
  border: none;
  color: var(--accent-color);
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  padding: 4px 0;
  display: flex;
  align-items: center;
  transition: opacity 0.2s;
}

.toggle-sources-btn:hover {
  opacity: 0.8;
}

.sources-list {
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-width: 100%;
}

.source-card {
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 12px;
  font-size: 12px;
}

.source-card-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 6px;
  font-weight: 600;
}

.source-pdf {
  color: var(--text-main);
}

.source-page {
  color: var(--accent-color);
}

.source-snippet {
  color: var(--text-secondary);
  line-height: 1.5;
  font-style: italic;
  white-space: pre-line;
}

/* Chat Input Bar */
.chat-input-area {
  padding: 24px 32px;
  border-top: 1px solid var(--border-color);
  z-index: 1;
  background: rgba(11, 15, 25, 0.9);
  backdrop-filter: blur(12px);
}

.input-form {
  max-width: 800px;
  margin: 0 auto;
  display: flex;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid var(--border-color);
  border-radius: 14px;
  padding: 6px 6px 6px 18px;
  align-items: center;
  transition: all 0.3s ease;
}

.input-form:focus-within {
  border-color: rgba(99, 102, 241, 0.5);
  box-shadow: 0 0 15px rgba(99, 102, 241, 0.1);
  background: rgba(255, 255, 255, 0.05);
}

.input-form input {
  flex-grow: 1;
  background: transparent;
  border: none;
  color: var(--text-main);
  font-size: 14px;
  padding: 10px 0;
  outline: none;
}

.input-form input::placeholder {
  color: var(--text-secondary);
}

.send-btn {
  background: var(--accent-color);
  border: none;
  color: #fff;
  width: 40px;
  height: 40px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s ease;
  flex-shrink: 0;
}

.send-btn:hover:not(:disabled) {
  background: var(--bg-bubble-user);
  transform: scale(1.05);
}

.send-btn:disabled {
  background: rgba(255, 255, 255, 0.05);
  color: var(--text-secondary);
  cursor: not-allowed;
}

/* Animations */
@keyframes bounce {
  0%, 80%, 100% { transform: scale(0); }
  40% { transform: scale(1.0); }
}

@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-10px); }
}

@keyframes pulse-green {
  0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
  70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
  100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

@keyframes pulse-yellow {
  0% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.4); }
  70% { box-shadow: 0 0 0 6px rgba(245, 158, 11, 0); }
  100% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
}

@keyframes pulse-red {
  0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
  70% { box-shadow: 0 0 0 6px rgba(239, 68, 68, 0); }
  100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}

.animate-slide-down {
  animation: slideDown 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

@keyframes slideDown {
  from {
    opacity: 0;
    transform: translateY(-8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Scrollbar customization */
::-webkit-scrollbar {
  width: 6px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.2);
}

/* Responsive design */
@media (max-width: 768px) {
  .app-container {
    flex-direction: column;
  }
  .sidebar {
    width: 100%;
    border-right: none;
    border-bottom: 1px solid var(--border-color);
    padding: 16px;
    height: auto;
  }
  .sidebar-header {
    margin-bottom: 16px;
  }
  .info-section {
    display: none; /* Hide technical details on mobile to save space */
  }
  .info-section.suggestions-section {
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    margin-bottom: 0;
  }
  .sug-btn {
    padding: 6px 12px;
  }
  .chat-header {
    padding: 16px;
  }
  .messages-container {
    padding: 16px;
  }
  .welcome-screen {
    margin: 20px auto;
  }
  .start-hints {
    grid-template-columns: 1fr;
  }
  .message-bubble-wrapper {
    max-width: 90%;
  }
  .chat-input-area {
    padding: 16px;
  }
}
</style>

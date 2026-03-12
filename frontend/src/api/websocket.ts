type MessageHandler = (msg: Record<string, unknown>) => void

/**
 * TTS WebSocket session manager.
 * Maintains a single connection, supports multiple serial sessions.
 */
class TTSWebSocket {
  private ws: WebSocket | null = null
  private handlers = new Map<string, MessageHandler>()
  private connectPromise: Promise<void> | null = null

  private getUrl(): string {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${protocol}//${location.host}/ws/tts`
  }

  async connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) return
    if (this.connectPromise) return this.connectPromise

    this.connectPromise = new Promise((resolve, reject) => {
      const ws = new WebSocket(this.getUrl())

      ws.onopen = () => {
        this.ws = ws
        this.connectPromise = null
        resolve()
      }

      ws.onerror = () => {
        this.connectPromise = null
        reject(new Error('WebSocket connection failed'))
      }

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data)
          const sessionId = msg.session_id as string
          if (sessionId) {
            this.handlers.get(sessionId)?.(msg)
          }
          // Also notify global handlers
          this.handlers.get('*')?.(msg)
        } catch {
          // ignore
        }
      }

      ws.onclose = () => {
        this.ws = null
        this.connectPromise = null
      }
    })

    return this.connectPromise
  }

  send(msg: Record<string, unknown>): void {
    if (this.ws?.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected')
    }
    this.ws.send(JSON.stringify(msg))
  }

  onSession(sessionId: string, handler: MessageHandler): () => void {
    this.handlers.set(sessionId, handler)
    return () => this.handlers.delete(sessionId)
  }

  onAll(handler: MessageHandler): () => void {
    this.handlers.set('*', handler)
    return () => this.handlers.delete('*')
  }

  async startSession(sessionId: string, config: {
    engine: string
    voice_preset?: string
    voice?: string
    speed?: number
    sample_rate?: number
    emotion?: string
  }): Promise<void> {
    await this.connect()
    this.send({ type: 'start_session', session_id: sessionId, ...config })
  }

  synthesize(sessionId: string, text: string, emotion?: string): void {
    this.send({ type: 'synthesize', session_id: sessionId, text, emotion })
  }

  endSession(sessionId: string): void {
    this.send({ type: 'end_session', session_id: sessionId })
  }

  disconnect(): void {
    this.ws?.close()
    this.ws = null
    this.handlers.clear()
  }
}

export const ttsWS = new TTSWebSocket()

// Re-export legacy hook for backward compat with existing task WS
export { useTaskWebSocket } from './legacyWebSocket'

import { useEffect, useRef } from 'react'
import { useExecutionStore } from '../stores/execution'

export function useTaskWebSocket(taskId: string | null) {
  const wsRef = useRef<WebSocket | null>(null)
  const { setProgress, succeed, fail } = useExecutionStore()

  useEffect(() => {
    if (!taskId) return

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${location.host}/ws/tasks/${taskId}`)
    wsRef.current = ws

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)

        if (msg.progress !== undefined) {
          setProgress(msg.progress)
        }

        if (msg.status === 'completed' && msg.result) {
          succeed({
            audioBase64: msg.result.audio_base64,
            sampleRate: msg.result.sample_rate,
            duration: msg.result.duration_seconds,
          })
          ws.close()
        }

        if (msg.status === 'failed') {
          fail(msg.error || 'Task failed')
          ws.close()
        }
      } catch {
        // ignore non-JSON messages
      }
    }

    ws.onerror = () => {
      fail('WebSocket connection error')
    }

    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [taskId, setProgress, succeed, fail])
}

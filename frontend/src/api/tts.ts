import { useMutation } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface SynthesizeRequest {
  engine: string
  text: string
  voice?: string
  speed?: number
  sample_rate?: number
  reference_audio?: string
  reference_text?: string
  emotion?: string
  cache?: boolean
}

export interface SynthesizeResponse {
  audio_base64: string
  sample_rate: number
  duration_seconds: number
  engine: string
  rtf: number
  format: string
  cached: boolean
}

export interface StreamChunk {
  seq: number
  audio: string
  format: string
}

export interface StreamDone {
  total_chunks: number
  duration_ms: number
  usage: { characters: number; rtf: number }
}

export interface StreamError {
  code: string
  message: string
}

export function useSynthesize() {
  return useMutation({
    mutationFn: (req: SynthesizeRequest) =>
      apiFetch<SynthesizeResponse>('/api/v1/tts/synthesize', {
        method: 'POST',
        body: JSON.stringify(req),
      }),
  })
}

/**
 * Consume SSE stream from POST /api/v1/tts/stream.
 * Calls onChunk for each audio chunk and onDone when complete.
 */
export async function streamTTS(
  req: SynthesizeRequest,
  callbacks: {
    onChunk: (chunk: StreamChunk) => void
    onDone: (done: StreamDone) => void
    onError: (err: StreamError) => void
  },
  signal?: AbortSignal,
): Promise<void> {
  const resp = await fetch('/api/v1/tts/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal,
  })

  if (!resp.ok || !resp.body) {
    callbacks.onError({ code: 'HTTP_ERROR', message: `HTTP ${resp.status}` })
    return
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    let currentEvent = ''
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEvent = line.slice(7).trim()
      } else if (line.startsWith('data: ')) {
        const data = line.slice(6)
        try {
          const parsed = JSON.parse(data)
          if (currentEvent === 'audio') callbacks.onChunk(parsed)
          else if (currentEvent === 'done') callbacks.onDone(parsed)
          else if (currentEvent === 'error') callbacks.onError(parsed)
        } catch {
          // ignore parse errors
        }
        currentEvent = ''
      }
    }
  }
}

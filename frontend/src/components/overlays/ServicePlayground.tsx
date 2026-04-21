import { useEffect, useRef, useState } from 'react'
import { Play, Square, Eye, EyeOff } from 'lucide-react'
import type { CatalogService } from '../../api/apiGateway'

interface Props {
  svc: CatalogService
}

export default function ServicePlayground({ svc }: Props) {
  const [apiKey, setApiKey] = useState('')
  const [showKey, setShowKey] = useState(false)
  const [input, setInput] = useState('你好，简单介绍一下自己。')
  const [output, setOutput] = useState('')
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => () => abortRef.current?.abort(), [])

  const handleToggle = async () => {
    if (running) {
      abortRef.current?.abort()
      setRunning(false)
      return
    }
    if (!apiKey.trim()) {
      setError('请先填写 API Key')
      return
    }
    setError(null)
    setOutput('')
    setRunning(true)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const resp = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: svc.instance_name,
          messages: [{ role: 'user', content: input }],
          stream: true,
        }),
        signal: controller.signal,
      })
      if (!resp.ok || !resp.body) {
        const text = await resp.text().catch(() => '')
        throw new Error(`HTTP ${resp.status}: ${text.slice(0, 200)}`)
      }
      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop() || ''
        for (const line of lines) {
          const s = line.trim()
          if (!s.startsWith('data:')) continue
          const payload = s.slice(5).trim()
          if (payload === '[DONE]' || !payload) continue
          try {
            const chunk = JSON.parse(payload)
            const delta = chunk.choices?.[0]?.delta?.content
            if (delta) setOutput((prev) => prev + delta)
          } catch {
            // skip malformed chunk, keep stream alive
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== 'AbortError') {
        setError((e as Error).message)
      }
    } finally {
      setRunning(false)
      abortRef.current = null
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <div
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '6px 10px',
            borderRadius: 4,
            border: '1px solid var(--border)',
            background: 'var(--bg)',
          }}
        >
          <span style={{ fontSize: 12, color: 'var(--muted)' }}>API Key</span>
          <input
            type={showKey ? 'text' : 'password'}
            placeholder="sk-..."
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            style={{
              flex: 1, background: 'transparent', outline: 'none', border: 'none',
              fontSize: 12, color: 'var(--fg)', fontFamily: 'JetBrains Mono, monospace',
            }}
          />
          <button
            type="button"
            onClick={() => setShowKey((s) => !s)}
            style={{ color: 'var(--muted)', cursor: 'pointer' }}
            aria-label={showKey ? 'hide key' : 'show key'}
          >
            {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
          </button>
        </div>
      </div>

      <textarea
        value={input}
        onChange={(e) => setInput(e.target.value)}
        rows={3}
        placeholder="输入 prompt..."
        style={{
          width: '100%',
          padding: 10,
          borderRadius: 4,
          border: '1px solid var(--border)',
          background: 'var(--bg)',
          color: 'var(--fg)',
          fontSize: 13,
          fontFamily: 'inherit',
          resize: 'vertical',
        }}
      />

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={handleToggle}
          style={{
            display: 'inline-flex', alignItems: 'center', gap: 4,
            padding: '6px 14px', borderRadius: 4,
            background: running ? 'var(--muted)' : 'var(--accent)',
            color: 'white', fontSize: 13, fontWeight: 500,
            border: 'none', cursor: 'pointer',
          }}
        >
          {running ? <Square size={13} /> : <Play size={13} />}
          {running ? 'Stop' : 'Run'}
        </button>
        {error && (
          <span style={{ fontSize: 12, color: 'var(--accent)' }}>{error}</span>
        )}
      </div>

      <pre
        style={{
          minHeight: 80,
          padding: 10,
          borderRadius: 4,
          background: 'var(--bg)',
          border: '1px solid var(--border)',
          color: 'var(--fg)',
          fontSize: 12,
          fontFamily: 'JetBrains Mono, monospace',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          margin: 0,
        }}
      >
        {output || (running ? '…' : '输出会显示在这里')}
      </pre>
    </div>
  )
}

import { useEffect, useRef, useState } from 'react'
import { Play, Square, Eye, EyeOff } from 'lucide-react'
import type { CatalogService } from '../../api/apiGateway'

interface Props {
  svc: CatalogService
}

export default function ServicePlayground({ svc }: Props) {
  // ASR(语音识别)= 音频进文本出,走 multipart /v1/audio/transcriptions,与 LLM 的
  // JSON chat 流式完全不同 → 单独分支(2026-06-21:用户反馈 ASR 没法在 playground 测)。
  const isAsr = svc.category === 'asr'
  const [apiKey, setApiKey] = useState('')
  const [showKey, setShowKey] = useState(false)
  const [input, setInput] = useState('你好，简单介绍一下自己。')
  const [file, setFile] = useState<File | null>(null)  // ASR 音频文件
  const [output, setOutput] = useState('')
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => () => abortRef.current?.abort(), [])

  // ASR:multipart 上传音频 → 转写文本(非流式,一次返回)。
  const runAsr = async (controller: AbortController) => {
    if (!file) {
      setError('请先选择音频文件')
      setRunning(false)
      return
    }
    const fd = new FormData()
    fd.append('file', file)
    fd.append('model', svc.instance_name)
    // 不设 Content-Type —— 浏览器自动带 multipart boundary。
    const resp = await fetch('/v1/audio/transcriptions', {
      method: 'POST',
      headers: { Authorization: `Bearer ${apiKey}` },
      body: fd,
      signal: controller.signal,
    })
    if (!resp.ok) {
      const t = await resp.text().catch(() => '')
      throw new Error(`HTTP ${resp.status}: ${t.slice(0, 200)}`)
    }
    const data = await resp.json()
    setOutput(typeof data?.text === 'string' ? data.text : JSON.stringify(data, null, 2))
  }

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
    if (isAsr && !file) {
      setError('请先选择音频文件')
      return
    }
    setError(null)
    setOutput('')
    setRunning(true)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      if (isAsr) {
        await runAsr(controller)
        return
      }
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

      {isAsr ? (
        <label
          style={{
            display: 'flex', alignItems: 'center', gap: 8,
            padding: 10, borderRadius: 4,
            border: '1px dashed var(--border)', background: 'var(--bg)',
            color: 'var(--muted)', fontSize: 13, cursor: 'pointer',
          }}
        >
          <input
            type="file"
            accept="audio/*"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            style={{ display: 'none' }}
          />
          {file ? (
            <span style={{ color: 'var(--fg)' }}>
              {file.name} · {(file.size / 1024).toFixed(0)} KB
            </span>
          ) : (
            <span>选择音频文件(wav/mp3/m4a…)转写为文本</span>
          )}
        </label>
      ) : (
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
      )}

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

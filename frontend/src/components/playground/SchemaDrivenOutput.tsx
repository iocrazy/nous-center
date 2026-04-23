import type { ExposedParam } from '../../api/services'
import { paramKey } from '../../api/services'

export interface SchemaDrivenOutputProps {
  outputs: ExposedParam[]
  /** Run result. Treated as `{ key: value }` keyed by exposed_outputs[].key. */
  result: Record<string, unknown> | null
  taskInfo?: {
    task_id?: string
    status?: 'running' | 'completed' | 'failed'
    latency_ms?: number
    cost?: string
  }
  error?: string | null
}

function pickMime(p: ExposedParam): 'audio' | 'video' | 'image' | 'json' | 'text' {
  const t = (p.type ?? '').toLowerCase()
  if (t.includes('audio')) return 'audio'
  if (t.includes('video')) return 'video'
  if (t.includes('image')) return 'image'
  const c = (p.constraints ?? {}) as Record<string, unknown>
  const mime = String(c.mime ?? '')
  if (mime.startsWith('audio/')) return 'audio'
  if (mime.startsWith('video/')) return 'video'
  if (mime.startsWith('image/')) return 'image'
  if (t === 'object' || t === 'json') return 'json'
  return 'text'
}

export default function SchemaDrivenOutput({
  outputs,
  result,
  taskInfo,
  error,
}: SchemaDrivenOutputProps) {
  return (
    <div
      style={{
        background: 'var(--bg-accent)',
        border: '1px solid var(--border)',
        borderRadius: 8,
        padding: '16px 18px',
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
        height: '100%',
        overflow: 'auto',
      }}
    >
      <h3
        style={{
          fontSize: 12,
          color: 'var(--muted)',
          textTransform: 'uppercase',
          letterSpacing: 0.5,
        }}
      >
        输出
      </h3>

      {taskInfo && <TaskInfoBox info={taskInfo} />}
      {error && <ErrorBox message={error} />}

      {!result && !error && (
        <div
          style={{
            color: 'var(--muted)',
            fontSize: 12,
            padding: 24,
            textAlign: 'center',
          }}
        >
          运行后输出会显示在这里
        </div>
      )}

      {result &&
        outputs.map((p) => {
          const k = paramKey(p)
          if (!k) return null
          const v = result[k]
          if (v === undefined) return null
          return (
            <OutputBlock
              key={`${p.node_id}.${k}`}
              label={p.label || k}
              nodeId={p.node_id}
              kind={pickMime(p)}
              value={v}
            />
          )
        })}

      {result && outputs.length === 0 && (
        // No declared schema → just dump raw JSON.
        <pre
          style={{
            background: 'var(--bg)',
            border: '1px solid var(--border)',
            borderRadius: 4,
            padding: 12,
            fontSize: 11,
            fontFamily: 'var(--mono, monospace)',
            color: 'var(--text)',
            margin: 0,
            overflow: 'auto',
          }}
        >
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  )
}

function OutputBlock({
  label,
  nodeId,
  kind,
  value,
}: {
  label: string
  nodeId: string
  kind: 'audio' | 'video' | 'image' | 'json' | 'text'
  value: unknown
}) {
  return (
    <div>
      <div
        style={{
          fontSize: 11,
          color: 'var(--muted)',
          textTransform: 'uppercase',
          marginBottom: 6,
        }}
      >
        {label}{' '}
        <span style={{ color: 'var(--muted)', fontFamily: 'var(--mono, monospace)', fontSize: 10 }}>
          ← node {nodeId}
        </span>
      </div>
      <Renderer kind={kind} value={value} />
    </div>
  )
}

function asUrlOrText(value: unknown): { kind: 'url' | 'text'; value: string } {
  if (typeof value === 'string') {
    if (value.startsWith('http://') || value.startsWith('https://') || value.startsWith('data:') || value.startsWith('/')) {
      return { kind: 'url', value }
    }
    return { kind: 'text', value }
  }
  return { kind: 'text', value: JSON.stringify(value) }
}

function Renderer({ kind, value }: { kind: 'audio' | 'video' | 'image' | 'json' | 'text'; value: unknown }) {
  if (kind === 'audio') {
    const u = asUrlOrText(value)
    if (u.kind === 'url') return <audio controls src={u.value} style={{ width: '100%' }} />
    return <Mono>{u.value}</Mono>
  }
  if (kind === 'video') {
    const u = asUrlOrText(value)
    if (u.kind === 'url') return <video controls src={u.value} style={{ width: '100%', borderRadius: 4 }} />
    return <Mono>{u.value}</Mono>
  }
  if (kind === 'image') {
    const u = asUrlOrText(value)
    if (u.kind === 'url') return <img src={u.value} alt="" style={{ width: '100%', borderRadius: 4 }} />
    return <Mono>{u.value}</Mono>
  }
  if (kind === 'json') {
    return (
      <pre
        style={{
          background: 'var(--bg)',
          border: '1px solid var(--border)',
          borderRadius: 4,
          padding: 12,
          fontSize: 11,
          fontFamily: 'var(--mono, monospace)',
          color: 'var(--text)',
          margin: 0,
          overflow: 'auto',
        }}
      >
        {JSON.stringify(value, null, 2)}
      </pre>
    )
  }
  // text
  if (typeof value === 'string') {
    return (
      <div
        style={{
          background: 'var(--bg)',
          border: '1px solid var(--border)',
          borderRadius: 4,
          padding: 10,
          fontSize: 12,
          color: 'var(--text)',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}
      >
        {value}
      </div>
    )
  }
  return <Mono>{JSON.stringify(value)}</Mono>
}

function Mono({ children }: { children: React.ReactNode }) {
  return (
    <code
      style={{
        background: 'var(--bg)',
        border: '1px solid var(--border)',
        borderRadius: 4,
        padding: '2px 6px',
        fontSize: 11,
        fontFamily: 'var(--mono, monospace)',
        color: 'var(--text)',
        wordBreak: 'break-all',
      }}
    >
      {children}
    </code>
  )
}

function TaskInfoBox({
  info,
}: {
  info: { task_id?: string; status?: string; latency_ms?: number; cost?: string }
}) {
  return (
    <div
      style={{
        fontSize: 11,
        padding: '8px 10px',
        background: 'var(--bg)',
        borderRadius: 4,
        fontFamily: 'var(--mono, monospace)',
        color: 'var(--muted)',
      }}
    >
      {info.task_id && <Row k="task_id" v={info.task_id} />}
      {info.status && (
        <Row
          k="status"
          v={info.status}
          color={
            info.status === 'completed'
              ? 'var(--accent-2, #22c55e)'
              : info.status === 'failed'
                ? 'var(--error, #ef4444)'
                : 'var(--text)'
          }
        />
      )}
      {info.latency_ms != null && <Row k="latency" v={`${info.latency_ms} ms`} />}
      {info.cost && <Row k="cost" v={info.cost} />}
    </div>
  )
}

function Row({ k, v, color }: { k: string; v: string; color?: string }) {
  return (
    <div>
      <span style={{ color: 'var(--muted)' }}>{k}: </span>
      <span style={{ color: color ?? 'var(--text)' }}>{v}</span>
    </div>
  )
}

function ErrorBox({ message }: { message: string }) {
  return (
    <div
      style={{
        background: 'rgba(239, 68, 68, 0.1)',
        border: '1px solid var(--error, #ef4444)',
        color: 'var(--error, #ef4444)',
        padding: '8px 10px',
        borderRadius: 4,
        fontSize: 12,
      }}
    >
      {message}
    </div>
  )
}

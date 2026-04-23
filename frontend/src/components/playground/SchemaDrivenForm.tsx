import { useState, useMemo, type ChangeEvent, type FormEvent } from 'react'
import type { ExposedParam } from '../../api/services'
import { paramKey, paramSlot } from '../../api/services'

export interface SchemaDrivenFormProps {
  inputs: ExposedParam[]
  /** Called with `{ key: value }` keyed by `param.key`, ready to POST. */
  onSubmit: (values: Record<string, unknown>) => void
  submitting?: boolean
  submitLabel?: string
  estimateLine?: string
}

type FieldKind =
  | 'string'
  | 'string_multiline'
  | 'number'
  | 'integer'
  | 'boolean'
  | 'file'
  | 'select'

function classifyField(p: ExposedParam): FieldKind {
  const constraints = (p.constraints ?? {}) as Record<string, unknown>
  if (Array.isArray(constraints.enum) && constraints.enum.length > 0) return 'select'
  const t = (p.type ?? 'string').toLowerCase()
  if (t.includes('multiline') || t.includes('text') || t === 'string_multi') return 'string_multiline'
  if (t === 'integer' || t === 'int') return 'integer'
  if (t === 'number' || t === 'float') return 'number'
  if (t === 'boolean' || t === 'bool') return 'boolean'
  if (t === 'file' || t === 'image' || t === 'audio' || t === 'video' || t === 'binary') return 'file'
  return 'string'
}

function defaultFor(p: ExposedParam): unknown {
  if (p.default !== undefined && p.default !== null) return p.default
  switch (classifyField(p)) {
    case 'boolean':
      return false
    case 'number':
    case 'integer':
      return ''
    case 'file':
      return null
    default:
      return ''
  }
}

export default function SchemaDrivenForm({
  inputs,
  onSubmit,
  submitting,
  submitLabel = '▶ 运行',
  estimateLine,
}: SchemaDrivenFormProps) {
  const initial = useMemo(() => {
    const acc: Record<string, unknown> = {}
    for (const p of inputs) {
      const k = paramKey(p)
      if (!k) continue
      acc[k] = defaultFor(p)
    }
    return acc
  }, [inputs])

  const [values, setValues] = useState<Record<string, unknown>>(initial)

  const update = (k: string, v: unknown) => setValues((prev) => ({ ...prev, [k]: v }))

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    onSubmit(values)
  }

  const reset = () => setValues(initial)

  return (
    <form onSubmit={handleSubmit} className="flex flex-col h-full">
      <div className="flex-1 overflow-auto" style={{ padding: '16px 18px' }}>
        {inputs.length === 0 && (
          <div
            className="text-center"
            style={{ color: 'var(--muted)', fontSize: 12, padding: 24 }}
          >
            该服务没有暴露入参 · 直接点运行即可
          </div>
        )}
        {inputs.map((p) => (
          <Field
            key={`${p.node_id}.${paramKey(p)}`}
            param={p}
            value={values[paramKey(p) ?? '']}
            onChange={(v) => {
              const k = paramKey(p)
              if (k) update(k, v)
            }}
          />
        ))}
      </div>
      <div
        className="flex items-center gap-2"
        style={{
          padding: '12px 16px',
          borderTop: '1px solid var(--border)',
          background: 'var(--bg-accent)',
        }}
      >
        {estimateLine && (
          <span style={{ flex: 1, fontSize: 11, color: 'var(--muted)' }}>{estimateLine}</span>
        )}
        <button
          type="button"
          onClick={reset}
          className="btn"
          style={{
            fontSize: 12,
            padding: '6px 10px',
            background: 'transparent',
            color: 'var(--muted)',
            border: '1px solid var(--border)',
            borderRadius: 4,
            cursor: 'pointer',
          }}
        >
          重置
        </button>
        <button
          type="submit"
          disabled={submitting}
          style={{
            fontSize: 12,
            padding: '6px 14px',
            background: 'var(--accent)',
            color: '#fff',
            border: 'none',
            borderRadius: 4,
            cursor: submitting ? 'not-allowed' : 'pointer',
            opacity: submitting ? 0.6 : 1,
          }}
        >
          {submitting ? '运行中…' : submitLabel}
        </button>
      </div>
    </form>
  )
}

function Field({
  param,
  value,
  onChange,
}: {
  param: ExposedParam
  value: unknown
  onChange: (v: unknown) => void
}) {
  const kind = classifyField(param)
  const slot = paramSlot(param) ?? '?'

  return (
    <div className="flex flex-col gap-1.5 mb-3.5">
      <label
        className="flex items-center gap-1.5"
        style={{
          fontSize: 11,
          color: 'var(--muted)',
          textTransform: 'uppercase',
          letterSpacing: 0.5,
        }}
      >
        <span style={{ color: 'var(--text)' }}>{param.label || paramKey(param) || '(unnamed)'}</span>
        {param.required && <span style={{ color: 'var(--accent)', fontSize: 10 }}>*</span>}
        <span
          style={{
            marginLeft: 'auto',
            fontFamily: 'var(--mono, monospace)',
            fontSize: 10,
            color: 'var(--muted)',
          }}
        >
          node={param.node_id} · {slot}
        </span>
      </label>
      <FieldInput kind={kind} param={param} value={value} onChange={onChange} />
    </div>
  )
}

function FieldInput({
  kind,
  param,
  value,
  onChange,
}: {
  kind: FieldKind
  param: ExposedParam
  value: unknown
  onChange: (v: unknown) => void
}) {
  const inputStyle = {
    width: '100%',
    background: 'var(--bg)',
    color: 'var(--text)',
    border: '1px solid var(--border)',
    borderRadius: 4,
    padding: '6px 8px',
    fontSize: 12,
  } as const

  if (kind === 'string_multiline') {
    return (
      <textarea
        value={(value as string) ?? ''}
        onChange={(e: ChangeEvent<HTMLTextAreaElement>) => onChange(e.target.value)}
        rows={4}
        style={{
          ...inputStyle,
          minHeight: 80,
          resize: 'vertical',
          fontFamily: 'var(--mono, monospace)',
        }}
      />
    )
  }

  if (kind === 'select') {
    const opts = ((param.constraints as { enum?: unknown[] } | undefined)?.enum ?? []) as unknown[]
    return (
      <select
        value={String(value ?? '')}
        onChange={(e) => onChange(e.target.value)}
        style={inputStyle}
      >
        {opts.map((o) => (
          <option key={String(o)} value={String(o)}>
            {String(o)}
          </option>
        ))}
      </select>
    )
  }

  if (kind === 'boolean') {
    return (
      <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text)' }}>
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(e) => onChange(e.target.checked)}
        />
        {value ? 'true' : 'false'}
      </label>
    )
  }

  if (kind === 'number' || kind === 'integer') {
    return (
      <input
        type="number"
        step={kind === 'integer' ? 1 : 'any'}
        value={(value as string | number) ?? ''}
        onChange={(e) => {
          const v = e.target.value
          if (v === '') return onChange('')
          onChange(kind === 'integer' ? Number.parseInt(v, 10) : Number.parseFloat(v))
        }}
        style={inputStyle}
      />
    )
  }

  if (kind === 'file') {
    const f = value as File | null
    return (
      <label
        style={{
          display: 'block',
          border: '1px dashed var(--border)',
          borderRadius: 4,
          padding: 14,
          textAlign: 'center',
          cursor: 'pointer',
          color: 'var(--muted)',
          fontSize: 12,
        }}
      >
        {f ? `已选 ${f.name} · ${(f.size / 1024).toFixed(1)} KB` : '点击或拖入选择文件'}
        <input
          type="file"
          hidden
          onChange={(e) => onChange(e.target.files?.[0] ?? null)}
        />
      </label>
    )
  }

  return (
    <input
      type="text"
      value={(value as string) ?? ''}
      onChange={(e) => onChange(e.target.value)}
      style={inputStyle}
    />
  )
}

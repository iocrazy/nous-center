import { useState, useMemo, type ChangeEvent, type FormEvent } from 'react'
import { Dices } from 'lucide-react'
import type { ExposedParam } from '../../api/services'
import { paramKey, paramSlot } from '../../api/services'

export interface SchemaDrivenFormProps {
  inputs: ExposedParam[]
  /** Called with `{ key: value }` keyed by `param.key`, ready to POST. */
  onSubmit: (values: Record<string, unknown>) => void
  /** 预填初值(覆盖各字段 default),keyed by exposed key。「重跑(相同参数)」回填用。 */
  initialValues?: Record<string, unknown>
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
  | 'slider'

function num(v: unknown): number | undefined {
  const n = Number(v)
  return Number.isFinite(n) ? n : undefined
}

function classifyField(p: ExposedParam): FieldKind {
  const constraints = (p.constraints ?? {}) as Record<string, unknown>
  if (Array.isArray(constraints.enum) && constraints.enum.length > 0) return 'select'
  const t = (p.type ?? 'string').toLowerCase()
  const numeric = t === 'integer' || t === 'int' || t === 'number' || t === 'float'
  // Numeric field with a bounded range → slider (对齐 Infinite-Canvas / nous
  // 节点的 slider widget,min/max/step 从 ExposedParam.constraints 带出)。
  if (numeric && num(constraints.min) !== undefined && num(constraints.max) !== undefined) {
    return 'slider'
  }
  if (t === 'integer' || t === 'int') return 'integer'
  if (t === 'number' || t === 'float') return 'number'
  if (t === 'boolean' || t === 'bool') return 'boolean'
  if (t === 'file' || t === 'image' || t === 'audio' || t === 'video' || t === 'binary') return 'file'
  // Strings default to multiline. Single-line is opt-in via constraints.format='single_line'
  // — published service inputs are almost always free text (prompts, transcripts,
  // user content) and a single 32-char input is unusable for those.
  if (constraints.format === 'single_line') return 'string'
  return 'string_multiline'
}

/** Seed-like numeric field → show a 🎲 randomize button. Opt-in via
 *  constraints.random, or auto-detected by a `seed` key/slot name. */
function isRandomizable(p: ExposedParam): boolean {
  const constraints = (p.constraints ?? {}) as Record<string, unknown>
  if (constraints.random === true) return true
  const name = `${paramKey(p) ?? ''} ${paramSlot(p) ?? ''}`.toLowerCase()
  return /seed/.test(name)
}

function randomSeed(): number {
  // 0..2^31-1 — 安全整数范围内,够用作种子。crypto 优先,回退到时间。
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    return crypto.getRandomValues(new Uint32Array(1))[0] % 2147483647
  }
  return Math.floor(Date.now() % 2147483647)
}

function defaultFor(p: ExposedParam): unknown {
  if (p.default !== undefined && p.default !== null) return p.default
  switch (classifyField(p)) {
    case 'boolean':
      return false
    case 'slider': {
      const c = (p.constraints ?? {}) as Record<string, unknown>
      return num(c.min) ?? 0
    }
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
  initialValues,
  submitting,
  submitLabel = '▶ 运行',
  estimateLine,
}: SchemaDrivenFormProps) {
  const initial = useMemo(() => {
    const acc: Record<string, unknown> = {}
    for (const p of inputs) {
      const k = paramKey(p)
      if (!k) continue
      // 有预填值用预填(重跑回填),否则用字段 default。
      acc[k] = initialValues && k in initialValues ? initialValues[k] : defaultFor(p)
    }
    return acc
  }, [inputs, initialValues])

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
          清空
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
  const typeBadge =
    kind === 'string_multiline' ? 'string' : kind === 'slider' ? 'number' : kind

  return (
    <div className="flex flex-col gap-2 mb-4">
      <label
        className="flex items-center gap-2"
        title={`node=${param.node_id} · ${slot}`}
      >
        <span style={{ fontSize: 13, color: 'var(--text)', fontWeight: 500 }}>
          {param.label || paramKey(param) || '(unnamed)'}
        </span>
        <span style={{
          fontSize: 10, padding: '1px 6px', borderRadius: 3,
          background: 'var(--bg)', border: '1px solid var(--border)',
          color: 'var(--muted)', fontFamily: 'var(--mono, monospace)',
        }}>
          {typeBadge}
        </span>
        {param.required && (
          <span style={{
            fontSize: 10, color: 'var(--accent)', fontWeight: 500,
          }}>必填</span>
        )}
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
        onChange={(e: ChangeEvent<HTMLTextAreaElement>) => {
          // Auto-grow up to ~12 lines, then scroll
          const ta = e.target
          ta.style.height = 'auto'
          ta.style.height = Math.min(ta.scrollHeight, 264) + 'px'
          onChange(ta.value)
        }}
        ref={(el) => {
          // Initial sizing on mount + when value changes externally
          if (!el) return
          el.style.height = 'auto'
          el.style.height = Math.min(el.scrollHeight, 264) + 'px'
        }}
        rows={3}
        placeholder="支持多行输入"
        style={{
          ...inputStyle,
          padding: 10,
          minHeight: 72,
          resize: 'none',
          overflow: 'auto',
          lineHeight: 1.5,
          fontFamily: 'inherit',
        }}
      />
    )
  }

  if (kind === 'select') {
    const c = (param.constraints ?? {}) as { enum?: unknown[]; enum_labels?: unknown }
    const opts = (c.enum ?? []) as unknown[]
    // enum_labels: 平行数组或 {value: label} 映射,缺省用 value 本身。
    const labels = c.enum_labels
    const labelFor = (o: unknown, i: number): string => {
      if (Array.isArray(labels)) return String(labels[i] ?? o)
      if (labels && typeof labels === 'object') {
        const m = labels as Record<string, unknown>
        return String(m[String(o)] ?? o)
      }
      return String(o)
    }
    return (
      <select
        value={String(value ?? '')}
        onChange={(e) => onChange(e.target.value)}
        style={inputStyle}
      >
        {opts.map((o, i) => (
          <option key={String(o)} value={String(o)}>
            {labelFor(o, i)}
          </option>
        ))}
      </select>
    )
  }

  if (kind === 'slider') {
    const c = (param.constraints ?? {}) as Record<string, unknown>
    const min = num(c.min) ?? 0
    const max = num(c.max) ?? 1
    const step = num(c.step) ?? (Number.isInteger(min) && Number.isInteger(max) ? 1 : 0.01)
    const v = num(value) ?? min
    const set = (raw: string) => {
      if (raw === '') return onChange('')
      onChange(Number(raw))
    }
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={v}
          onChange={(e) => set(e.target.value)}
          style={{ flex: 1, accentColor: 'var(--accent)' }}
        />
        <input
          type="number"
          min={min}
          max={max}
          step={step}
          value={(value as number | string) ?? ''}
          onChange={(e) => set(e.target.value)}
          style={{ ...inputStyle, width: 92, flexShrink: 0 }}
        />
      </div>
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
    const numberInput = (
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
    if (!isRandomizable(param)) return numberInput
    // Seed-like field → 数字框 + 🎲 随机按钮。
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <div style={{ flex: 1 }}>{numberInput}</div>
        <button
          type="button"
          title="随机种子"
          aria-label="随机种子"
          onClick={() => onChange(randomSeed())}
          style={{
            display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
            width: 30, height: 30, flexShrink: 0,
            background: 'var(--bg)', border: '1px solid var(--border)',
            borderRadius: 4, color: 'var(--text)', cursor: 'pointer',
          }}
        >
          <Dices size={15} />
        </button>
      </div>
    )
  }

  if (kind === 'file') {
    // value 存的是 data-URI 字符串(JSON 可序列化);File 对象不能进 JSON body
    // —— 旧实现直接把 File 塞进 values,JSON.stringify 成 {} → 后端收到空图。
    const dataUri = typeof value === 'string' && value.startsWith('data:') ? value : null
    const t = (param.type ?? '').toLowerCase()
    const isImage = t === 'image' || (dataUri?.startsWith('data:image') ?? false)
    const accept = isImage ? 'image/*' : t === 'audio' ? 'audio/*' : t === 'video' ? 'video/*' : undefined
    const onPick = (e: ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0]
      if (!f) return onChange(null)
      const reader = new FileReader()
      reader.onload = () => onChange(reader.result as string)
      reader.readAsDataURL(f)
    }
    return (
      <label
        style={{
          display: 'block',
          border: '1px dashed var(--border)',
          borderRadius: 4,
          padding: dataUri && isImage ? 8 : 14,
          textAlign: 'center',
          cursor: 'pointer',
          color: 'var(--muted)',
          fontSize: 12,
        }}
      >
        {dataUri && isImage ? (
          <img
            src={dataUri}
            alt="preview"
            style={{ maxWidth: '100%', maxHeight: 180, borderRadius: 3, display: 'block', margin: '0 auto' }}
          />
        ) : dataUri ? (
          '已选文件 · 点击替换'
        ) : (
          '点击或拖入选择文件'
        )}
        <input type="file" accept={accept} hidden onChange={onPick} />
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

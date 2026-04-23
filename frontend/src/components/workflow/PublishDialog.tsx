import { useEffect, useMemo, useState } from 'react'
import { X, ChevronLeft, ChevronRight } from 'lucide-react'
import {
  NAME_RE,
  usePublishWorkflow,
  type ExposedParam,
  type ServiceCategory,
} from '../../api/services'

export interface PublishDialogProps {
  open: boolean
  onClose: () => void
  workflowId: string | number
  /** Editor-style nodes list ([{id, type, data}, …]); we let the user pick
   *  which ones become the public schema. */
  nodes: Array<{ id: string; type?: string; data?: Record<string, unknown> }>
  onPublished?: (serviceId: string) => void
}

type Step = 1 | 2 | 3

export default function PublishDialog({
  open,
  onClose,
  workflowId,
  nodes,
  onPublished,
}: PublishDialogProps) {
  const [step, setStep] = useState<Step>(1)
  const [inputNodeIds, setInputNodeIds] = useState<string[]>([])
  const [outputNodeIds, setOutputNodeIds] = useState<string[]>([])
  const [name, setName] = useState('')
  const [label, setLabel] = useState('')
  const [category, setCategory] = useState<ServiceCategory>('app')

  const publish = usePublishWorkflow()
  const resetMutation = publish.reset

  useEffect(() => {
    if (!open) {
      setStep(1)
      setInputNodeIds([])
      setOutputNodeIds([])
      setName('')
      setLabel('')
      setCategory('app')
      resetMutation()
      return
    }
    if (nodes.length === 0) return
    // Best-guess defaults: anything looking like input → step 1, anything
    // looking like output → step 2.
    const inputDefaults = nodes
      .filter((n) => /input|primitive|load/i.test(n.type ?? ''))
      .map((n) => n.id)
    const outputDefaults = nodes
      .filter((n) => /output|save|preview/i.test(n.type ?? ''))
      .map((n) => n.id)
    setInputNodeIds(inputDefaults)
    setOutputNodeIds(outputDefaults)
    // resetMutation is stable across renders (React Query memoizes it).
  }, [open, nodes, resetMutation])

  const exposedInputs = useMemo<ExposedParam[]>(
    () =>
      inputNodeIds.map((id, i) => ({
        node_id: id,
        key: `input_${i + 1}`,
        input_name: 'value',
        type: 'string',
        required: true,
      })),
    [inputNodeIds],
  )

  const exposedOutputs = useMemo<ExposedParam[]>(
    () =>
      outputNodeIds.map((id, i) => ({
        node_id: id,
        key: `output_${i + 1}`,
        input_name: 'value',
        type: 'string',
      })),
    [outputNodeIds],
  )

  if (!open) return null

  const nameValid = NAME_RE.test(name)
  const canSubmit = nameValid && inputNodeIds.length + outputNodeIds.length > 0

  const submit = async () => {
    if (!canSubmit) return
    try {
      const svc = await publish.mutateAsync({
        workflowId,
        body: {
          name,
          label: label.trim() || undefined,
          category,
          exposed_inputs: exposedInputs,
          exposed_outputs: exposedOutputs,
        },
      })
      onPublished?.(svc.id)
      onClose()
    } catch {
      /* surfaced in UI */
    }
  }

  return (
    <Modal onClose={onClose} title={`发布为服务 · 步骤 ${step} / 3`}>
      <Stepper step={step} />

      {step === 1 && (
        <NodePicker
          title="选输入节点（caller 提交时填这些字段）"
          nodes={nodes}
          selected={inputNodeIds}
          onToggle={(id) => toggle(setInputNodeIds, id)}
          hint="通常是 PrimitiveString / LoadImage / LoadAudio 等"
        />
      )}

      {step === 2 && (
        <NodePicker
          title="选输出节点（执行结果从这些节点读取）"
          nodes={nodes}
          selected={outputNodeIds}
          onToggle={(id) => toggle(setOutputNodeIds, id)}
          hint="通常是 SaveVideo / SaveAudio / PreviewImage 等"
        />
      )}

      {step === 3 && (
        <div>
          <Section label="服务名称">
            <input
              value={name}
              onChange={(e) => setName(e.target.value.trim())}
              placeholder="例如：ltx-drama"
              style={inputStyle}
            />
            {name && !nameValid && (
              <div style={{ fontSize: 11, color: 'var(--accent)', marginTop: 6 }}>
                必须匹配 {NAME_RE.source}
              </div>
            )}
          </Section>
          <Section label="显示标签 (可选)">
            <input value={label} onChange={(e) => setLabel(e.target.value)} style={inputStyle} />
          </Section>
          <Section label="分类">
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value as ServiceCategory)}
              style={inputStyle}
            >
              <option value="app">通用 (app)</option>
              <option value="llm">LLM</option>
              <option value="tts">TTS</option>
              <option value="vl">VL</option>
            </select>
          </Section>
          <Summary
            inputs={inputNodeIds.length}
            outputs={outputNodeIds.length}
            workflowId={String(workflowId)}
          />
        </div>
      )}

      {publish.error && (
        <div
          style={{
            background: 'rgba(239,68,68,0.1)',
            border: '1px solid var(--error, #ef4444)',
            color: 'var(--error, #ef4444)',
            padding: '8px 10px',
            borderRadius: 4,
            fontSize: 12,
            margin: '12px 0',
          }}
        >
          {(publish.error as Error).message}
        </div>
      )}

      <div style={{ display: 'flex', gap: 8, justifyContent: 'space-between', marginTop: 16 }}>
        <button
          type="button"
          onClick={() => setStep((s) => (s > 1 ? ((s - 1) as Step) : s))}
          disabled={step === 1}
          style={{ ...btnGhost, opacity: step === 1 ? 0.4 : 1 }}
        >
          <ChevronLeft size={14} style={{ marginRight: 4, verticalAlign: 'middle' }} />
          上一步
        </button>
        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={onClose} type="button" style={btnGhost}>
            取消
          </button>
          {step < 3 ? (
            <button
              type="button"
              onClick={() => setStep((s) => ((s + 1) as Step))}
              style={btnPrimary(true)}
            >
              下一步
              <ChevronRight size={14} style={{ marginLeft: 4, verticalAlign: 'middle' }} />
            </button>
          ) : (
            <button
              type="button"
              onClick={submit}
              disabled={!canSubmit || publish.isPending}
              style={btnPrimary(canSubmit && !publish.isPending)}
            >
              {publish.isPending ? '发布中…' : '发布服务'}
            </button>
          )}
        </div>
      </div>
    </Modal>
  )
}

function toggle(setter: (fn: (prev: string[]) => string[]) => void, id: string) {
  setter((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]))
}

function NodePicker({
  title,
  nodes,
  selected,
  onToggle,
  hint,
}: {
  title: string
  nodes: PublishDialogProps['nodes']
  selected: string[]
  onToggle: (id: string) => void
  hint: string
}) {
  return (
    <div>
      <div style={{ fontSize: 13, color: 'var(--text)', marginBottom: 4 }}>{title}</div>
      <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 10 }}>{hint}</div>
      {nodes.length === 0 && (
        <div style={{ fontSize: 12, color: 'var(--muted)', padding: 16, textAlign: 'center' }}>
          这个 workflow 还没有节点 · 先到画布加节点
        </div>
      )}
      <div
        style={{
          maxHeight: 240,
          overflow: 'auto',
          border: '1px solid var(--border)',
          borderRadius: 4,
        }}
      >
        {nodes.map((n) => {
          const isSel = selected.includes(n.id)
          return (
            <label
              key={n.id}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '8px 10px',
                borderBottom: '1px solid var(--border)',
                cursor: 'pointer',
                background: isSel ? 'var(--accent-subtle, rgba(99,102,241,0.08))' : 'transparent',
              }}
            >
              <input type="checkbox" checked={isSel} onChange={() => onToggle(n.id)} />
              <span style={{ fontSize: 12, color: 'var(--text)' }}>
                {n.type ?? 'node'}
              </span>
              <span
                style={{
                  marginLeft: 'auto',
                  fontSize: 10,
                  color: 'var(--muted)',
                  fontFamily: 'var(--mono, monospace)',
                }}
              >
                #{n.id}
              </span>
            </label>
          )
        })}
      </div>
    </div>
  )
}

function Stepper({ step }: { step: Step }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
      {[1, 2, 3].map((s) => (
        <div key={s} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div
            style={{
              width: 22,
              height: 22,
              borderRadius: 11,
              background: s <= step ? 'var(--accent)' : 'var(--border)',
              color: s <= step ? '#fff' : 'var(--muted)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 11,
              fontWeight: 500,
            }}
          >
            {s}
          </div>
          {s < 3 && (
            <div style={{ width: 32, height: 2, background: s < step ? 'var(--accent)' : 'var(--border)' }} />
          )}
        </div>
      ))}
      <span style={{ marginLeft: 8, fontSize: 11, color: 'var(--muted)' }}>
        {step === 1 ? '选输入节点' : step === 2 ? '选输出节点' : '命名与发布'}
      </span>
    </div>
  )
}

function Summary({
  inputs,
  outputs,
  workflowId,
}: {
  inputs: number
  outputs: number
  workflowId: string
}) {
  return (
    <div
      style={{
        marginTop: 8,
        padding: '10px 12px',
        background: 'var(--bg)',
        border: '1px solid var(--border)',
        borderRadius: 4,
        fontSize: 11,
        color: 'var(--muted)',
        fontFamily: 'var(--mono, monospace)',
      }}
    >
      workflow_id: {workflowId} · 入参 {inputs} · 出参 {outputs}
    </div>
  )
}

function Modal({
  onClose,
  title,
  children,
}: {
  onClose: () => void
  title: string
  children: React.ReactNode
}) {
  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.55)',
        zIndex: 50,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: 'var(--bg-elevated, var(--bg))',
          border: '1px solid var(--border)',
          borderRadius: 8,
          width: 540,
          maxWidth: '92vw',
          maxHeight: '88vh',
          overflow: 'auto',
          padding: 20,
          boxShadow: '0 20px 50px rgba(0,0,0,0.5)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 16 }}>
          <h2 style={{ flex: 1, fontSize: 16, fontWeight: 600, color: 'var(--text)' }}>{title}</h2>
          <button
            onClick={onClose}
            type="button"
            style={{ background: 'transparent', border: 'none', color: 'var(--muted)', cursor: 'pointer' }}
          >
            <X size={18} />
          </button>
        </div>
        {children}
      </div>
    </div>
  )
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div
        style={{
          fontSize: 11,
          color: 'var(--muted)',
          textTransform: 'uppercase',
          letterSpacing: 0.5,
          marginBottom: 6,
        }}
      >
        {label}
      </div>
      {children}
    </div>
  )
}

const inputStyle = {
  width: '100%',
  background: 'var(--bg)',
  color: 'var(--text)',
  border: '1px solid var(--border)',
  borderRadius: 4,
  padding: '7px 9px',
  fontSize: 12,
} as const

const btnGhost = {
  padding: '7px 14px',
  fontSize: 12,
  background: 'transparent',
  color: 'var(--muted)',
  border: '1px solid var(--border)',
  borderRadius: 4,
  cursor: 'pointer',
} as const

const btnPrimary = (enabled: boolean) =>
  ({
    padding: '7px 14px',
    fontSize: 12,
    background: 'var(--accent)',
    color: '#fff',
    border: 'none',
    borderRadius: 4,
    cursor: enabled ? 'pointer' : 'not-allowed',
    opacity: enabled ? 1 : 0.5,
  }) as const

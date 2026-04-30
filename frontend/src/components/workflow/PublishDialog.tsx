import { useEffect, useMemo, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  X, ChevronLeft, ChevronRight, Check, Info, Box,
} from 'lucide-react'
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

/**
 * The slot name on `node.data` that the executor reads for a given node
 * type. PublishDialog used to hard-code `input_name='value'` for every
 * exposed param, but `text_input` reads `data.text`, so the merge in
 * apps.py wrote to the wrong field — the node returned its frozen
 * snapshot value and the LLM answered the OLD prompt instead of the
 * caller's new input.
 *
 * Keep in sync with backend/src/services/nodes/*.py invoke signatures.
 */
function defaultSlotForNode(nodeType: string | undefined): string {
  const t = (nodeType ?? '').toLowerCase()
  if (t.includes('text_input') || t.includes('text_output')) return 'text'
  if (t.includes('multimodal_input')) return 'text'
  if (t.includes('reference_audio') || t.includes('audio_input')) return 'audio'
  if (t.includes('image_input')) return 'image'
  // PrimitiveString-style legacy default
  return 'value'
}

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

  // Effect 1: close → wipe local state + reset mutation. ONLY depend on
  // open + resetMutation (resetMutation is stable across React Query
  // renders). Putting `nodes` here would cause this effect to refire
  // every render the parent passed a fresh `nodes` literal, which used
  // to feed an infinite-loop the manual gate caught.
  useEffect(() => {
    if (open) return
    setStep(1)
    setInputNodeIds([])
    setOutputNodeIds([])
    setName('')
    setLabel('')
    setCategory('app')
    resetMutation()
  }, [open, resetMutation])

  // Effect 2: when open and nodes exist, pick default I/O selections.
  // setState with identical arrays is a React no-op (bail-out via
  // Object.is on each entry would not bail; but the values are computed
  // deterministically from `nodes`, so re-running with the same `nodes`
  // contents produces equal-by-value arrays — React still bails on the
  // same reference only). To avoid superfluous setState calls when the
  // parent passes a fresh `nodes` literal each render, we depend on
  // `nodes` shallowly: same array length + same id sequence is good
  // enough.
  const nodesKey = nodes.map((n) => n.id).join('|')
  useEffect(() => {
    if (!open || nodes.length === 0) return
    const inputDefaults = nodes
      .filter((n) => /input|primitive|load/i.test(n.type ?? ''))
      .map((n) => n.id)
    const outputDefaults = nodes
      .filter((n) => /output|save|preview/i.test(n.type ?? ''))
      .map((n) => n.id)
    setInputNodeIds(inputDefaults)
    setOutputNodeIds(outputDefaults)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, nodesKey])

  const exposedInputs = useMemo<ExposedParam[]>(
    () =>
      inputNodeIds.map((id, i) => {
        const node = nodes.find((n) => n.id === id)
        return {
          node_id: id,
          key: `input_${i + 1}`,
          input_name: defaultSlotForNode(node?.type),
          type: 'string',
          required: true,
        }
      }),
    [inputNodeIds, nodes],
  )

  const exposedOutputs = useMemo<ExposedParam[]>(
    () =>
      outputNodeIds.map((id, i) => {
        const node = nodes.find((n) => n.id === id)
        return {
          node_id: id,
          key: `output_${i + 1}`,
          input_name: defaultSlotForNode(node?.type),
          type: 'string',
        }
      }),
    [outputNodeIds, nodes],
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
    <Modal onClose={onClose} title="发布为服务">
      <Stepper step={step} />

      {step === 1 && (
        <NodePicker
          kind="input"
          title="选输入节点"
          subtitle="caller 调 /run 时要填的字段，从这些节点的入参里取"
          nodes={nodes}
          selected={inputNodeIds}
          onToggle={(id) => toggle(setInputNodeIds, id)}
          onSetAll={(ids) => setInputNodeIds(ids)}
          hint="一般是 PrimitiveString / LoadImage / LoadAudio 这类入口节点"
        />
      )}

      {step === 2 && (
        <NodePicker
          kind="output"
          title="选输出节点"
          subtitle="执行结果会从这些节点读出来返回给 caller"
          nodes={nodes}
          selected={outputNodeIds}
          onToggle={(id) => toggle(setOutputNodeIds, id)}
          onSetAll={(ids) => setOutputNodeIds(ids)}
          hint="一般是 SaveVideo / SaveAudio / PreviewImage 这类出口节点"
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
  kind,
  title,
  subtitle,
  nodes,
  selected,
  onToggle,
  onSetAll,
  hint,
}: {
  kind: 'input' | 'output'
  title: string
  subtitle: string
  nodes: PublishDialogProps['nodes']
  selected: string[]
  onToggle: (id: string) => void
  onSetAll: (ids: string[]) => void
  hint: string
}) {
  const accent = kind === 'input' ? 'var(--info, #3b82f6)' : 'var(--ok, #34c759)'
  const allIds = nodes.map((n) => n.id)
  const allSelected = nodes.length > 0 && selected.length === nodes.length
  const noneSelected = selected.length === 0

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 4 }}>
        <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--text)' }}>{title}</span>
        <span style={{
          fontSize: 10, padding: '2px 7px', borderRadius: 10,
          background: kind === 'input' ? 'rgba(59,130,246,0.12)' : 'rgba(52,199,89,0.14)',
          color: accent, fontWeight: 500,
        }}>
          {selected.length} / {nodes.length}
        </span>
      </div>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6,
        fontSize: 11, color: 'var(--muted)', marginBottom: 12,
      }}>
        <Info size={12} style={{ flexShrink: 0 }} />
        <span>{subtitle}</span>
      </div>

      {nodes.length === 0 ? (
        <div style={{
          fontSize: 12, color: 'var(--muted)', padding: '32px 16px',
          textAlign: 'center', border: '1px dashed var(--border)', borderRadius: 8,
        }}>
          这个 workflow 还没有节点 · 先到画布加节点
        </div>
      ) : (
        <>
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '8px 12px', background: 'var(--bg)',
            border: '1px solid var(--border)', borderBottom: 'none',
            borderRadius: '8px 8px 0 0',
          }}>
            <span style={{ fontSize: 11, color: 'var(--muted)' }}>{hint}</span>
            <button
              type="button"
              onClick={() => onSetAll(allSelected ? [] : allIds)}
              style={{
                fontSize: 11, color: accent, background: 'transparent',
                border: 'none', cursor: 'pointer', padding: 0, fontWeight: 500,
              }}
            >
              {allSelected ? '全部取消' : noneSelected ? '全选' : '全选其余'}
            </button>
          </div>
          <div style={{
            maxHeight: 260, overflow: 'auto',
            border: '1px solid var(--border)', borderRadius: '0 0 8px 8px',
          }}>
            {nodes.map((n, i) => (
              <NodeRow
                key={n.id}
                node={n}
                accent={accent}
                isSelected={selected.includes(n.id)}
                isLast={i === nodes.length - 1}
                onToggle={() => onToggle(n.id)}
              />
            ))}
          </div>
        </>
      )}
    </div>
  )
}

function NodeRow({
  node,
  accent,
  isSelected,
  isLast,
  onToggle,
}: {
  node: PublishDialogProps['nodes'][number]
  accent: string
  isSelected: boolean
  isLast: boolean
  onToggle: () => void
}) {
  const [hover, setHover] = useState(false)
  const type = node.type ?? 'node'
  return (
    <label
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        display: 'flex', alignItems: 'center', gap: 10,
        padding: '10px 12px',
        borderBottom: isLast ? 'none' : '1px solid var(--border)',
        cursor: 'pointer',
        background: isSelected
          ? `linear-gradient(90deg, ${accent}1a 0%, ${accent}08 100%)`
          : hover ? 'var(--bg-accent, rgba(255,255,255,0.03))' : 'transparent',
        borderLeft: `3px solid ${isSelected ? accent : 'transparent'}`,
        transition: 'background 0.12s, border-color 0.12s',
      }}
    >
      {/* Custom checkbox */}
      <span style={{
        width: 16, height: 16, borderRadius: 4, flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: isSelected ? accent : 'transparent',
        border: `1.5px solid ${isSelected ? accent : 'var(--border)'}`,
        transition: 'background 0.12s, border-color 0.12s',
      }}>
        {isSelected && <Check size={11} strokeWidth={3} color="#fff" />}
      </span>

      {/* Type glyph */}
      <span style={{
        width: 22, height: 22, borderRadius: 4, flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'var(--bg)', border: '1px solid var(--border)',
        color: 'var(--muted)',
      }}>
        <Box size={12} />
      </span>

      <span style={{ fontSize: 13, color: 'var(--text)', fontWeight: 500 }}>{type}</span>

      <input type="checkbox" checked={isSelected} onChange={onToggle} style={{
        position: 'absolute', opacity: 0, pointerEvents: 'none',
      }} />

      <span style={{
        marginLeft: 'auto', fontSize: 10, padding: '2px 6px',
        background: 'var(--bg)', border: '1px solid var(--border)',
        borderRadius: 3, color: 'var(--muted)',
        fontFamily: 'var(--mono, ui-monospace, monospace)',
      }}>
        #{node.id.length > 10 ? `${node.id.slice(0, 8)}…` : node.id}
      </span>
    </label>
  )
}

function Stepper({ step }: { step: Step }) {
  const labels = ['选输入', '选输出', '命名发布'] as const
  return (
    <div style={{ display: 'flex', alignItems: 'center', marginBottom: 20 }}>
      {[1, 2, 3].map((s, i) => {
        const done = s < step
        const active = s === step
        return (
          <div key={s} style={{ display: 'flex', alignItems: 'center', flex: i < 2 ? 1 : 0 }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
              <div
                style={{
                  width: 26, height: 26, borderRadius: 13,
                  background: done || active ? 'var(--accent)' : 'transparent',
                  border: `1.5px solid ${done || active ? 'var(--accent)' : 'var(--border)'}`,
                  color: done || active ? '#fff' : 'var(--muted)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 12, fontWeight: 600,
                  boxShadow: active ? '0 0 0 4px color-mix(in oklab, var(--accent) 18%, transparent)' : 'none',
                  transition: 'all 0.15s',
                }}
              >
                {done ? <Check size={14} strokeWidth={3} /> : s}
              </div>
              <span style={{
                fontSize: 10,
                color: active ? 'var(--text)' : 'var(--muted)',
                fontWeight: active ? 600 : 400,
              }}>
                {labels[i]}
              </span>
            </div>
            {i < 2 && (
              <div style={{
                flex: 1, height: 2, margin: '0 8px',
                marginBottom: 18,
                background: done ? 'var(--accent)' : 'var(--border)',
                transition: 'background 0.15s',
              }} />
            )}
          </div>
        )
      })}
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
  // Portal to body — Topbar (the typical caller) sets backdrop-filter, which
  // creates a containing block that hijacks `position:fixed` children. Without
  // the portal the modal is centered inside the 36px-tall topbar instead of
  // the viewport, pushing the title + stepper above the visible area.
  return createPortal(
    <div
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.6)',
        backdropFilter: 'blur(4px)',
        WebkitBackdropFilter: 'blur(4px)',
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
          width: 600,
          maxWidth: '92vw',
          maxHeight: '88vh',
          overflow: 'auto',
          padding: 24,
          boxShadow: '0 20px 60px rgba(0,0,0,0.55)',
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
    </div>,
    document.body,
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

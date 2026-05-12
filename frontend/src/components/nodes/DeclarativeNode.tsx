import { useState, useEffect, useRef, useCallback } from 'react'
import { NodeResizer, type NodeProps } from '@xyflow/react'
import { Zap, Check, ArrowUp, ArrowDown, X, Plus, ImageIcon } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { useWorkspaceStore } from '../../stores/workspace'
import { NODE_DEFS, type NodeType } from '../../models/workflow'
import { DECLARATIVE_NODES, type WidgetDef } from '../../models/nodeRegistry'
import { useAgents } from '../../api/agents'
import { apiFetch } from '../../api/client'
import { useEnginesLiveSync, type EngineInfo } from '../../api/engines'
import { useLoras } from '../../api/loras'
import BaseNode, { NodeWidgetRow, NodeInput, NodeSelect, NodeNumberDrag, NodeTextarea } from './BaseNode'

function LoraSelectWidget({
  value,
  onChange,
}: {
  value: string
  onChange: (v: string) => void
}) {
  // V1' Lane C LoadLoRA component node uses this to pick a single LoRA
  // by display name (vs the lora_stack widget which manages an ordered
  // list with strengths for the integrated image_generate node). Source
  // is the same /api/v1/loras scanner endpoint that lora_stack reads —
  // so newly-dropped LoRA files appear in both without an edit.
  const { data: loras } = useLoras()
  return (
    <NodeSelect value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">— 不应用 LoRA —</option>
      {loras?.map((lora) => (
        <option key={lora.name} value={lora.name}>
          {lora.name}
        </option>
      ))}
    </NodeSelect>
  )
}


function AgentSelectWidget({
  value,
  onChange,
}: {
  value: string
  onChange: (v: string) => void
}) {
  const { data: agents } = useAgents()

  return (
    <NodeSelect value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">选择 Agent...</option>
      {agents?.map((a) => (
        <option key={a.name} value={a.name}>
          {a.display_name || a.name}
        </option>
      ))}
    </NodeSelect>
  )
}

interface LoraEntry {
  name: string
  strength: number
}

function LoraStackWidget({
  value,
  onChange,
}: {
  value: LoraEntry[]
  onChange: (v: LoraEntry[]) => void
}) {
  const { data: loras } = useLoras()
  const items = Array.isArray(value) ? value : []

  const update = (next: LoraEntry[]) => onChange(next)
  const add = () => update([...items, { name: '', strength: 1.0 }])
  const remove = (idx: number) => update(items.filter((_, i) => i !== idx))
  const move = (idx: number, dir: -1 | 1) => {
    const target = idx + dir
    if (target < 0 || target >= items.length) return
    const next = items.slice()
    ;[next[idx], next[target]] = [next[target], next[idx]]
    update(next)
  }
  const setName = (idx: number, name: string) => {
    const next = items.slice()
    next[idx] = { ...next[idx], name }
    update(next)
  }
  const setStrength = (idx: number, strength: number) => {
    const next = items.slice()
    next[idx] = { ...next[idx], strength }
    update(next)
  }

  const btnStyle: React.CSSProperties = {
    background: 'var(--bg-hover)',
    border: '1px solid var(--border)',
    borderRadius: 3,
    padding: '2px 4px',
    cursor: 'pointer',
    color: 'var(--muted)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, width: '100%' }}>
      {items.map((row, idx) => (
        <div key={idx} style={{ display: 'flex', gap: 3, alignItems: 'center' }}>
          <NodeSelect
            value={row.name}
            onChange={(e) => setName(idx, e.target.value)}
            style={{ flex: 1, minWidth: 0 }}
          >
            <option value="">选择 LoRA...</option>
            {(loras ?? []).map((l) => (
              <option key={l.name} value={l.name}>
                {l.name}
              </option>
            ))}
          </NodeSelect>
          <div style={{ width: 50 }}>
            <NodeNumberDrag
              value={row.strength}
              onChange={(v) => setStrength(idx, Number(v))}
              min={-2}
              max={2}
              step={0.1}
              precision={2}
            />
          </div>
          <button
            className="nodrag"
            type="button"
            aria-label={`上移 LoRA ${row.name || idx + 1}`}
            onClick={() => move(idx, -1)}
            style={btnStyle}
          >
            <ArrowUp size={10} />
          </button>
          <button
            className="nodrag"
            type="button"
            aria-label={`下移 LoRA ${row.name || idx + 1}`}
            onClick={() => move(idx, 1)}
            style={btnStyle}
          >
            <ArrowDown size={10} />
          </button>
          <button
            className="nodrag"
            type="button"
            aria-label={`删除 LoRA ${row.name || idx + 1}`}
            onClick={() => remove(idx)}
            style={{ ...btnStyle, color: 'var(--err)' }}
          >
            <X size={10} />
          </button>
        </div>
      ))}
      <button
        className="nodrag"
        type="button"
        onClick={add}
        style={{
          ...btnStyle,
          padding: '4px 6px',
          color: 'var(--muted)',
          fontSize: 10,
          gap: 4,
        }}
      >
        <Plus size={10} />
        添加 LoRA
      </button>
    </div>
  )
}

function ModelSelectWidget({
  value,
  onChange,
  filter,
}: {
  value: string
  onChange: (v: string) => void
  filter?: string
}) {
  // Subscribe to /ws/models so the dropdown stays current as models load /
  // unload, even when no other component on the page mounts useEngines().
  useEnginesLiveSync()
  const params = filter ? `?type=${filter}` : ''
  const { data: engines } = useQuery({
    queryKey: ['engines', filter],
    queryFn: () => apiFetch<EngineInfo[]>(`/api/v1/engines${params}`),
  })

  const loaded = (engines ?? []).filter((e) => e.status === 'loaded')
  const unloaded = (engines ?? []).filter((e) => e.status !== 'loaded' && e.local_exists)

  return (
    <NodeSelect value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">选择模型...</option>
      {loaded.map((e) => (
        <option key={e.name} value={e.name}>
          {e.display_name}
        </option>
      ))}
      {unloaded.length > 0 && loaded.length > 0 && (
        <option disabled>──── 未加载 ────</option>
      )}
      {unloaded.map((e) => (
        <option key={e.name} value={e.name} disabled>
          {e.display_name} (未加载)
        </option>
      ))}
    </NodeSelect>
  )
}

function resolveValue(value: unknown, widget: WidgetDef): unknown {
  if (value !== undefined && value !== null) return value
  return widget.default
}

function WidgetRenderer({
  widget,
  value,
  onChange,
}: {
  widget: WidgetDef
  value: unknown
  onChange: (v: unknown) => void
}) {
  const resolved = resolveValue(value, widget)

  switch (widget.widget) {
    case 'input':
      return (
        <NodeInput
          value={String(resolved ?? '')}
          onChange={(e) => onChange(e.target.value)}
          placeholder={widget.label}
        />
      )
    case 'textarea':
      return (
        <NodeTextarea
          value={String(resolved ?? '')}
          onChange={(e) => onChange(e.target.value)}
          style={widget.rows ? { height: widget.rows * 16 } : undefined}
        />
      )
    case 'select':
      return (
        <NodeSelect value={String(resolved ?? '')} onChange={(e) => onChange(e.target.value)}>
          {widget.options?.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </NodeSelect>
      )
    case 'slider':
      return (
        <NodeNumberDrag
          value={Number(resolved ?? widget.min ?? 0)}
          onChange={onChange}
          min={widget.min}
          max={widget.max}
          step={widget.step}
          precision={widget.precision}
        />
      )
    case 'checkbox':
      return (
        <div
          onClick={() => onChange(!resolved)}
          className="nodrag"
          style={{
            width: 32, height: 16, borderRadius: 8, cursor: 'pointer',
            background: resolved ? 'var(--accent)' : 'var(--bg)',
            border: '1px solid var(--border)',
            position: 'relative', transition: 'background 0.2s',
          }}
        >
          <div style={{
            width: 12, height: 12, borderRadius: 6,
            background: '#fff', position: 'absolute', top: 1,
            left: resolved ? 17 : 1, transition: 'left 0.2s',
          }} />
        </div>
      )
    case 'agent_select':
      return (
        <AgentSelectWidget
          value={String(resolved ?? '')}
          onChange={(v) => onChange(v)}
        />
      )
    case 'model_select':
      return (
        <ModelSelectWidget
          value={String(resolved ?? '')}
          onChange={(v) => onChange(v)}
          filter={widget.filter}
        />
      )
    case 'lora_stack':
      return (
        <LoraStackWidget
          value={Array.isArray(resolved) ? (resolved as LoraEntry[]) : []}
          onChange={(v) => onChange(v)}
        />
      )
    case 'lora_select':
      return (
        <LoraSelectWidget
          value={String(resolved ?? '')}
          onChange={(v) => onChange(v)}
        />
      )
    default:
      return null
  }
}

export default function DeclarativeNode({ id, type, data, selected }: NodeProps) {
  const updateNode = useWorkspaceStore((s) => s.updateNode)
  const nodeType = type as NodeType
  const declDef = DECLARATIVE_NODES[nodeType]
  const portDef = NODE_DEFS[nodeType]

  // Token stats state
  const [tokenStats, setTokenStats] = useState<{
    phase: 'streaming' | 'done'
    outputTokens: number
    inputTokens: number
    totalTokens: number
    tokensPerSec: number
    durationSec: number
  } | null>(null)
  const tokenCountRef = useRef(0)
  const firstTokenAtRef = useRef<number | null>(null)
  const throttleRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Image generation stage state (text_encode → denoise → vae_decode → done).
  // Backend image adapter doesn't emit per-step events yet; phase advances on
  // a known time budget driven by the node's `steps` config so the UI doesn't
  // sit silent for ~50s. V1 will replace the timer with real per-step events
  // once DiffusersImageBackend.infer_stream lands.
  const [imageStage, setImageStage] = useState<{
    phase: 'text_encode' | 'denoise' | 'vae_decode' | 'done'
    elapsedSec: number
  } | null>(null)
  const imageStageTimersRef = useRef<ReturnType<typeof setTimeout>[]>([])
  const imageStartAtRef = useRef<number | null>(null)

  const updateStreamingStats = useCallback(() => {
    const count = tokenCountRef.current
    const first = firstTokenAtRef.current
    if (count < 2 || !first) return
    const elapsed = (performance.now() - first) / 1000
    const rate = elapsed > 0 ? (count - 1) / elapsed : 0
    setTokenStats({
      phase: 'streaming',
      outputTokens: count,
      inputTokens: 0,
      totalTokens: 0,
      tokensPerSec: Math.round(rate * 10) / 10,
      durationSec: Math.round(elapsed * 10) / 10,
    })
  }, [])

  useEffect(() => {
    const handler = (event: CustomEvent) => {
      const data = event.detail
      if (data.type === 'node_stream' && data.node_id === id) {
        tokenCountRef.current++
        if (!firstTokenAtRef.current && tokenCountRef.current === 1) {
          firstTokenAtRef.current = performance.now()
        }
        if (!throttleRef.current) {
          throttleRef.current = setTimeout(() => {
            throttleRef.current = null
            updateStreamingStats()
          }, 250)
        }
      }
      if (data.type === 'node_start' && data.node_id === id) {
        // New run on this node — clear previous run's stats
        tokenCountRef.current = 0
        firstTokenAtRef.current = null
        setTokenStats(null)

        // Image-only: start the 3-stage simulation. text_encode (~1s) →
        // denoise (~stepsBudget) → vae_decode (~0.5s). Per-step time
        // estimate is conservative for cpu_offload single-card; V1 swaps
        // for real infer_stream events.
        if (nodeType === 'image_generate') {
          // Cancel any timers from a previous run on this node.
          for (const t of imageStageTimersRef.current) clearTimeout(t)
          imageStageTimersRef.current = []
          imageStartAtRef.current = performance.now()

          const steps = Math.max(1, Number(data.steps ?? 25))
          const perStepSec = 1.0  // ernie cpu_offload baseline (see PR-8 verify: 10 steps in ~50s)
          const denoiseSec = steps * perStepSec
          const textEncodeSec = 1.0

          setImageStage({ phase: 'text_encode', elapsedSec: 0 })
          imageStageTimersRef.current.push(
            setTimeout(() => setImageStage({ phase: 'denoise', elapsedSec: textEncodeSec }), textEncodeSec * 1000),
            setTimeout(
              () => setImageStage({ phase: 'vae_decode', elapsedSec: textEncodeSec + denoiseSec }),
              (textEncodeSec + denoiseSec) * 1000,
            ),
          )
        }
      }
      if (data.type === 'node_complete' && data.node_id === id) {
        if (throttleRef.current) {
          clearTimeout(throttleRef.current)
          throttleRef.current = null
        }
        // Image-only: stop the simulated stage timers and pin "done" with
        // the actual elapsed from backend duration_ms (or measured locally).
        if (nodeType === 'image_generate') {
          for (const t of imageStageTimersRef.current) clearTimeout(t)
          imageStageTimersRef.current = []
          const start = imageStartAtRef.current
          const realElapsed = data.duration_ms
            ? data.duration_ms / 1000
            : start
              ? (performance.now() - start) / 1000
              : 0
          setImageStage({ phase: 'done', elapsedSec: realElapsed })
          imageStartAtRef.current = null
        }
        const usage = data.usage
        const durationMs = data.duration_ms
        const first = firstTokenAtRef.current
        const elapsed = durationMs
          ? durationMs / 1000
          : first
            ? (performance.now() - first) / 1000
            : 0
        const outTok = usage?.completion_tokens ?? usage?.output_tokens ?? tokenCountRef.current
        const inTok = usage?.prompt_tokens ?? usage?.input_tokens ?? 0
        const total = usage?.total_tokens ?? inTok + outTok
        const rate = elapsed > 0 ? outTok / elapsed : 0
        setTokenStats({
          phase: 'done',
          outputTokens: outTok,
          inputTokens: inTok,
          totalTokens: total,
          tokensPerSec: Math.round(rate * 10) / 10,
          durationSec: Math.round(elapsed * 10) / 10,
        })
        tokenCountRef.current = 0
        firstTokenAtRef.current = null
        // Keep the final stats visible until the next run of this node
        // triggers node_start; no auto-hide timer.
      }
    }
    window.addEventListener('node-progress', handler as any)
    return () => {
      window.removeEventListener('node-progress', handler as any)
      if (throttleRef.current) clearTimeout(throttleRef.current)
      for (const t of imageStageTimersRef.current) clearTimeout(t)
      imageStageTimersRef.current = []
    }
  }, [id, nodeType, data.steps, updateStreamingStats])

  if (!declDef || !portDef) return null

  const handleResizeEnd = () => window.dispatchEvent(new Event('node-resize-end'))

  return (
    <>
    <NodeResizer
      isVisible={selected}
      minWidth={220}
      minHeight={80}
      onResizeEnd={handleResizeEnd}
      lineStyle={{ border: 'none' }}
      handleStyle={{ width: 12, height: 12, background: 'transparent', border: 'none' }}
    />
    <BaseNode
      title={declDef.label}
      badge={{
        label: declDef.badge,
        bg: `color-mix(in srgb, ${declDef.badgeColor} 15%, transparent)`,
        color: declDef.badgeColor,
      }}
      selected={selected}
      inputs={portDef.inputs}
      outputs={portDef.outputs}
    >
      {declDef.widgets.map((w) => (
        <NodeWidgetRow key={w.name} label={w.label} stretch={w.widget === 'textarea'}>
          <WidgetRenderer
            widget={w}
            value={data[w.name] as unknown}
            onChange={(v) => updateNode(id, { [w.name]: v })}
          />
        </NodeWidgetRow>
      ))}
      {/* Streaming text intentionally rendered ONLY in the downstream
          TextOutput node (data flows along edges). LLM node keeps only
          token stats below. */}
      {tokenStats && (
        <div
          className="flex items-center gap-1.5"
          style={{
            fontSize: 9,
            color: 'var(--muted)',
            padding: '4px 10px 6px',
            transition: 'opacity 0.5s',
            opacity: tokenStats.phase === 'done' ? 0.7 : 1,
          }}
        >
          {tokenStats.phase === 'streaming' ? (
            <Zap size={10} style={{ color: 'var(--warn)', flexShrink: 0 }} />
          ) : (
            <Check size={10} style={{ color: 'var(--ok)', flexShrink: 0 }} />
          )}
          {tokenStats.phase === 'streaming' ? (
            <span>
              生成中 · {tokenStats.tokensPerSec} tok/s · 输出 {tokenStats.outputTokens}
            </span>
          ) : (
            <span>
              输入 {tokenStats.inputTokens} · 输出 {tokenStats.outputTokens} · 合计 {tokenStats.totalTokens} · {tokenStats.tokensPerSec} tok/s · {tokenStats.durationSec}s
            </span>
          )}
        </div>
      )}
      {imageStage && (
        <div
          className="flex items-center gap-1.5"
          style={{
            fontSize: 9,
            color: 'var(--muted)',
            padding: '4px 10px 6px',
            transition: 'opacity 0.5s',
            opacity: imageStage.phase === 'done' ? 0.7 : 1,
          }}
        >
          {imageStage.phase === 'done' ? (
            <Check size={10} style={{ color: 'var(--ok)', flexShrink: 0 }} />
          ) : (
            <ImageIcon size={10} style={{ color: 'var(--info)', flexShrink: 0 }} />
          )}
          <span>
            {imageStage.phase === 'text_encode' && 'Text encode...'}
            {imageStage.phase === 'denoise' && `Denoise (~${Math.max(1, Math.round(Number(data.steps ?? 25)))} steps)...`}
            {imageStage.phase === 'vae_decode' && 'VAE decode...'}
            {imageStage.phase === 'done' && `完成 · ${Math.round(imageStage.elapsedSec * 10) / 10}s`}
          </span>
        </div>
      )}
    </BaseNode>
    </>
  )
}

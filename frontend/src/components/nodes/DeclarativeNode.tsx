import { useState, useEffect, useRef, useCallback } from 'react'
import { NodeResizer, type NodeProps } from '@xyflow/react'
import { Zap, Check } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { useWorkspaceStore } from '../../stores/workspace'
import { NODE_DEFS, type NodeType } from '../../models/workflow'
import { DECLARATIVE_NODES, type WidgetDef } from '../../models/nodeRegistry'
import { useAgents } from '../../api/agents'
import { apiFetch } from '../../api/client'
import type { EngineInfo } from '../../api/engines'
import BaseNode, { NodeWidgetRow, NodeInput, NodeSelect, NodeNumberDrag, NodeTextarea } from './BaseNode'

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

function ModelSelectWidget({
  value,
  onChange,
  filter,
}: {
  value: string
  onChange: (v: string) => void
  filter?: string
}) {
  const params = filter ? `?type=${filter}` : ''
  const { data: engines } = useQuery({
    queryKey: ['engines', filter],
    queryFn: () => apiFetch<EngineInfo[]>(`/api/v1/engines${params}`),
    refetchInterval: 10_000,
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
    default:
      return null
  }
}

export default function DeclarativeNode({ id, type, data, selected }: NodeProps) {
  const updateNode = useWorkspaceStore((s) => s.updateNode)
  const nodeType = type as NodeType
  const declDef = DECLARATIVE_NODES[nodeType]
  const portDef = NODE_DEFS[nodeType]

  const [streamText, setStreamText] = useState('')

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
  const fadeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

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
        setStreamText((prev) => prev + data.token)
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
      if (data.type === 'node_complete' && data.node_id === id) {
        setStreamText('')
        if (throttleRef.current) {
          clearTimeout(throttleRef.current)
          throttleRef.current = null
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
        if (fadeTimerRef.current) clearTimeout(fadeTimerRef.current)
        fadeTimerRef.current = setTimeout(() => setTokenStats(null), 8000)
      }
    }
    window.addEventListener('node-progress', handler as any)
    return () => {
      window.removeEventListener('node-progress', handler as any)
      if (throttleRef.current) clearTimeout(throttleRef.current)
      if (fadeTimerRef.current) clearTimeout(fadeTimerRef.current)
    }
  }, [id, updateStreamingStats])

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
      {streamText && (
        <div style={{
          padding: '6px 8px', margin: '4px 8px 8px', background: 'var(--bg)',
          borderRadius: 4, fontSize: 11, flex: 1, minHeight: 40, overflow: 'auto',
          whiteSpace: 'pre-wrap', color: 'var(--text-secondary)',
          border: '1px solid var(--border)',
        }}>
          {streamText}
          <span style={{ animation: 'blink 1s infinite' }}>▍</span>
        </div>
      )}
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
    </BaseNode>
    </>
  )
}

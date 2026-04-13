import { useState, useEffect } from 'react'
import { NodeResizer, type NodeProps } from '@xyflow/react'
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

  useEffect(() => {
    const handler = (event: CustomEvent) => {
      const data = event.detail
      if (data.type === 'node_stream' && data.node_id === id) {
        setStreamText((prev) => prev + data.token)
      }
      if (data.type === 'node_complete' && data.node_id === id) {
        setStreamText('')
      }
    }
    window.addEventListener('node-progress', handler as any)
    return () => window.removeEventListener('node-progress', handler as any)
  }, [id])

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
    </BaseNode>
    </>
  )
}

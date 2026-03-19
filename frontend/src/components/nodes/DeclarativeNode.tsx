import type { NodeProps } from '@xyflow/react'
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
  })

  return (
    <NodeSelect value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">选择模型...</option>
      {(engines ?? []).map((e) => (
        <option key={e.name} value={e.name} disabled={!e.local_exists}>
          {e.display_name} {e.local_exists ? '' : '(未下载)'}
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

  if (!declDef || !portDef) return null

  return (
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
        <NodeWidgetRow key={w.name} label={w.label}>
          <WidgetRenderer
            widget={w}
            value={data[w.name] as unknown}
            onChange={(v) => updateNode(id, { [w.name]: v })}
          />
        </NodeWidgetRow>
      ))}
    </BaseNode>
  )
}

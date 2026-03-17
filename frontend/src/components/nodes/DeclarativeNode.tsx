import type { NodeProps } from '@xyflow/react'
import { useWorkspaceStore } from '../../stores/workspace'
import { NODE_DEFS, type NodeType } from '../../models/workflow'
import { DECLARATIVE_NODES, type WidgetDef } from '../../models/nodeRegistry'
import { useAgents } from '../../api/agents'
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

function WidgetRenderer({
  widget,
  value,
  onChange,
}: {
  widget: WidgetDef
  value: unknown
  onChange: (v: unknown) => void
}) {
  switch (widget.widget) {
    case 'input':
      return (
        <NodeInput
          value={(value as string) ?? (widget.default as string) ?? ''}
          onChange={(e) => onChange(e.target.value)}
        />
      )
    case 'textarea':
      return (
        <NodeTextarea
          value={(value as string) ?? ''}
          onChange={(e) => onChange(e.target.value)}
          style={widget.rows ? { height: widget.rows * 16 } : undefined}
        />
      )
    case 'select':
      return (
        <NodeSelect value={(value as string) ?? ''} onChange={(e) => onChange(e.target.value)}>
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
          value={(value as number) ?? widget.min ?? 0}
          onChange={onChange}
          min={widget.min}
          max={widget.max}
          step={widget.step}
          precision={widget.precision}
        />
      )
    case 'agent_select':
      return (
        <AgentSelectWidget
          value={(value as string) ?? ''}
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

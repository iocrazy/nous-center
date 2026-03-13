import type { NodeProps } from '@xyflow/react'
import { useWorkspaceStore } from '../../stores/workspace'
import { NODE_DEFS } from '../../models/workflow'
import BaseNode, { NodeWidgetRow, NodeNumberDrag } from './BaseNode'

export default function ConcatNode({ id, data, selected }: NodeProps) {
  const updateNode = useWorkspaceStore((s) => s.updateNode)
  const def = NODE_DEFS.concat

  return (
    <BaseNode
      title={def.label}
      badge={{ label: 'Audio', bg: 'rgba(59,130,246,0.15)', color: 'var(--info)' }}
      selected={selected}
      inputs={def.inputs}
      outputs={def.outputs}
    >
      <NodeWidgetRow label="gap_ms">
        <NodeNumberDrag
          value={(data.gap_ms as number) ?? 500}
          onChange={(v) => updateNode(id, { gap_ms: v })}
          min={0}
          max={5000}
          step={50}
          precision={0}
        />
      </NodeWidgetRow>
    </BaseNode>
  )
}

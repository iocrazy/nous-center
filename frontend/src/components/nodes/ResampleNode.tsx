import type { NodeProps } from '@xyflow/react'
import { useWorkspaceStore } from '../../stores/workspace'
import { NODE_DEFS } from '../../models/workflow'
import BaseNode, { NodeWidgetRow, NodeSelect } from './BaseNode'

export default function ResampleNode({ id, data, selected }: NodeProps) {
  const updateNode = useWorkspaceStore((s) => s.updateNode)
  const def = NODE_DEFS.resample

  return (
    <BaseNode
      title={def.label}
      badge={{ label: 'Audio', bg: 'rgba(59,130,246,0.15)', color: 'var(--info)' }}
      selected={selected}
      inputs={def.inputs}
      outputs={def.outputs}
    >
      <NodeWidgetRow label="rate">
        <NodeSelect
          value={(data.target_rate as number) ?? 24000}
          onChange={(e) => updateNode(id, { target_rate: parseInt(e.target.value) })}
        >
          <option value={16000}>16000 Hz</option>
          <option value={22050}>22050 Hz</option>
          <option value={24000}>24000 Hz</option>
          <option value={44100}>44100 Hz</option>
          <option value={48000}>48000 Hz</option>
        </NodeSelect>
      </NodeWidgetRow>
    </BaseNode>
  )
}

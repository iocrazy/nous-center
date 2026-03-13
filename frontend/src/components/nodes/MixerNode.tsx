import type { NodeProps } from '@xyflow/react'
import { useWorkspaceStore } from '../../stores/workspace'
import { NODE_DEFS } from '../../models/workflow'
import BaseNode, { NodeWidgetRow, NodeNumberDrag } from './BaseNode'

export default function MixerNode({ id, data, selected }: NodeProps) {
  const updateNode = useWorkspaceStore((s) => s.updateNode)
  const def = NODE_DEFS.mixer

  return (
    <BaseNode
      title={def.label}
      badge={{ label: 'Audio', bg: 'rgba(59,130,246,0.15)', color: 'var(--info)' }}
      selected={selected}
      inputs={def.inputs}
      outputs={def.outputs}
    >
      <NodeWidgetRow label="vol_1">
        <NodeNumberDrag
          value={(data.volume_audio_1 as number) ?? 1.0}
          onChange={(v) => updateNode(id, { volume_audio_1: v })}
          min={0}
          max={1}
          step={0.1}
          precision={1}
        />
      </NodeWidgetRow>
      <NodeWidgetRow label="vol_2">
        <NodeNumberDrag
          value={(data.volume_audio_2 as number) ?? 1.0}
          onChange={(v) => updateNode(id, { volume_audio_2: v })}
          min={0}
          max={1}
          step={0.1}
          precision={1}
        />
      </NodeWidgetRow>
    </BaseNode>
  )
}

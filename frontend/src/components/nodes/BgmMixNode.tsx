import type { NodeProps } from '@xyflow/react'
import { useWorkspaceStore } from '../../stores/workspace'
import { NODE_DEFS } from '../../models/workflow'
import BaseNode, { NodeWidgetRow, NodeNumberDrag } from './BaseNode'

export default function BgmMixNode({ id, data, selected }: NodeProps) {
  const updateNode = useWorkspaceStore((s) => s.updateNode)
  const def = NODE_DEFS.bgm_mix

  return (
    <BaseNode
      title={def.label}
      badge={{ label: 'Audio', bg: 'rgba(59,130,246,0.15)', color: 'var(--info)' }}
      selected={selected}
      inputs={def.inputs}
      outputs={def.outputs}
    >
      <NodeWidgetRow label="bgm_vol">
        <NodeNumberDrag
          value={(data.bgm_volume as number) ?? 0.3}
          onChange={(v) => updateNode(id, { bgm_volume: v })}
          min={0}
          max={1}
          step={0.05}
          precision={2}
        />
      </NodeWidgetRow>
    </BaseNode>
  )
}

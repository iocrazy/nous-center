import type { NodeProps } from '@xyflow/react'
import { useWorkspaceStore } from '../../stores/workspace'
import { NODE_DEFS } from '../../models/workflow'
import BaseNode, { NodeWidgetRow } from './BaseNode'
import NodeSelectPopover from './NodeSelectPopover'

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
        <NodeSelectPopover
          value={String((data.target_rate as number) ?? 24000)}
          onChange={(v) => updateNode(id, { target_rate: parseInt(v) })}
          options={[16000, 22050, 24000, 44100, 48000].map((r) => ({ value: String(r), label: `${r} Hz` }))}
          size="compact"
        />
      </NodeWidgetRow>
    </BaseNode>
  )
}

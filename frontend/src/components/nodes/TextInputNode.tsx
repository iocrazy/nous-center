import type { NodeProps } from '@xyflow/react'
import { useWorkspaceStore } from '../../stores/workspace'
import { NODE_DEFS } from '../../models/workflow'
import BaseNode, { NodeWidgetRow, NodeTextarea } from './BaseNode'

export default function TextInputNode({ id, data, selected }: NodeProps) {
  const updateNode = useWorkspaceStore((s) => s.updateNode)
  const def = NODE_DEFS.text_input

  return (
    <BaseNode
      title={def.label}
      badge={{ label: 'TTS', bg: 'rgba(34,197,94,0.15)', color: 'var(--ok)' }}
      selected={selected}
      inputs={def.inputs}
      outputs={def.outputs}
    >
      <NodeWidgetRow label="text">
        <NodeTextarea
          value={(data.text as string) ?? ''}
          onChange={(e) => updateNode(id, { text: e.target.value })}
          placeholder="输入文本..."
        />
      </NodeWidgetRow>
    </BaseNode>
  )
}

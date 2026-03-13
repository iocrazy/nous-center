import type { NodeProps } from '@xyflow/react'
import { useWorkspaceStore } from '../../stores/workspace'
import { NODE_DEFS } from '../../models/workflow'
import BaseNode, { NodeWidgetRow, NodeInput } from './BaseNode'

export default function RefAudioNode({ id, data, selected }: NodeProps) {
  const updateNode = useWorkspaceStore((s) => s.updateNode)
  const def = NODE_DEFS.ref_audio

  return (
    <BaseNode
      title={def.label}
      badge={{ label: 'TTS', bg: 'var(--accent-2-subtle)', color: 'var(--accent-2)' }}
      selected={selected}
      inputs={def.inputs}
      outputs={def.outputs}
    >
      <NodeWidgetRow label="file">
        <NodeInput
          value={(data.path as string) ?? ''}
          onChange={(e) => updateNode(id, { path: e.target.value })}
          placeholder="音频路径或 URL"
        />
      </NodeWidgetRow>
      <NodeWidgetRow label="ref_text">
        <NodeInput
          value={(data.ref_text as string) ?? ''}
          onChange={(e) => updateNode(id, { ref_text: e.target.value })}
          placeholder="参考文本 (可选)"
        />
      </NodeWidgetRow>
    </BaseNode>
  )
}

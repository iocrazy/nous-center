import { useState } from 'react'
import type { NodeProps } from '@xyflow/react'
import { useWorkspaceStore } from '../../stores/workspace'
import { NODE_DEFS } from '../../models/workflow'
import BaseNode from './BaseNode'
import TextareaPortalEditor from './TextareaPortalEditor'

export default function TextInputNode({ id, data, selected }: NodeProps) {
  const updateNode = useWorkspaceStore((s) => s.updateNode)
  const def = NODE_DEFS.text_input
  const [editorOpen, setEditorOpen] = useState(false)
  const text = (data.text as string) ?? ''

  return (
    <BaseNode
      title={def.label}
      badge={{ label: 'IO', bg: 'rgba(59,130,246,0.15)', color: 'var(--info)' }}
      selected={selected}
      inputs={def.inputs}
      outputs={def.outputs}
    >
      <div style={{ padding: '4px 10px', position: 'relative' }}>
        <textarea
          className="nodrag nowheel"
          value={text}
          onChange={(e) => updateNode(id, { text: e.target.value })}
          onDoubleClick={() => setEditorOpen(true)}
          placeholder="输入文本…（双击展开编辑，避免画布变换下的输入法飘位）"
          rows={5}
          style={{
            width: '100%',
            minHeight: 100,
            padding: 8,
            borderRadius: 4,
            border: '1px solid var(--border)',
            background: 'var(--bg)',
            color: 'var(--text)',
            fontSize: 13,
            lineHeight: 1.5,
            resize: 'vertical',
            fontFamily: 'inherit',
            transform: 'translateZ(0)',
          }}
        />
      </div>
      <TextareaPortalEditor
        open={editorOpen}
        initialValue={text}
        title="编辑输入文本"
        onSave={(v) => {
          updateNode(id, { text: v })
          setEditorOpen(false)
        }}
        onCancel={() => setEditorOpen(false)}
      />
    </BaseNode>
  )
}

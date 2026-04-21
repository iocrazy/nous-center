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
        {/* Read-only preview. Clicking opens the portal editor — the inline
            <textarea> used to live here but it sits inside React Flow's
            transformed viewport, and Chromium + fcitx/ibus cannot reliably
            report the caret rect under a transformed ancestor, so Chinese IME
            candidate popups drift to the browser top bar. The portal editor
            renders at document.body → no transformed ancestors → IME works. */}
        <div
          className="nodrag nowheel"
          onClick={() => setEditorOpen(true)}
          title="点击编辑文本"
          style={{
            width: '100%',
            minHeight: 100,
            padding: 8,
            borderRadius: 4,
            border: '1px solid var(--border)',
            background: 'var(--bg)',
            color: text ? 'var(--text)' : 'var(--muted)',
            fontSize: 13,
            lineHeight: 1.5,
            fontFamily: 'inherit',
            cursor: 'text',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            overflow: 'auto',
            maxHeight: 200,
          }}
        >
          {text || '点击编辑文本…'}
        </div>
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

import { useState } from 'react'
import type { NodeProps } from '@xyflow/react'
import { Library } from 'lucide-react'
import { useWorkspaceStore } from '../../stores/workspace'
import { NODE_DEFS } from '../../models/workflow'
import BaseNode from './BaseNode'
import TextareaPortalEditor from './TextareaPortalEditor'
import {
  PROMPT_TEMPLATES,
  formatCharCount,
  isOverLimit,
  PROMPT_DEFAULT_MAX,
} from './promptLibrary'

// 对齐 Infinite-Canvas PROMPT 卡:工具行(提示库 pill + 字数计数)+ 提示词预览。
// 文本编辑仍走 TextareaPortalEditor(Linux fcitx IME 坑,见 BaseNode.NodeTextarea 注释)。
export default function TextInputNode({ id, data, selected }: NodeProps) {
  const updateNode = useWorkspaceStore((s) => s.updateNode)
  const def = NODE_DEFS.text_input
  const [editorOpen, setEditorOpen] = useState(false)
  const [libOpen, setLibOpen] = useState(false)
  const text = (data.text as string) ?? ''
  const over = isOverLimit(text.length, PROMPT_DEFAULT_MAX)

  return (
    <BaseNode
      title={def.label}
      badge={{ label: 'IO', bg: 'rgba(59,130,246,0.15)', color: 'var(--info)' }}
      selected={selected}
      inputs={def.inputs}
      outputs={def.outputs}
    >
      <div style={{ padding: '4px 10px 8px', position: 'relative' }}>
        {/* 工具行:提示库 pill + 字数计数 */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 5 }}>
          <button
            type="button"
            className="nodrag"
            onClick={() => setLibOpen((v) => !v)}
            title="提示库"
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 4,
              height: 22, padding: '0 8px', borderRadius: 9,
              border: '1px solid var(--border)',
              background: libOpen ? 'var(--accent-subtle)' : 'var(--bg)',
              color: libOpen ? 'var(--accent)' : 'var(--muted)',
              fontSize: 10, fontWeight: 600, cursor: 'pointer',
            }}
          >
            <Library size={12} />
            <span>提示库</span>
          </button>
          <span
            style={{
              marginLeft: 'auto', fontSize: 10, fontFamily: 'var(--mono)',
              color: over ? 'var(--accent)' : 'var(--muted)',
            }}
          >
            {formatCharCount(text.length)}
          </span>
        </div>

        {/* 提示库下拉:点选模板填入文本 */}
        {libOpen && (
          <div
            className="nodrag nowheel"
            style={{
              position: 'absolute', zIndex: 20, left: 10, right: 10, top: 30,
              maxHeight: 220, overflow: 'auto',
              background: 'var(--bg-elevated)', border: '1px solid var(--border)',
              borderRadius: 10, boxShadow: 'var(--shadow-lg)', padding: 4,
            }}
          >
            {PROMPT_TEMPLATES.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => { updateNode(id, { text: t.prompt }); setLibOpen(false) }}
                style={{
                  display: 'block', width: '100%', textAlign: 'left',
                  padding: '6px 8px', borderRadius: 7, border: 'none',
                  background: 'transparent', cursor: 'pointer',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-hover)')}
                onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
              >
                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text)' }}>{t.name}</div>
                <div style={{ fontSize: 9.5, color: 'var(--muted)', marginTop: 1 }}>{t.hint}</div>
              </button>
            ))}
          </div>
        )}

        {/* 提示词预览(点击 → portal 编辑器) */}
        <div
          className="nodrag nowheel"
          onClick={() => setEditorOpen(true)}
          title="点击编辑提示词"
          style={{
            width: '100%',
            minHeight: 100,
            padding: 8,
            borderRadius: 13,
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
          {text || '输入提示词...'}
        </div>
      </div>
      <TextareaPortalEditor
        open={editorOpen}
        initialValue={text}
        title="编辑提示词"
        onSave={(v) => {
          updateNode(id, { text: v })
          setEditorOpen(false)
        }}
        onCancel={() => setEditorOpen(false)}
      />
    </BaseNode>
  )
}

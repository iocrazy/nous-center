import type { NodeProps } from '@xyflow/react'
import { useWorkspaceStore } from '../../stores/workspace'
import { NODE_DEFS } from '../../models/workflow'
import BaseNode from './BaseNode'

export default function TextInputNode({ id, data, selected }: NodeProps) {
  const updateNode = useWorkspaceStore((s) => s.updateNode)
  const def = NODE_DEFS.text_input

  return (
    <BaseNode
      title={def.label}
      badge={{ label: 'IO', bg: 'rgba(59,130,246,0.15)', color: 'var(--info)' }}
      selected={selected}
      inputs={def.inputs}
      outputs={def.outputs}
    >
      <div style={{ padding: '4px 10px' }}>
        <textarea
          className="nodrag nowheel"
          value={(data.text as string) ?? ''}
          onChange={(e) => updateNode(id, { text: e.target.value })}
          placeholder="输入文本..."
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
            // IME fix: React Flow 的 viewport 用 CSS transform 平移/缩放，
            // Chromium 的输入法候选窗在 transformed 祖先下定位错误（飘到画布原点）。
            // 让 textarea 自己提升为合成层，IME 从 compositor 查屏幕坐标就对了。
            transform: 'translateZ(0)',
          }}
        />
      </div>
    </BaseNode>
  )
}

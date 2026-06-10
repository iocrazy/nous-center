// 应用编辑器里的只读节点卡片(spec 2026-06-09 PR-2 → 2026-06-10 弹窗化)。
// 卡片**纯展示**:标题 + 类目 badge + 「N 已用」计数 + 高亮。点击交互统一由
// WorkflowAppEditor 的 React Flow `onNodeClick` 处理(节点内 DOM onClick 在 React Flow
// 里不可靠 —— 真机验证逮到卡片 onClick 点不开弹窗)。非输出→开属性弹窗,输出→切「暴露为输出」。
import { memo } from 'react'
import { Handle, Position } from '@xyflow/react'

export interface AppEditorNodeData {
  label: string
  badge?: string
  badgeColor?: string
  /** 该节点已暴露的字段数(>0 显示「N 已用」+ 高亮边框)。 */
  exposedCount: number
  /** 该节点弹窗当前是否打开(高亮)。 */
  active?: boolean
  isOutput?: boolean
  outputChecked?: boolean
  [key: string]: unknown
}

function Box({ on }: { on: boolean }) {
  return (
    <span style={{
      width: 15, height: 15, borderRadius: 4, flexShrink: 0,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: on ? 'var(--accent)' : 'transparent',
      border: `1.5px solid ${on ? 'var(--accent)' : 'var(--border)'}`,
    }}>
      {on && <span style={{ color: '#fff', fontSize: 10, fontWeight: 900, lineHeight: 1 }}>✓</span>}
    </span>
  )
}

function AppEditorNodeImpl({ data }: { data: AppEditorNodeData }) {
  const color = data.badgeColor || 'var(--accent)'
  const exposed = data.isOutput ? !!data.outputChecked : data.exposedCount > 0
  const highlight = exposed || data.active
  return (
    <div
      title={data.isOutput ? '点击切换暴露为输出' : '点击配置暴露参数'}
      style={{
        width: 220,
        background: 'var(--card)',
        border: `1px solid ${highlight ? color : 'var(--border-strong, var(--border))'}`,
        borderRadius: 'var(--node-radius, 14px)',
        overflow: 'hidden',
        boxShadow: data.active
          ? `var(--shadow-md), 0 0 0 2px ${color}`
          : exposed
            ? `var(--shadow-md), 0 0 0 1px ${color}`
            : 'var(--shadow-md)',
        fontSize: 12,
        cursor: 'pointer',
      }}
    >
      <Handle type="target" position={Position.Left} style={{ background: 'var(--muted)', width: 10, height: 10, border: '2px solid var(--card)' }} />
      <Handle type="source" position={Position.Right} style={{ background: 'var(--muted)', width: 10, height: 10, border: '2px solid var(--card)' }} />

      {/* header — 彩色左条 + uppercase 标题 + 类目 badge(同 BaseNode) */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6,
        padding: '7px 10px',
        background: 'var(--card-hl)',
        borderRadius: 'var(--node-radius, 14px) var(--node-radius, 14px) 0 0',
        position: 'relative', overflow: 'hidden',
      }}>
        <span style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 3, background: color }} />
        <span style={{
          flex: 1, color: 'var(--text)', fontWeight: 700, fontSize: 10.5,
          textTransform: 'uppercase', letterSpacing: '0.06em',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>
          {data.label}
        </span>
        {data.badge && (
          <span style={{
            fontSize: 8, padding: '1px 5px', borderRadius: 3, flexShrink: 0,
            background: `color-mix(in srgb, ${color} 15%, transparent)`,
            color, fontWeight: 600,
          }}>
            {data.badge}
          </span>
        )}
      </div>

      {/* output node → 节点级「暴露为输出」(点击由 onNodeClick 处理,这里纯展示) */}
      {data.isOutput ? (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8, width: '100%',
          padding: '8px 10px', borderTop: '1px solid var(--border)', color: 'var(--text)',
        }}>
          <Box on={!!data.outputChecked} />
          <span>暴露为输出</span>
        </div>
      ) : (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6,
          padding: '7px 10px', borderTop: '1px solid var(--border)', color: 'var(--muted)',
        }}>
          {data.exposedCount > 0 ? (
            <span style={{
              fontSize: 10, fontWeight: 700, padding: '2px 7px', borderRadius: 999,
              background: `color-mix(in srgb, ${color} 16%, transparent)`, color,
            }}>
              {data.exposedCount} 已用
            </span>
          ) : (
            <span style={{ fontSize: 10.5 }}>点击配置</span>
          )}
        </div>
      )}
    </div>
  )
}

export default memo(AppEditorNodeImpl)

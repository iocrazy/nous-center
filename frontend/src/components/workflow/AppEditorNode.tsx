// 应用编辑器里的只读节点卡片(spec 2026-06-09 PR-2 → 2026-06-10 弹窗化)。
// 复刻 Infinite-Canvas:节点卡**整卡可点 → 弹出属性编辑窗**(逐 widget 勾选在弹窗里),
// 卡上只显示标题 + 类目 badge + 「N 已用」计数。输出节点保留节点级「暴露为输出」开关。
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
  /** 非输出节点:点卡片打开属性弹窗。 */
  onOpenPopup?: () => void
  isOutput?: boolean
  outputChecked?: boolean
  onToggleOutput?: () => void
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
      onClick={data.isOutput ? undefined : data.onOpenPopup}
      title={data.isOutput ? undefined : '点击配置暴露参数'}
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
        cursor: data.isOutput ? 'default' : 'pointer',
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

      {/* output node → 节点级「暴露为输出」 */}
      {data.isOutput ? (
        <button
          type="button"
          onClick={data.onToggleOutput}
          style={{
            display: 'flex', alignItems: 'center', gap: 8, width: '100%',
            padding: '8px 10px', background: 'transparent', border: 'none',
            borderTop: '1px solid var(--border)', cursor: 'pointer', color: 'var(--text)', textAlign: 'left',
          }}
        >
          <Box on={!!data.outputChecked} />
          <span>暴露为输出</span>
        </button>
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

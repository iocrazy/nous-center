// 应用编辑器里的只读节点卡片(spec 2026-06-09 PR-2)。沿用 nous 节点视觉
// (类目色条 + badge),body 把每个 widget 列成一行带 checkbox —— 勾中即暴露
// 到左侧表单。复刻 Infinite-Canvas 的「在节点上勾参数」交互。
import { memo } from 'react'
import { Handle, Position } from '@xyflow/react'
import { Check } from 'lucide-react'
import type { ExposableRow } from './appEditorSchema'
import { paramId } from './appEditorSchema'

export interface AppEditorNodeData {
  label: string
  badge?: string
  badgeColor?: string
  rows: ExposableRow[]
  checked: Set<string>
  onToggle: (input_name: string) => void
  isOutput?: boolean
  outputChecked?: boolean
  onToggleOutput?: () => void
  [key: string]: unknown
}

function Box({ on }: { on: boolean }) {
  return (
    <span
      style={{
        width: 15, height: 15, borderRadius: 4, flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: on ? 'var(--accent)' : 'transparent',
        border: `1.5px solid ${on ? 'var(--accent)' : 'var(--border)'}`,
        transition: 'background .12s, border-color .12s',
      }}
    >
      {on && <Check size={10} strokeWidth={3} color="#fff" />}
    </span>
  )
}

function AppEditorNodeImpl({ data }: { data: AppEditorNodeData }) {
  const color = data.badgeColor || 'var(--accent)'
  const anyExposed = data.rows.some((r) => data.checked.has(paramId(r.param))) || data.outputChecked
  return (
    <div
      style={{
        width: 240,
        // 对齐 BaseNode 的卡片 token(--card/--node-radius/--shadow-md),
        // 让节点框在画布上明显、且和真工作流节点风格一致(原来用 --bg-elevated
        // + 圆角8 在浅色下几乎和画布融在一起 → 用户反馈「节点框不明显」)。
        background: 'var(--card)',
        border: `1px solid ${anyExposed ? color : 'var(--border-strong, var(--border))'}`,
        borderRadius: 'var(--node-radius, 14px)',
        overflow: 'hidden',
        boxShadow: anyExposed
          ? `var(--shadow-md), 0 0 0 1px ${color}`
          : 'var(--shadow-md)',
        fontSize: 12,
      }}
    >
      <Handle type="target" position={Position.Left} style={{ background: 'var(--muted)', width: 10, height: 10, border: '2px solid var(--card)' }} />
      <Handle type="source" position={Position.Right} style={{ background: 'var(--muted)', width: 10, height: 10, border: '2px solid var(--card)' }} />

      {/* header — 彩色左条 + uppercase 标题 + badge(同 BaseNode) */}
      <div
        style={{
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '7px 10px', borderBottom: '1px solid var(--border)',
          background: 'var(--card-hl)',
          borderRadius: 'var(--node-radius, 14px) var(--node-radius, 14px) 0 0',
          position: 'relative', overflow: 'hidden',
        }}
      >
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
      {data.isOutput && (
        <button
          type="button"
          onClick={data.onToggleOutput}
          style={{
            display: 'flex', alignItems: 'center', gap: 8, width: '100%',
            padding: '8px 10px', background: 'transparent', border: 'none',
            cursor: 'pointer', color: 'var(--text)', textAlign: 'left',
          }}
        >
          <Box on={!!data.outputChecked} />
          <span>暴露为输出</span>
        </button>
      )}

      {/* widget rows */}
      {!data.isOutput && data.rows.length === 0 && (
        <div style={{ padding: '8px 10px', color: 'var(--muted)' }}>无可暴露参数</div>
      )}
      {!data.isOutput && data.rows.map((r) => {
        const on = data.checked.has(paramId(r.param))
        return (
          <button
            key={r.input_name}
            type="button"
            onClick={() => data.onToggle(r.input_name)}
            style={{
              display: 'flex', alignItems: 'center', gap: 8, width: '100%',
              padding: '7px 10px', background: on ? 'var(--accent-subtle, rgba(99,102,241,0.08))' : 'transparent',
              border: 'none', borderTop: '1px solid var(--border)', cursor: 'pointer',
              color: 'var(--text)', textAlign: 'left',
            }}
          >
            <Box on={on} />
            <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {r.label}
            </span>
            <span style={{
              fontSize: 9, padding: '1px 5px', borderRadius: 3,
              background: 'var(--bg)', border: '1px solid var(--border)',
              color: 'var(--muted)', fontFamily: 'var(--mono, monospace)',
            }}>
              {String(r.param.type)}
            </span>
          </button>
        )
      })}
    </div>
  )
}

export default memo(AppEditorNodeImpl)

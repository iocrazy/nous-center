/**
 * NodeCreateMenu — 画布上「在光标处建节点」的浮层菜单。复用于:
 *   - 端口拖到空白 → 兼容节点候选(借鉴 Infinite-Canvas 的 drag-from-port-create)
 *   - 画布右键 → 全部节点
 * 带搜索框、彩色点、键盘 ESC 关闭、点外关闭。位置为画布容器内像素(absolute)。
 */
import { useEffect, useRef, useState } from 'react'
import type { NodeChoice } from './nodeChoices'

export interface NodeCreateMenuProps {
  x: number
  y: number
  choices: NodeChoice[]
  title?: string
  onPick: (type: string) => void
  onClose: () => void
}

export default function NodeCreateMenu({ x, y, choices, title, onPick, onClose }: NodeCreateMenuProps) {
  const [search, setSearch] = useState('')
  const wrapRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    inputRef.current?.focus()
    const onDoc = (e: MouseEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) onClose()
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.stopPropagation(); onClose() }
    }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDoc)
      document.removeEventListener('keydown', onKey)
    }
  }, [onClose])

  const q = search.trim().toLowerCase()
  const shown = q ? choices.filter((c) => c.label.toLowerCase().includes(q) || c.type.toLowerCase().includes(q)) : choices

  return (
    <div
      ref={wrapRef}
      className="absolute z-40 flex flex-col"
      style={{
        top: y, left: x, width: 220, maxHeight: 320,
        background: 'var(--bg-elevated)', border: '1px solid var(--border)',
        borderRadius: 8, boxShadow: 'var(--shadow-md)', overflow: 'hidden',
      }}
    >
      {title && (
        <div style={{ padding: '6px 10px 2px', fontSize: 10, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: 0.5 }}>
          {title}
        </div>
      )}
      <input
        ref={inputRef}
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder="搜索节点…"
        className="nodrag"
        style={{
          margin: '6px 8px', padding: '4px 8px', fontSize: 12,
          background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 4,
          color: 'var(--text)', outline: 'none',
        }}
      />
      <div style={{ overflowY: 'auto', padding: '0 4px 6px' }} className="nowheel">
        {shown.length === 0 && (
          <div style={{ padding: '8px', fontSize: 11, color: 'var(--muted)', textAlign: 'center' }}>无匹配节点</div>
        )}
        {shown.map((c) => (
          <button
            key={c.type}
            type="button"
            onClick={() => { onPick(c.type); onClose() }}
            className="w-full flex items-center gap-2"
            style={{
              padding: '5px 8px', fontSize: 12, textAlign: 'left',
              background: 'transparent', border: 'none', borderRadius: 4,
              color: 'var(--text)', cursor: 'pointer',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-hover)')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
          >
            <span className="shrink-0 rounded-full" style={{ width: 7, height: 7, background: c.color }} />
            <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{c.label}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

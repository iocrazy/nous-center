/**
 * NodeSelectPopover — 对齐 paperclip `NewIssueDialog` 状态选择器风格的下拉。
 *
 * 替代原生 `<select>`,渲染为「触发按钮 + 浮层列表」:每项 可选彩色点(color) + 标签 + 副标题(description)
 * + 选中高亮(背景 var(--accent-subtle) + 左 Check)。Click-outside / ESC 关闭。
 *
 * 没用 radix/shadcn(nous 风格 = 自定义轻量组件,无新 dep);用 lucide-react Check/ChevronDown +
 * nous CSS 变量贴主题。WidgetDef.options 现支持 {value,label,description?,color?}。
 */
import { useEffect, useRef, useState } from 'react'
import { Check, ChevronDown } from 'lucide-react'

export interface SelectOption {
  value: string
  label: string
  description?: string
  color?: string
}

export interface NodeSelectPopoverProps {
  value: string
  onChange: (v: string) => void
  options: SelectOption[]
  placeholder?: string
  /** compact = 画布节点内联(更小 font/padding);normal = 属性面板。 */
  size?: 'compact' | 'normal'
  /** 触发按钮宽度,默认 100%。 */
  width?: string | number
}

export function NodeSelectPopover({
  value,
  onChange,
  options,
  placeholder,
  size = 'normal',
  width = '100%',
}: NodeSelectPopoverProps) {
  const [open, setOpen] = useState(false)
  const wrapRef = useRef<HTMLDivElement>(null)
  const current = options.find((o) => o.value === value)

  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDoc)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  const compact = size === 'compact'
  const fs = compact ? 10 : 12
  const triggerPad = compact ? '3px 8px' : '4px 10px'

  return (
    <div ref={wrapRef} className="nodrag" style={{ position: 'relative', width }}>
      <button
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={(e) => {
          e.preventDefault()
          setOpen((o) => !o)
        }}
        style={{
          width: '100%',
          padding: triggerPad,
          fontSize: fs,
          display: 'inline-flex',
          alignItems: 'center',
          gap: 6,
          background: 'var(--bg)',
          border: `1px solid ${open ? 'var(--border-strong)' : 'var(--border)'}`,
          borderRadius: 4,
          color: 'var(--text)',
          fontFamily: 'var(--font)',
          cursor: 'pointer',
          outline: 'none',
          transition: 'border-color 0.15s, background 0.15s',
        }}
        onMouseEnter={(e) => {
          if (!open) e.currentTarget.style.borderColor = 'var(--border-strong)'
        }}
        onMouseLeave={(e) => {
          if (!open) e.currentTarget.style.borderColor = 'var(--border)'
        }}
      >
        {current?.color && (
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: current.color,
              flexShrink: 0,
            }}
          />
        )}
        <span
          style={{
            flex: 1,
            textAlign: 'left',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {current?.label ?? placeholder ?? ''}
        </span>
        <ChevronDown size={compact ? 10 : 12} style={{ flexShrink: 0, color: 'var(--muted)' }} />
      </button>

      {open && (
        <div
          role="listbox"
          className="nowheel"
          style={{
            position: 'absolute',
            top: 'calc(100% + 4px)',
            left: 0,
            minWidth: '100%',
            maxHeight: 280,
            overflowY: 'auto',
            background: 'var(--bg-elevated)',
            border: '1px solid var(--border)',
            borderRadius: 6,
            boxShadow: 'var(--shadow-md)',
            padding: 4,
            zIndex: 1000,
          }}
        >
          {options.map((opt) => {
            const selected = opt.value === value
            return (
              <button
                key={opt.value}
                type="button"
                role="option"
                aria-selected={selected}
                onClick={() => {
                  onChange(opt.value)
                  setOpen(false)
                }}
                style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 8,
                  padding: '6px 8px',
                  background: selected ? 'var(--accent-subtle)' : 'transparent',
                  border: 'none',
                  borderRadius: 4,
                  cursor: 'pointer',
                  textAlign: 'left',
                  color: 'var(--text)',
                  fontFamily: 'var(--font)',
                  transition: 'background 0.1s',
                }}
                onMouseEnter={(e) => {
                  if (!selected) e.currentTarget.style.background = 'var(--bg-hover)'
                }}
                onMouseLeave={(e) => {
                  if (!selected) e.currentTarget.style.background = 'transparent'
                }}
              >
                <span
                  style={{
                    width: 14,
                    height: 14,
                    marginTop: 2,
                    flexShrink: 0,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  {opt.color ? (
                    <span
                      style={{ width: 8, height: 8, borderRadius: '50%', background: opt.color }}
                    />
                  ) : selected ? (
                    <Check size={12} style={{ color: 'var(--accent)' }} />
                  ) : null}
                </span>
                <span
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    lineHeight: 1.3,
                    flex: 1,
                    minWidth: 0,
                  }}
                >
                  <span style={{ fontSize: fs, color: 'var(--text)' }}>{opt.label}</span>
                  {opt.description && (
                    <span style={{ fontSize: 10, color: 'var(--muted)', marginTop: 1 }}>
                      {opt.description}
                    </span>
                  )}
                </span>
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}

export default NodeSelectPopover

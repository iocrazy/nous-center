import { useCallback, useRef } from 'react'
import { usePanelStore } from '../../stores/panel'

interface FloatingPanelProps {
  title: string
  children: React.ReactNode
  searchPlaceholder?: string
  onSearch?: (query: string) => void
  actions?: React.ReactNode
}

export default function FloatingPanel({ title, children, searchPlaceholder, onSearch, actions }: FloatingPanelProps) {
  const { panelWidth, setPanelWidth } = usePanelStore()
  const resizing = useRef(false)
  const startX = useRef(0)
  const startW = useRef(0)

  const onResizeStart = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault()
      resizing.current = true
      startX.current = e.clientX
      startW.current = panelWidth

      const onMove = (ev: MouseEvent) => {
        if (!resizing.current) return
        const delta = ev.clientX - startX.current
        setPanelWidth(startW.current + delta)
      }

      const onUp = () => {
        resizing.current = false
        document.removeEventListener('mousemove', onMove)
        document.removeEventListener('mouseup', onUp)
      }

      document.addEventListener('mousemove', onMove)
      document.addEventListener('mouseup', onUp)
    },
    [panelWidth, setPanelWidth],
  )

  return (
    <div
      className="absolute left-0 top-0 bottom-0 z-[15] flex flex-col"
      style={{
        width: panelWidth,
        background: 'var(--bg-elevated)',
        borderRight: '1px solid var(--border)',
        backdropFilter: 'blur(12px)',
        boxShadow: 'var(--shadow-md)',
      }}
    >
      {/* Header */}
      <div
        className="flex items-center gap-2 shrink-0"
        style={{
          padding: '10px 12px',
          borderBottom: '1px solid var(--border)',
          fontSize: 13,
          fontWeight: 600,
          color: 'var(--text-strong)',
        }}
      >
        <span>{title}</span>
        {actions && <div className="ml-auto">{actions}</div>}
      </div>

      {/* Search */}
      {searchPlaceholder && (
        <input
          className="shrink-0"
          placeholder={searchPlaceholder}
          onChange={(e) => onSearch?.(e.target.value)}
          style={{
            margin: 8,
            padding: '6px 10px',
            fontSize: 11,
            background: 'var(--bg)',
            border: '1px solid var(--border)',
            borderRadius: 5,
            color: 'var(--text)',
            fontFamily: 'var(--font)',
            outline: 'none',
          }}
        />
      )}

      {/* Body */}
      <div className="flex-1 overflow-y-auto" style={{ padding: '4px 8px' }}>
        {children}
      </div>

      {/* Resize handle */}
      <div
        onMouseDown={onResizeStart}
        className="absolute top-0 bottom-0"
        style={{
          right: -3,
          width: 6,
          cursor: 'col-resize',
          zIndex: 5,
        }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLDivElement).style.background = 'var(--accent-subtle)'
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLDivElement).style.background = 'transparent'
        }}
      />
    </div>
  )
}

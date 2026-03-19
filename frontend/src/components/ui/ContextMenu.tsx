import { useEffect, useRef, useState, useCallback } from 'react'

export interface MenuItem {
  label: string
  onClick?: () => void
  disabled?: boolean
  danger?: boolean
  divider?: boolean
  submenu?: MenuItem[]
}

export interface ContextMenuProps {
  items: MenuItem[]
  position: { x: number; y: number }
  onClose: () => void
}

export default function ContextMenu({ items, position, onClose }: ContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null)
  const [openSub, setOpenSub] = useState<number | null>(null)
  const [adjusted, setAdjusted] = useState(position)

  // Adjust position so menu doesn't overflow viewport
  useEffect(() => {
    const el = menuRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    let { x, y } = position
    if (x + rect.width > window.innerWidth) x = window.innerWidth - rect.width - 8
    if (y + rect.height > window.innerHeight) y = window.innerHeight - rect.height - 8
    if (x < 0) x = 8
    if (y < 0) y = 8
    setAdjusted({ x, y })
  }, [position])

  // Close on outside click or Escape
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    const handleClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose()
      }
    }
    document.addEventListener('keydown', handleKey)
    document.addEventListener('mousedown', handleClick)
    return () => {
      document.removeEventListener('keydown', handleKey)
      document.removeEventListener('mousedown', handleClick)
    }
  }, [onClose])

  const handleItemClick = useCallback(
    (item: MenuItem) => {
      if (item.disabled || item.divider) return
      item.onClick?.()
      onClose()
    },
    [onClose],
  )

  return (
    <div
      ref={menuRef}
      style={{
        position: 'fixed',
        left: adjusted.x,
        top: adjusted.y,
        zIndex: 9999,
        minWidth: 180,
        background: 'var(--card)',
        border: '1px solid var(--border)',
        borderRadius: 6,
        boxShadow: '0 8px 24px rgba(0,0,0,0.45)',
        padding: '4px 0',
        fontSize: 11,
      }}
    >
      {items.map((item, i) => {
        if (item.divider) {
          return (
            <div
              key={`div-${i}`}
              style={{
                height: 1,
                background: 'var(--border)',
                margin: '4px 0',
              }}
            />
          )
        }

        const hasSubmenu = item.submenu && item.submenu.length > 0

        return (
          <div
            key={i}
            onMouseEnter={() => hasSubmenu && setOpenSub(i)}
            onMouseLeave={() => hasSubmenu && setOpenSub(null)}
            onClick={() => !hasSubmenu && handleItemClick(item)}
            style={{
              position: 'relative',
              padding: '6px 12px',
              cursor: item.disabled ? 'default' : 'pointer',
              color: item.disabled
                ? 'var(--muted)'
                : item.danger
                  ? '#e55'
                  : 'var(--text)',
              opacity: item.disabled ? 0.5 : 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              background: 'transparent',
              transition: 'background 0.1s',
            }}
            onMouseOver={(e) => {
              if (!item.disabled) {
                ;(e.currentTarget as HTMLDivElement).style.background =
                  'color-mix(in srgb, var(--accent) 15%, transparent)'
              }
            }}
            onMouseOut={(e) => {
              ;(e.currentTarget as HTMLDivElement).style.background = 'transparent'
            }}
          >
            <span>{item.label}</span>
            {hasSubmenu && <span style={{ fontSize: 9, marginLeft: 8 }}>{'>'}</span>}

            {/* Submenu */}
            {hasSubmenu && openSub === i && (
              <div
                style={{
                  position: 'absolute',
                  left: '100%',
                  top: -4,
                  minWidth: 180,
                  background: 'var(--card)',
                  border: '1px solid var(--border)',
                  borderRadius: 6,
                  boxShadow: '0 8px 24px rgba(0,0,0,0.45)',
                  padding: '4px 0',
                  zIndex: 10000,
                }}
              >
                {item.submenu!.map((sub, si) => (
                  <div
                    key={si}
                    onClick={(e) => {
                      e.stopPropagation()
                      if (!sub.disabled) {
                        sub.onClick?.()
                        onClose()
                      }
                    }}
                    style={{
                      padding: '6px 12px',
                      cursor: sub.disabled ? 'default' : 'pointer',
                      color: sub.disabled ? 'var(--accent)' : 'var(--text)',
                      opacity: sub.disabled ? 0.7 : 1,
                      fontWeight: sub.disabled ? 600 : 400,
                      background: 'transparent',
                      transition: 'background 0.1s',
                    }}
                    onMouseOver={(e) => {
                      if (!sub.disabled) {
                        ;(e.currentTarget as HTMLDivElement).style.background =
                          'color-mix(in srgb, var(--accent) 15%, transparent)'
                      }
                    }}
                    onMouseOut={(e) => {
                      ;(e.currentTarget as HTMLDivElement).style.background = 'transparent'
                    }}
                  >
                    {sub.disabled && <span style={{ marginRight: 4 }}>●</span>}
                    {sub.label}
                  </div>
                ))}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

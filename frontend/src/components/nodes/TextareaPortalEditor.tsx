import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { X } from 'lucide-react'

/**
 * Full-screen textarea overlay for safe IME input on Linux + Chromium.
 *
 * Why a portal: React Flow renders nodes inside a CSS-transformed viewport.
 * Chromium on Linux cannot reliably report the caret rect to fcitx/ibus when
 * the focused <textarea> has a transformed ancestor — the candidate popup
 * falls back to the top bar and you can't pick words by position.
 *
 * Double-clicking a node's textarea opens this portal editor; the textarea
 * lives at document.body → no transformed ancestors → IME works natively.
 */
export default function TextareaPortalEditor({
  open,
  initialValue,
  title,
  onSave,
  onCancel,
}: {
  open: boolean
  initialValue: string
  title?: string
  onSave: (v: string) => void
  onCancel: () => void
}) {
  const [value, setValue] = useState(initialValue)
  const ref = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (open) {
      setValue(initialValue)
      setTimeout(() => ref.current?.focus(), 0)
    }
  }, [open, initialValue])

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCancel()
      // Ctrl/Cmd+Enter saves
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        onSave(value)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, value, onSave, onCancel])

  if (!open) return null

  return createPortal(
    <div
      onClick={onCancel}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.45)',
        zIndex: 9999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: 'var(--card)',
          border: '1px solid var(--border)',
          borderRadius: 8,
          boxShadow: 'var(--shadow-md)',
          width: 'min(720px, 90vw)',
          maxHeight: '80vh',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            padding: '10px 14px',
            borderBottom: '1px solid var(--border)',
          }}
        >
          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-strong)' }}>
            {title || '编辑文本'}
          </span>
          <span style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--muted)', marginRight: 8 }}>
            Ctrl+Enter 保存 · Esc 取消
          </span>
          <button
            onClick={onCancel}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: 2,
              color: 'var(--muted)',
              display: 'flex',
              alignItems: 'center',
            }}
          >
            <X size={16} />
          </button>
        </div>
        <textarea
          ref={ref}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          style={{
            flex: 1,
            minHeight: 320,
            padding: 14,
            border: 'none',
            background: 'var(--bg)',
            color: 'var(--text)',
            fontSize: 14,
            lineHeight: 1.6,
            fontFamily: 'inherit',
            resize: 'none',
            outline: 'none',
          }}
        />
        <div
          style={{
            display: 'flex',
            justifyContent: 'flex-end',
            gap: 8,
            padding: '10px 14px',
            borderTop: '1px solid var(--border)',
          }}
        >
          <button
            onClick={onCancel}
            style={{
              padding: '4px 12px',
              fontSize: 12,
              background: 'transparent',
              border: '1px solid var(--border)',
              borderRadius: 4,
              color: 'var(--text)',
              cursor: 'pointer',
            }}
          >
            取消
          </button>
          <button
            onClick={() => onSave(value)}
            style={{
              padding: '4px 12px',
              fontSize: 12,
              background: 'var(--accent)',
              border: '1px solid var(--accent)',
              borderRadius: 4,
              color: 'white',
              cursor: 'pointer',
            }}
          >
            保存
          </button>
        </div>
      </div>
    </div>,
    document.body,
  )
}

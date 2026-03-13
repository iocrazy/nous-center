import { useToastStore, type Toast } from '../../stores/toast'

const COLOR_MAP: Record<Toast['type'], string> = {
  success: 'var(--ok)',
  error: '#ef4444',
  info: 'var(--accent)',
}

export default function ToastContainer() {
  const { toasts, remove } = useToastStore()

  if (toasts.length === 0) return null

  return (
    <div
      className="fixed z-50 flex flex-col gap-2"
      style={{ bottom: 16, right: 16, maxWidth: 320 }}
    >
      {toasts.map((t) => (
        <div
          key={t.id}
          onClick={() => remove(t.id)}
          className="cursor-pointer"
          style={{
            padding: '8px 14px',
            fontSize: 11,
            borderRadius: 6,
            background: 'var(--bg-elevated)',
            border: `1px solid ${COLOR_MAP[t.type]}`,
            color: 'var(--text-strong)',
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
            animation: 'fadeIn 0.2s ease-out',
          }}
        >
          <span style={{ color: COLOR_MAP[t.type], marginRight: 6, fontWeight: 600 }}>
            {t.type === 'success' ? 'OK' : t.type === 'error' ? 'ERR' : 'i'}
          </span>
          {t.message}
        </div>
      ))}
    </div>
  )
}

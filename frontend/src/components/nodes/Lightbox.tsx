import { useEffect } from 'react'
import { ChevronLeft, ChevronRight, X } from 'lucide-react'
import { useLightboxStore } from '../../stores/lightbox'

// 全屏图片预览(对齐 Infinite-Canvas):←/→ 切上/下一张,Esc/点击空白关闭。
// 单例,挂在 NodeEditor;数据来自 useLightboxStore。
export default function Lightbox() {
  const open = useLightboxStore((s) => s.open)
  const images = useLightboxStore((s) => s.images)
  const index = useLightboxStore((s) => s.index)
  const close = useLightboxStore((s) => s.close)
  const next = useLightboxStore((s) => s.next)
  const prev = useLightboxStore((s) => s.prev)

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); close() }
      else if (e.key === 'ArrowRight') { e.preventDefault(); next() }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); prev() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, close, next, prev])

  if (!open || !images.length) return null
  const url = images[index]
  const multi = images.length > 1

  const navBtn = (onClick: () => void, side: 'left' | 'right', icon: React.ReactNode) => (
    <button
      type="button"
      onClick={(e) => { e.stopPropagation(); onClick() }}
      aria-label={side === 'left' ? '上一张' : '下一张'}
      style={{
        position: 'absolute', top: '50%', transform: 'translateY(-50%)', [side]: 18,
        width: 44, height: 44, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'rgba(0,0,0,0.5)', border: '1px solid rgba(255,255,255,0.25)', color: '#fff', cursor: 'pointer',
      } as React.CSSProperties}
    >
      {icon}
    </button>
  )

  return (
    <div
      onClick={close}
      style={{
        position: 'fixed', inset: 0, zIndex: 9999, background: 'rgba(0,0,0,0.85)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'zoom-out',
      }}
    >
      <img
        src={url}
        alt="preview"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: '92vw', maxHeight: '92vh', objectFit: 'contain', cursor: 'default' }}
      />

      <button
        type="button"
        onClick={(e) => { e.stopPropagation(); close() }}
        aria-label="关闭"
        style={{
          position: 'absolute', top: 18, right: 18, width: 40, height: 40, borderRadius: '50%',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: 'rgba(0,0,0,0.5)', border: '1px solid rgba(255,255,255,0.25)', color: '#fff', cursor: 'pointer',
        }}
      >
        <X size={20} />
      </button>

      {multi && (
        <>
          {navBtn(prev, 'left', <ChevronLeft size={24} />)}
          {navBtn(next, 'right', <ChevronRight size={24} />)}
          <div
            style={{
              position: 'absolute', bottom: 20, left: '50%', transform: 'translateX(-50%)',
              fontSize: 12, color: '#fff', fontFamily: 'var(--mono)',
              background: 'rgba(0,0,0,0.5)', padding: '3px 10px', borderRadius: 999,
            }}
          >
            {index + 1} / {images.length}
          </div>
        </>
      )}
    </div>
  )
}

import { useEffect, useRef, useState } from 'react'
import { ChevronLeft, ChevronRight, X } from 'lucide-react'
import { useLightboxStore } from '../../stores/lightbox'
import { IDENTITY, clampPan, zoomAt, type ZoomState } from './lightboxZoom'

// 全屏图片预览(对齐 Infinite-Canvas):←/→ 切上/下一张,Esc/点击空白关闭。
// 滚轮缩放(1–6x,光标为锚)+ 放大后拖拽平移 + 双击复位。
// 单例,挂在 NodeEditor;数据来自 useLightboxStore。

/** 可缩放/平移的图。靠父级 `key={index}` 切图时 remount 复位缩放(无需 effect 重置)。 */
function ZoomableImage({ src }: { src: string }) {
  const [zoom, setZoom] = useState<ZoomState>(IDENTITY)
  const [dragging, setDragging] = useState(false)
  const imgRef = useRef<HTMLImageElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const drag = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null)
  const zoomed = zoom.scale > 1

  // 滚轮缩放:native listener({passive:false})才能 preventDefault。
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const cx = e.clientX - window.innerWidth / 2
      const cy = e.clientY - window.innerHeight / 2
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
      setZoom((z) => {
        const zoomedSt = zoomAt(z, factor, cx, cy)
        const img = imgRef.current
        return img ? clampPan(zoomedSt, img.clientWidth, img.clientHeight) : zoomedSt
      })
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  // 拖拽平移(scale>1)。
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!drag.current) return
      const img = imgRef.current
      setZoom((z) => {
        const moved = { scale: z.scale, tx: drag.current!.tx + (e.clientX - drag.current!.x), ty: drag.current!.ty + (e.clientY - drag.current!.y) }
        return img ? clampPan(moved, img.clientWidth, img.clientHeight) : moved
      })
    }
    const onUp = () => { drag.current = null; setDragging(false) }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp) }
  }, [])

  return (
    <div ref={wrapRef} onClick={(e) => e.stopPropagation()} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <img
        ref={imgRef}
        src={src}
        alt="preview"
        draggable={false}
        onDoubleClick={(e) => { e.stopPropagation(); setZoom(IDENTITY) }}
        onMouseDown={(e) => {
          if (!zoomed) return
          e.preventDefault(); e.stopPropagation()
          drag.current = { x: e.clientX, y: e.clientY, tx: zoom.tx, ty: zoom.ty }
          setDragging(true)
        }}
        style={{
          maxWidth: '92vw', maxHeight: '92vh', objectFit: 'contain',
          transform: `translate(${zoom.tx}px, ${zoom.ty}px) scale(${zoom.scale})`,
          transformOrigin: 'center center',
          transition: dragging ? 'none' : 'transform 0.08s',
          cursor: zoomed ? (dragging ? 'grabbing' : 'grab') : 'default',
          willChange: 'transform',
        }}
      />
    </div>
  )
}

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
      {/* key=index → 切图 remount,缩放/平移自动复位 */}
      <ZoomableImage key={index} src={url} />

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

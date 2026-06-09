import { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import { NodeResizer, type NodeProps } from '@xyflow/react'
import { ImageOff } from 'lucide-react'
import { NODE_DEFS } from '../../models/workflow'
import { useWorkspaceStore } from '../../stores/workspace'
import { useLightboxStore } from '../../stores/lightbox'
import BaseNode from './BaseNode'

/** 图像对比节点 —— A/B 滑动对比(对标 rgthree Image Comparer)。
 * 两路上游 image 各自从 node_complete 的 image_url 捕获(同 ImageOutputNode 的单路模式,扩成两路);
 * 渲染:B 作底图,A 裁到分隔线左侧,拖竖直分隔线左露 A / 右露 B。 */
export default function ImageCompareNode({ id, data, selected }: NodeProps) {
  const def = NODE_DEFS.image_compare
  const tabs = useWorkspaceStore((s) => s.tabs)
  const activeTabId = useWorkspaceStore((s) => s.activeTabId)
  const updateNode = useWorkspaceStore((s) => s.updateNode)
  const openLightbox = useLightboxStore((s) => s.openFromUrl)

  // 两路上游 source 节点 id(按 targetHandle image_a / image_b 分)。
  const { srcA, srcB } = useMemo(() => {
    const wf = tabs.find((t) => t.id === activeTabId)?.workflow
    const edges = wf?.edges ?? []
    const a = edges.find((e) => e.target === id && e.targetHandle === 'image_a')?.source ?? null
    const b = edges.find((e) => e.target === id && e.targetHandle === 'image_b')?.source ?? null
    return { srcA: a, srcB: b }
  }, [tabs, activeTabId, id])

  const urlA = (data.image_a_url as string) || ''
  const urlB = (data.image_b_url as string) || ''

  // 捕获两路上游出图(node_complete 带 image_url)。
  useEffect(() => {
    const handler = (event: Event) => {
      const d = (event as CustomEvent).detail
      if (d?.type !== 'node_complete' || !d.image_url) return
      if (srcA && d.node_id === srcA) updateNode(id, { image_a_url: d.image_url })
      if (srcB && d.node_id === srcB) updateNode(id, { image_b_url: d.image_url })
    }
    window.addEventListener('node-progress', handler)
    return () => window.removeEventListener('node-progress', handler)
  }, [srcA, srcB, id, updateNode])

  // 分隔线位置(0-100%)。拖动更新。
  const [pos, setPos] = useState(50)
  const boxRef = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)

  const moveTo = useCallback((clientX: number) => {
    const rect = boxRef.current?.getBoundingClientRect()
    if (!rect || rect.width === 0) return
    const p = ((clientX - rect.left) / rect.width) * 100
    setPos(Math.max(0, Math.min(100, p)))
  }, [])

  useEffect(() => {
    const onMove = (e: PointerEvent) => { if (dragging.current) moveTo(e.clientX) }
    const onUp = () => { dragging.current = false }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    return () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
  }, [moveTo])

  const both = urlA && urlB
  const oneOnly = (urlA || urlB) && !both

  return (
    <>
      <NodeResizer
        isVisible={selected}
        minWidth={280}
        minHeight={300}
        onResizeEnd={() => window.dispatchEvent(new Event('node-resize-end'))}
        lineStyle={{ border: 'none' }}
        handleStyle={{ width: 12, height: 12, background: 'transparent', border: 'none' }}
      />
      <BaseNode
        title={def.label}
        badge={{ label: 'A/B', bg: 'rgba(59,130,246,0.15)', color: 'var(--info)' }}
        selected={selected}
        inputs={def.inputs}
        outputs={def.outputs}
      >
        <div style={{ padding: 8 }}>
          <div
            ref={boxRef}
            className="nodrag nowheel"
            style={{
              width: '100%',
              aspectRatio: '1 / 1',
              background: 'var(--bg)',
              borderRadius: 4,
              border: '1px solid var(--border)',
              position: 'relative',
              overflow: 'hidden',
              userSelect: 'none',
              cursor: both ? 'ew-resize' : 'default',
            }}
            onPointerDown={(e) => {
              if (!both) return
              dragging.current = true
              moveTo(e.clientX)
            }}
          >
            {both ? (
              <>
                {/* B 作底图(右侧露出) */}
                <img
                  src={urlB}
                  alt="B"
                  draggable={false}
                  style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'contain' }}
                />
                {/* A 裁到分隔线左侧(clip 右边 100-pos%) */}
                <img
                  src={urlA}
                  alt="A"
                  draggable={false}
                  style={{
                    position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'contain',
                    clipPath: `inset(0 ${100 - pos}% 0 0)`,
                  }}
                />
                {/* 分隔线 + 把手 */}
                <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${pos}%`, width: 2, background: '#fff', boxShadow: '0 0 4px rgba(0,0,0,0.6)', transform: 'translateX(-1px)', pointerEvents: 'none' }} />
                <div style={{ position: 'absolute', top: '50%', left: `${pos}%`, width: 20, height: 20, marginLeft: -10, marginTop: -10, borderRadius: '50%', background: '#fff', boxShadow: '0 0 4px rgba(0,0,0,0.6)', pointerEvents: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, color: '#333' }}>⇄</div>
                {/* A/B 角标 */}
                <span style={{ position: 'absolute', top: 4, left: 4, fontSize: 9, color: '#fff', background: 'rgba(0,0,0,0.5)', borderRadius: 3, padding: '1px 4px', pointerEvents: 'none' }}>A</span>
                <span style={{ position: 'absolute', top: 4, right: 4, fontSize: 9, color: '#fff', background: 'rgba(0,0,0,0.5)', borderRadius: 3, padding: '1px 4px', pointerEvents: 'none' }}>B</span>
              </>
            ) : oneOnly ? (
              <img
                src={urlA || urlB}
                alt="single"
                draggable={false}
                className="nodrag"
                onClick={() => { const u = urlA || urlB; if (u) openLightbox(u) }}
                style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'contain', cursor: 'zoom-in' }}
              />
            ) : (
              <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 4, color: 'var(--muted-strong)' }}>
                <ImageOff size={20} />
                <span style={{ fontSize: 10 }}>连接图像 A / B</span>
              </div>
            )}
          </div>
          {oneOnly && (
            <div style={{ fontSize: 9, color: 'var(--muted)', marginTop: 4 }}>
              只连了一路 —— 两路都连上才能滑动对比
            </div>
          )}
        </div>
      </BaseNode>
    </>
  )
}

/**
 * GroupLayer — ComfyUI 式节点分组「可视框」。groups 存在 workflow.groups(不入执行图,
 * 后端执行器只读 nodes/edges)。本层渲染在画布内,按 React Flow 当前 viewport(x/y/zoom)
 * 把 flow 坐标的框换算成屏幕坐标 → 跟随平移/缩放。
 *
 * 交互:
 * - 拖组头 = 移动框 + 框内节点(拖起时按 node.position 是否落在框内圈定)
 * - 右下角手柄 = 缩放框(不动节点)
 * - 双击标题 = 改名;组头右侧 X = 删除组(不删节点)
 *
 * pointer-events 策略:容器/框体 none(让节点照常可点),仅组头/手柄 auto。框绘制在
 * 节点下层(zIndex 低),组头一般在节点上方不被遮挡,可正常拖。
 */
import { useRef, useState, useCallback } from 'react'
import { useViewport } from '@xyflow/react'
import { X } from 'lucide-react'
import { useWorkspaceStore } from '../../stores/workspace'
import type { WorkflowGroup } from '../../models/workflow'

// 稳定空数组 —— zustand selector 不能每次返回新 `[]`(useSyncExternalStore 会判定
// snapshot 变化 → 无限重渲染)。fallback 放 selector 外、用模块级常量。
const EMPTY_GROUPS: WorkflowGroup[] = []

function hexA(hex: string, a: number): string {
  // #rrggbb → rgba();非 hex 直接回退原值(允许 CSS 变量色)。
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim())
  if (!m) return hex
  const n = parseInt(m[1], 16)
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`
}

interface DragState {
  groupId: string
  startX: number
  startY: number
  groupStart: { x: number; y: number; width: number; height: number }
  nodes: Array<{ id: string; x: number; y: number }>
  mode: 'move' | 'resize'
}

export default function GroupLayer() {
  const { x: vx, y: vy, zoom } = useViewport()
  const groups = useWorkspaceStore((s) => s.getActiveWorkflow().groups) ?? EMPTY_GROUPS
  const getActiveWorkflow = useWorkspaceStore((s) => s.getActiveWorkflow)
  const setWorkflow = useWorkspaceStore((s) => s.setWorkflow)
  const updateGroup = useWorkspaceStore((s) => s.updateGroup)
  const removeGroup = useWorkspaceStore((s) => s.removeGroup)
  const [editingId, setEditingId] = useState<string | null>(null)

  // 闭包里 window 监听拿不到最新 zoom → 用 ref。
  const zoomRef = useRef(zoom)
  zoomRef.current = zoom
  const dragRef = useRef<DragState | null>(null)

  const onPointerMove = useCallback((e: PointerEvent) => {
    const d = dragRef.current
    if (!d) return
    const z = zoomRef.current || 1
    const dx = (e.clientX - d.startX) / z
    const dy = (e.clientY - d.startY) / z
    const wf = getActiveWorkflow()
    if (d.mode === 'move') {
      const moved = new Map(d.nodes.map((n) => [n.id, n]))
      setWorkflow({
        ...wf,
        groups: (wf.groups ?? []).map((g) =>
          g.id === d.groupId ? { ...g, x: d.groupStart.x + dx, y: d.groupStart.y + dy } : g,
        ),
        nodes: wf.nodes.map((n) => {
          const sn = moved.get(n.id)
          return sn ? { ...n, position: { x: sn.x + dx, y: sn.y + dy } } : n
        }),
      })
    } else {
      setWorkflow({
        ...wf,
        groups: (wf.groups ?? []).map((g) =>
          g.id === d.groupId
            ? { ...g, width: Math.max(120, d.groupStart.width + dx), height: Math.max(80, d.groupStart.height + dy) }
            : g,
        ),
      })
    }
  }, [getActiveWorkflow, setWorkflow])

  // ref 持有自身,避免在 useCallback 体内按名引用自己(TDZ / no-use-before-define)。
  const onPointerUpRef = useRef<() => void>(() => {})
  const onPointerUp = useCallback(() => {
    dragRef.current = null
    window.removeEventListener('pointermove', onPointerMove)
    window.removeEventListener('pointerup', onPointerUpRef.current)
  }, [onPointerMove])
  onPointerUpRef.current = onPointerUp

  const startMove = (e: React.PointerEvent, g: typeof groups[number]) => {
    if (editingId === g.id) return
    e.stopPropagation()
    e.preventDefault()
    const wf = getActiveWorkflow()
    // 拖起时圈定框内节点(node.position 落在框矩形内)。
    const contained = wf.nodes.filter(
      (n) =>
        n.position.x >= g.x && n.position.x <= g.x + g.width &&
        n.position.y >= g.y && n.position.y <= g.y + g.height,
    )
    dragRef.current = {
      groupId: g.id, startX: e.clientX, startY: e.clientY,
      groupStart: { x: g.x, y: g.y, width: g.width, height: g.height },
      nodes: contained.map((n) => ({ id: n.id, x: n.position.x, y: n.position.y })),
      mode: 'move',
    }
    window.addEventListener('pointermove', onPointerMove)
    window.addEventListener('pointerup', onPointerUp)
  }

  const startResize = (e: React.PointerEvent, g: typeof groups[number]) => {
    e.stopPropagation()
    e.preventDefault()
    dragRef.current = {
      groupId: g.id, startX: e.clientX, startY: e.clientY,
      groupStart: { x: g.x, y: g.y, width: g.width, height: g.height },
      nodes: [], mode: 'resize',
    }
    window.addEventListener('pointermove', onPointerMove)
    window.addEventListener('pointerup', onPointerUp)
  }

  if (groups.length === 0) return null

  const headerH = 26 * zoom
  const fontSize = Math.min(20, Math.max(9, 13 * zoom))

  return (
    <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 1, overflow: 'hidden' }}>
      {groups.map((g) => {
        const left = g.x * zoom + vx
        const top = g.y * zoom + vy
        return (
          <div
            key={g.id}
            style={{
              position: 'absolute', left, top, width: g.width * zoom, height: g.height * zoom,
              border: `2px solid ${g.color}`, borderRadius: 8, background: hexA(g.color, 0.05),
              pointerEvents: 'none', boxSizing: 'border-box',
            }}
          >
            {/* 组头:拖动 / 双击改名 / 删除 */}
            <div
              onPointerDown={(e) => startMove(e, g)}
              onDoubleClick={(e) => { e.stopPropagation(); setEditingId(g.id) }}
              style={{
                height: headerH, display: 'flex', alignItems: 'center', gap: 6,
                padding: `0 ${6 * zoom}px`, background: hexA(g.color, 0.18),
                borderRadius: '6px 6px 0 0', pointerEvents: 'auto', cursor: 'move',
                overflow: 'hidden',
              }}
            >
              {editingId === g.id ? (
                <input
                  autoFocus
                  defaultValue={g.title}
                  onPointerDown={(e) => e.stopPropagation()}
                  onBlur={(e) => { updateGroup(g.id, { title: e.target.value || '分组' }); setEditingId(null) }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') { updateGroup(g.id, { title: (e.target as HTMLInputElement).value || '分组' }); setEditingId(null) }
                    if (e.key === 'Escape') setEditingId(null)
                  }}
                  style={{
                    flex: 1, minWidth: 0, fontSize, fontWeight: 600, color: 'var(--text)',
                    background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 3, padding: '1px 4px',
                  }}
                />
              ) : (
                <span style={{ flex: 1, minWidth: 0, fontSize, fontWeight: 600, color: 'var(--text-strong)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {g.title}
                </span>
              )}
              <button
                type="button"
                title="删除分组(不删节点)"
                onPointerDown={(e) => e.stopPropagation()}
                onClick={(e) => { e.stopPropagation(); removeGroup(g.id) }}
                style={{ display: 'flex', flexShrink: 0, padding: 2, background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--text)' }}
              >
                <X size={Math.min(16, Math.max(10, 13 * zoom))} />
              </button>
            </div>
            {/* 右下角缩放手柄 */}
            <div
              onPointerDown={(e) => startResize(e, g)}
              style={{
                position: 'absolute', right: 0, bottom: 0, width: 16, height: 16,
                pointerEvents: 'auto', cursor: 'nwse-resize',
                background: `linear-gradient(135deg, transparent 50%, ${g.color} 50%)`,
                borderRadius: '0 0 6px 0',
              }}
            />
          </div>
        )
      })}
    </div>
  )
}

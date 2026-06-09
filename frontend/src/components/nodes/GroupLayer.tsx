/**
 * GroupLayer — ComfyUI/Infinite-Canvas 式节点分组「可视框」。groups 存在 workflow.groups
 * (不入执行图,后端执行器只读 nodes/edges)。本层渲染在画布内,按 React Flow 当前
 * viewport(x/y/zoom)把 flow 坐标的框换算成屏幕坐标 → 跟随平移/缩放。
 *
 * 自适应(PR-4):有显式 g.nodeIds 的分组,框矩形按成员**实测包围盒**实时派生
 * (订阅 useNodes,成员拖动/缩放时框自动适配,对齐 Infinite-Canvas),不再依赖存储 x/y/w/h;
 * 头部显示「N张图片 · M个提示词 已成组」计数副标题。legacy 分组(无 nodeIds)回退到
 * 存储矩形 + 「按 position 落框内」隐式判定 + 右下角缩放手柄。
 *
 * 交互:
 * - 拖组头 = 移动框 + 成员节点(autoFit 按 nodeIds;legacy 按落框内)
 * - 右下角手柄 = 缩放框(仅 legacy;autoFit 由内容决定尺寸)
 * - 双击标题 = 改名;组头右侧 X = 删除组(不删节点)
 */
import { useRef, useState, useCallback } from 'react'
import { useViewport, useNodes } from '@xyflow/react'
import { X } from 'lucide-react'
import { useWorkspaceStore } from '../../stores/workspace'
import type { WorkflowGroup } from '../../models/workflow'
import { computeGroupBounds, countGroupMembers, groupSubtitle, type Rect } from './groupGeometry'

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
  const liveNodes = useNodes()
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

  // 实时节点矩形 + 类型表(派生自适应框 + 计数副标题用)。
  const nodeRectById = new Map<string, Rect & { type: string }>()
  for (const n of liveNodes) {
    const width = (n.measured?.width ?? (n.width as number | undefined) ?? (n.style?.width as number | undefined) ?? 320)
    const height = (n.measured?.height ?? (n.height as number | undefined) ?? 160)
    nodeRectById.set(n.id, { x: n.position.x, y: n.position.y, width, height, type: n.type ?? '' })
  }

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

  const startMove = (e: React.PointerEvent, g: WorkflowGroup, rect: Rect) => {
    if (editingId === g.id) return
    e.stopPropagation()
    e.preventDefault()
    const wf = getActiveWorkflow()
    // autoFit:按 nodeIds 圈成员;legacy:按 node.position 落框内。
    const memberSet = g.nodeIds?.length
      ? wf.nodes.filter((n) => g.nodeIds!.includes(n.id))
      : wf.nodes.filter(
          (n) =>
            n.position.x >= rect.x && n.position.x <= rect.x + rect.width &&
            n.position.y >= rect.y && n.position.y <= rect.y + rect.height,
        )
    dragRef.current = {
      groupId: g.id, startX: e.clientX, startY: e.clientY,
      groupStart: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
      nodes: memberSet.map((n) => ({ id: n.id, x: n.position.x, y: n.position.y })),
      mode: 'move',
    }
    window.addEventListener('pointermove', onPointerMove)
    window.addEventListener('pointerup', onPointerUp)
  }

  const startResize = (e: React.PointerEvent, g: WorkflowGroup, rect: Rect) => {
    e.stopPropagation()
    e.preventDefault()
    dragRef.current = {
      groupId: g.id, startX: e.clientX, startY: e.clientY,
      groupStart: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
      nodes: [], mode: 'resize',
    }
    window.addEventListener('pointermove', onPointerMove)
    window.addEventListener('pointerup', onPointerUp)
  }

  if (groups.length === 0) return null

  const headerH = 30 * zoom
  const titleSize = Math.min(16, Math.max(8, 11 * zoom))
  const subSize = Math.min(13, Math.max(7, 9 * zoom))

  return (
    <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 1, overflow: 'hidden' }}>
      {groups.map((g) => {
        // autoFit:框矩形 + 计数从成员实测派生;无成员或 legacy 回退存储矩形。
        const autoFit = !!g.nodeIds?.length
        const memberRects = autoFit
          ? g.nodeIds!.map((id) => nodeRectById.get(id)).filter(Boolean) as (Rect & { type: string })[]
          : []
        const rect: Rect = (autoFit && computeGroupBounds(memberRects)) || {
          x: g.x, y: g.y, width: g.width, height: g.height,
        }
        const subtitle = autoFit
          ? groupSubtitle(countGroupMembers(memberRects.map((m) => m.type)))
          : null

        const left = rect.x * zoom + vx
        const top = rect.y * zoom + vy
        return (
          <div
            key={g.id}
            style={{
              position: 'absolute', left, top, width: rect.width * zoom, height: rect.height * zoom,
              border: `1.5px solid ${hexA(g.color, 0.55)}`, borderRadius: 'var(--node-radius, 14px)',
              background: hexA(g.color, 0.05),
              pointerEvents: 'none', boxSizing: 'border-box', backdropFilter: 'blur(1px)',
            }}
          >
            {/* 组头:拖动 / 双击改名 / 删除 + 计数副标题 */}
            <div
              onPointerDown={(e) => startMove(e, g, rect)}
              onDoubleClick={(e) => { e.stopPropagation(); setEditingId(g.id) }}
              style={{
                minHeight: headerH, display: 'flex', alignItems: 'center', gap: 6,
                padding: `${3 * zoom}px ${8 * zoom}px`, background: hexA(g.color, 0.16),
                borderRadius: 'var(--node-radius, 14px) var(--node-radius, 14px) 0 0',
                pointerEvents: 'auto', cursor: 'move', overflow: 'hidden',
              }}
            >
              <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 1 }}>
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
                      width: '100%', fontSize: titleSize, fontWeight: 700, color: 'var(--text)',
                      background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 3, padding: '1px 4px',
                    }}
                  />
                ) : (
                  <span style={{
                    fontSize: titleSize, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase',
                    color: 'var(--text-strong)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                  }}>
                    {g.title}
                  </span>
                )}
                {subtitle && (
                  <span style={{ fontSize: subSize, color: 'var(--muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {subtitle}
                  </span>
                )}
              </div>
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
            {/* 右下角缩放手柄(仅 legacy;autoFit 框尺寸由内容决定) */}
            {!autoFit && (
              <div
                onPointerDown={(e) => startResize(e, g, rect)}
                style={{
                  position: 'absolute', right: 0, bottom: 0, width: 16, height: 16,
                  pointerEvents: 'auto', cursor: 'nwse-resize',
                  background: `linear-gradient(135deg, transparent 50%, ${g.color} 50%)`,
                  borderRadius: '0 0 6px 0',
                }}
              />
            )}
          </div>
        )
      })}
    </div>
  )
}

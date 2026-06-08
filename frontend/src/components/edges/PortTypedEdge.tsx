import { memo, useState } from 'react'
import { BaseEdge, EdgeLabelRenderer, getBezierPath, useReactFlow, type EdgeProps } from '@xyflow/react'
import { useExecutionStore } from '../../stores/execution'
import { useWorkspaceStore } from '../../stores/workspace'

// 自定义 React Flow edge(spec 2026-06-04 §2.2 连接精修):实线 bezier + 按 source
// 端口的 PortType 着色(颜色经 edge.data.color 由 NodeEditor 预算,复用端口圆点配色)。
// hover/选中提亮走 theme.css 的 `.react-flow__edge:hover` / `.selected`(纯 CSS,稳;
// BaseEdge interactionWidth 给宽 hit 区)。运行时**只对活跃边**(目标节点 running)走
// dasharray 流动动画 —— 大图避免全图动画卡顿(spec §4 风险)。替换默认细灰虚线。
// hover 时中点显 × 删除按钮(借鉴 Infinite-Canvas,比双击/右键删边更可发现)。
function PortTypedEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  target,
  markerEnd,
}: EdgeProps) {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  })
  const color = (data?.color as string | undefined) ?? 'var(--muted-strong)'
  const portType = data?.portType as string | undefined
  // selector 返回 boolean,只在该目标节点 running 态翻转时重渲染本边,开销可控。
  const active = useExecutionStore((s) => s.nodeStates[target] === 'running')
  const [hovered, setHovered] = useState(false)
  const { setEdges } = useReactFlow()

  const removeEdge = () => {
    setEdges((eds) => eds.filter((e) => e.id !== id))
    useWorkspaceStore.getState().removeEdge(id)
  }

  return (
    <>
      {/* 类型名 tooltip(作为 React Flow edge `<g>` 的子 `<title>`) */}
      {portType ? <title>{portType}</title> : null}
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        interactionWidth={20}
        style={{
          stroke: color,
          strokeWidth: active ? 2.6 : 2,
          opacity: active ? 1 : 0.9,
          strokeDasharray: active ? '7 5' : undefined,
          animation: active ? 'nous-edge-flow 0.5s linear infinite' : undefined,
          transition: 'stroke-width 0.12s ease, opacity 0.12s ease',
        }}
      />
      {/* 宽透明命中路径:捕获 hover(BaseEdge 内部 path 不便挂 handler)。 */}
      <path
        d={edgePath}
        fill="none"
        stroke="transparent"
        strokeWidth={20}
        style={{ pointerEvents: 'stroke', cursor: 'pointer' }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      />
      {hovered && (
        <EdgeLabelRenderer>
          <button
            type="button"
            className="nodrag nopan"
            title="删除连接"
            onMouseEnter={() => setHovered(true)}
            onMouseLeave={() => setHovered(false)}
            onClick={(e) => { e.stopPropagation(); removeEdge() }}
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              pointerEvents: 'all',
              width: 18,
              height: 18,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              borderRadius: '50%',
              background: 'var(--bg-elevated)',
              border: `1px solid ${color}`,
              color: 'var(--err, #ef4444)',
              fontSize: 13,
              lineHeight: 1,
              cursor: 'pointer',
              boxShadow: 'var(--shadow-md)',
            }}
          >
            ×
          </button>
        </EdgeLabelRenderer>
      )}
    </>
  )
}

export default memo(PortTypedEdge)

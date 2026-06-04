import { memo } from 'react'
import { BaseEdge, getBezierPath, type EdgeProps } from '@xyflow/react'
import { useExecutionStore } from '../../stores/execution'

// 自定义 React Flow edge(spec 2026-06-04 §2.2 连接精修):实线 bezier + 按 source
// 端口的 PortType 着色(颜色经 edge.data.color 由 NodeEditor 预算,复用端口圆点配色)。
// hover/选中提亮走 theme.css 的 `.react-flow__edge:hover` / `.selected`(纯 CSS,稳;
// BaseEdge interactionWidth 给宽 hit 区)。运行时**只对活跃边**(目标节点 running)走
// dasharray 流动动画 —— 大图避免全图动画卡顿(spec §4 风险)。替换默认细灰虚线。
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
  const [edgePath] = getBezierPath({
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
    </>
  )
}

export default memo(PortTypedEdge)

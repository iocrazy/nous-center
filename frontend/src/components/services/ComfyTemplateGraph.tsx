// ComfyTemplateEditor 的 React Flow 渲染层(Task 9)。独立成子组件,好处两个:
// 1) 让 ComfyTemplateEditor.test.tsx 能 vi.mock 掉这一层,不用在 jsdom 里真跑 xyflow
//    (对齐 WorkflowAppEditor 已验证可行,但这里节点数可能更多,拆出去更稳)。
// 2) 渲染逻辑(卡片/边)和编辑器状态逻辑(popover/表/保存)解耦。
// 卡片纯展示:标题=class_type,#id,已用则出 teal 徽标 + 描边高亮;点击统一走 React Flow
// 的 onNodeClick(同 AppEditorNode 的教训 —— 卡片内 DOM onClick 在 xyflow 里不可靠)。
import { memo, useMemo } from 'react'
import { ReactFlow, Background, Controls, type Edge, type Node, type NodeMouseHandler } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import type { ComfyLayoutEdge, ComfyLayoutNode } from './comfyGraphLayout'

const TEAL = 'rgb(20,184,166)'

interface ComfyCardData {
  classType: string
  nodeId: string
  usedCount: number
  active: boolean
  [key: string]: unknown
}

function ComfyNodeCardImpl({ data }: { data: ComfyCardData }) {
  const highlight = data.usedCount > 0 || data.active
  return (
    <div
      title="点击配置该节点的暴露字段"
      style={{
        width: 190,
        background: 'var(--card)',
        border: `1px solid ${highlight ? TEAL : 'var(--border-strong, var(--border))'}`,
        borderRadius: 'var(--node-radius, 14px)',
        overflow: 'hidden',
        boxShadow: data.active
          ? `var(--shadow-md), 0 0 0 2px ${TEAL}`
          : highlight
            ? `var(--shadow-md), 0 0 0 1px ${TEAL}`
            : 'var(--shadow-md)',
        fontSize: 12,
        cursor: 'pointer',
      }}
    >
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6, padding: '7px 10px',
        background: 'var(--card-hl)', borderRadius: 'var(--node-radius, 14px) var(--node-radius, 14px) 0 0',
        position: 'relative', overflow: 'hidden',
      }}>
        <span style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 3, background: TEAL }} />
        <span style={{
          flex: 1, color: 'var(--text)', fontWeight: 700, fontSize: 10.5,
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>
          {data.classType}
        </span>
      </div>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6,
        padding: '7px 10px', borderTop: '1px solid var(--border)', color: 'var(--muted)',
      }}>
        <code style={{ fontFamily: 'var(--mono, monospace)', fontSize: 10.5 }}>#{data.nodeId}</code>
        {data.usedCount > 0 ? (
          <span style={{
            fontSize: 10, fontWeight: 700, padding: '2px 7px', borderRadius: 999,
            background: 'color-mix(in srgb, ' + TEAL + ' 16%, transparent)', color: TEAL,
          }}>
            {data.usedCount} 已用
          </span>
        ) : (
          <span style={{ fontSize: 10.5 }}>点击配置</span>
        )}
      </div>
    </div>
  )
}

const ComfyNodeCard = memo(ComfyNodeCardImpl)
const nodeTypes = { comfyNode: ComfyNodeCard }

export interface ComfyTemplateGraphProps {
  nodes: ComfyLayoutNode[]
  edges: ComfyLayoutEdge[]
  activeNodeId: string | null
  onNodeClick: (nodeId: string) => void
}

export default function ComfyTemplateGraph({ nodes, edges, activeNodeId, onNodeClick }: ComfyTemplateGraphProps) {
  const rfNodes = useMemo<Node<ComfyCardData>[]>(
    () =>
      nodes.map((n) => ({
        id: n.id,
        type: 'comfyNode',
        position: { x: n.x, y: n.y },
        data: { classType: n.class_type, nodeId: n.id, usedCount: n.usedCount, active: n.id === activeNodeId },
      })),
    [nodes, activeNodeId],
  )

  const rfEdges = useMemo<Edge[]>(
    () =>
      edges.map((e, i) => ({
        id: `e${i}-${e.source}-${e.target}`,
        source: e.source,
        target: e.target,
        style: { stroke: 'var(--border)' },
      })),
    [edges],
  )

  const handleNodeClick: NodeMouseHandler = (_e, node) => onNodeClick(node.id)

  return (
    <ReactFlow
      nodes={rfNodes}
      edges={rfEdges}
      nodeTypes={nodeTypes}
      onNodeClick={handleNodeClick}
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable={false}
      fitView
      proOptions={{ hideAttribution: true }}
      style={{ background: 'var(--bg)' }}
    >
      <Background gap={24} size={1.4} color="var(--grid, var(--border))" />
      <Controls showInteractive={false} />
    </ReactFlow>
  )
}

// ComfyTemplateEditor 的 React Flow 渲染层(Task 9)。独立成子组件,好处两个:
// 1) 让 ComfyTemplateEditor.test.tsx 能 vi.mock 掉这一层,不用在 jsdom 里真跑 xyflow
//    (对齐 WorkflowAppEditor 已验证可行,但这里节点数可能更多,拆出去更稳)。
// 2) 渲染逻辑(卡片/边)和编辑器状态逻辑(popover/表/保存)解耦。
// 卡片纯展示:标题=class_type,#id,已用则出 accent-2 徽标 + 描边高亮,当前点开配置弹窗的
// 节点额外描 accent 高亮环(对齐 Infinite-Canvas 参考实现的 is-active 描边,用主题
// --accent 而不是写死颜色,浅色主题下也分得清)。点击统一走 React Flow 的 onNodeClick
// (同 AppEditorNode 的教训 —— 卡片内 DOM onClick 在 xyflow 里不可靠);同时把
// event.currentTarget(即 xyflow 的 .react-flow__node 包裹 div,已确认点击 handler 就绑在
// 它上面)透传给上层,ComfyTemplateEditor 拿它的 getBoundingClientRect() 算弹窗贴靠位置
// (popoverPlacement.ts),不用另外查 DOM。
import { memo, useMemo } from 'react'
import { ReactFlow, Background, Controls, type Edge, type Node, type NodeMouseHandler } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import type { ComfyLayoutEdge, ComfyLayoutNode } from './comfyGraphLayout'

const USED_COLOR = 'var(--accent-2)'
const ACTIVE_COLOR = 'var(--accent)'
// 同 ExposedSummaryTable 的 stale 行用的红色语义(var(--error, #ef4444))——「本机不存在
// 的模型引用」跟「指向已消失节点的映射」是同一类问题(工作流里冻结了别处的状态,本机
// 兑不了现),复用同一套错误配色,不另起一个新视觉语言。
const ERROR_COLOR = 'var(--error, #ef4444)'

interface ComfyCardData {
  classType: string
  nodeId: string
  usedCount: number
  active: boolean
  /** 该节点有原始输入的值是 combo/enum 但不在 object_info 取值表里(见
   *  workflowModelCheck.ts)——多半是导入了别的机器冻结下来的模型文件名/路径。 */
  invalidModel?: boolean
  [key: string]: unknown
}

function ComfyNodeCardImpl({ data }: { data: ComfyCardData }) {
  const used = data.usedCount > 0
  const invalidModel = !!data.invalidModel
  const ringColor = data.active ? ACTIVE_COLOR : invalidModel ? ERROR_COLOR : used ? USED_COLOR : null
  return (
    <div
      title="点击配置该节点的暴露字段"
      style={{
        width: 190,
        background: 'var(--card)',
        border: `1px solid ${ringColor ?? 'var(--border-strong, var(--border))'}`,
        borderRadius: 'var(--node-radius, 14px)',
        overflow: 'hidden',
        boxShadow: data.active
          ? `var(--shadow-md), 0 0 0 2px ${ACTIVE_COLOR}`
          : invalidModel
            ? `var(--shadow-md), 0 0 0 1px ${ERROR_COLOR}`
            : used
              ? `var(--shadow-md), 0 0 0 1px ${USED_COLOR}`
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
        <span style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 3, background: ringColor ?? 'var(--border-strong, var(--border))' }} />
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
        {invalidModel ? (
          <span
            title="有原始字段的值不在本机模型库里，点开检查/修正"
            style={{
              fontSize: 10, fontWeight: 700, padding: '2px 7px', borderRadius: 999,
              background: 'color-mix(in srgb, ' + ERROR_COLOR + ' 16%, transparent)', color: ERROR_COLOR,
            }}
          >
            模型引用无效
          </span>
        ) : used ? (
          <span style={{
            fontSize: 10, fontWeight: 700, padding: '2px 7px', borderRadius: 999,
            background: 'color-mix(in srgb, ' + USED_COLOR + ' 16%, transparent)', color: USED_COLOR,
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
  nodes: Array<ComfyLayoutNode & { invalidModel?: boolean }>
  edges: ComfyLayoutEdge[]
  activeNodeId: string | null
  /** nodeEl = 该节点在 xyflow 里的包裹 DOM(.react-flow__node),供上层算弹窗贴靠位置。 */
  onNodeClick: (nodeId: string, nodeEl: HTMLElement) => void
}

export default function ComfyTemplateGraph({ nodes, edges, activeNodeId, onNodeClick }: ComfyTemplateGraphProps) {
  const rfNodes = useMemo<Node<ComfyCardData>[]>(
    () =>
      nodes.map((n) => ({
        id: n.id,
        type: 'comfyNode',
        position: { x: n.x, y: n.y },
        data: {
          classType: n.class_type,
          nodeId: n.id,
          usedCount: n.usedCount,
          active: n.id === activeNodeId,
          invalidModel: n.invalidModel,
        },
      })),
    [nodes, activeNodeId],
  )

  const rfEdges = useMemo<Edge[]>(
    () =>
      edges.map((e, i) => ({
        id: `e${i}-${e.source}-${e.target}`,
        source: e.source,
        target: e.target,
        // 之前用 --border(近乎全透明的分隔线色),边基本看不见(用户反馈的一部分)。
        // 换成 --border-strong 加粗到 1.8,边渲染层在 xyflow 里天然排在节点层前面
        // (EdgeRenderer 在 DOM 里先于 NodeRenderer),不用额外 z-index 就在节点后面。
        style: { stroke: 'var(--border-strong)', strokeWidth: 1.8 },
      })),
    [edges],
  )

  const handleNodeClick: NodeMouseHandler = (e, node) => onNodeClick(node.id, e.currentTarget as HTMLElement)

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

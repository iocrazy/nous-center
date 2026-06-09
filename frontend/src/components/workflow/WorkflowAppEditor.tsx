// 工作流 → WebUI 应用编辑器(spec 2026-06-09 PR-2)。复刻 Infinite-Canvas 测试画布:
// 右侧只读 nous 节点图(节点上逐 widget 勾选暴露),左侧从勾选项实时生成的表单。
// 被发布弹窗(预览态)和服务详情页「应用编辑」tab(可运行)复用。
import { useCallback, useMemo, type ReactNode } from 'react'
import { ReactFlow, Background, Controls, type Edge, type Node } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import type { ExposedParam } from '../../api/services'
import { DECLARATIVE_NODES } from '../../models/nodeRegistry'
import SchemaDrivenForm from '../playground/SchemaDrivenForm'
import AppEditorNode, { type AppEditorNodeData } from './AppEditorNode'
import {
  dedupeKeys,
  exposableRowsFor,
  layeredLayout,
  paramId,
  type EditorNodeLike,
} from './appEditorSchema'

const nodeTypes = { appEditor: AppEditorNode }

const OUTPUT_RE = /output|save|preview/i

function isOutputNode(type: string): boolean {
  return OUTPUT_RE.test(type)
}

function defaultOutputSlot(type: string): string {
  const t = type.toLowerCase()
  if (t.includes('text')) return 'text'
  if (t.includes('audio')) return 'audio'
  // image outputs emit image_url (publish envelope 白名单)
  return 'image_url'
}

export interface AppEditorValue {
  inputs: ExposedParam[]
  outputs: ExposedParam[]
}

export interface WorkflowAppEditorProps {
  nodes: EditorNodeLike[]
  edges: Array<{ source: string; target: string }>
  value: AppEditorValue
  onChange: (v: AppEditorValue) => void
  /** true = 左表单可真跑(服务页 tab);false/缺省 = 预览态(发布弹窗,未发布)。 */
  runnable?: boolean
  running?: boolean
  onRun?: (values: Record<string, unknown>) => void
  /** 表单底部额外节点(如「运行后输出」展示),可选。 */
  formFooter?: ReactNode
}

export default function WorkflowAppEditor({
  nodes,
  edges,
  value,
  onChange,
  runnable,
  running,
  onRun,
  formFooter,
}: WorkflowAppEditorProps) {
  const checked = useMemo(
    () => new Set(value.inputs.map((p) => paramId(p))),
    [value.inputs],
  )
  const outputChecked = useMemo(
    () => new Set(value.outputs.map((p) => String(p.node_id))),
    [value.outputs],
  )

  const toggleInput = useCallback(
    (node: EditorNodeLike, inputName: string) => {
      const rows = exposableRowsFor(node)
      const row = rows.find((r) => r.input_name === inputName)
      if (!row) return
      const id = paramId(row.param)
      const exists = value.inputs.some((p) => paramId(p) === id)
      const next = exists
        ? value.inputs.filter((p) => paramId(p) !== id)
        : dedupeKeys([...value.inputs, row.param])
      onChange({ ...value, inputs: next })
    },
    [value, onChange],
  )

  const toggleOutput = useCallback(
    (node: EditorNodeLike) => {
      const exists = value.outputs.some((p) => String(p.node_id) === node.id)
      if (exists) {
        onChange({ ...value, outputs: value.outputs.filter((p) => String(p.node_id) !== node.id) })
        return
      }
      const slot = defaultOutputSlot(node.type)
      const param: ExposedParam = {
        node_id: node.id,
        key: `output_${value.outputs.length + 1}`,
        input_name: slot,
        label: DECLARATIVE_NODES[node.type]?.label || node.type,
        type: 'string',
      }
      onChange({ ...value, outputs: dedupeKeys([...value.outputs, param]) })
    },
    [value, onChange],
  )

  const rfNodes = useMemo<Node<AppEditorNodeData>[]>(() => {
    const auto = layeredLayout(nodes, edges)
    return nodes.map((n) => {
      const def = DECLARATIVE_NODES[n.type]
      const out = isOutputNode(n.type)
      return {
        id: n.id,
        type: 'appEditor',
        position: n.position ?? auto[n.id] ?? { x: 0, y: 0 },
        data: {
          label: def?.label || n.type,
          badge: def?.badge,
          badgeColor: def?.badgeColor,
          rows: out ? [] : exposableRowsFor(n),
          checked,
          onToggle: (inputName: string) => toggleInput(n, inputName),
          isOutput: out,
          outputChecked: outputChecked.has(n.id),
          onToggleOutput: () => toggleOutput(n),
        },
      }
    })
  }, [nodes, edges, checked, outputChecked, toggleInput, toggleOutput])

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

  return (
    <div style={{ display: 'flex', height: '100%', minHeight: 420 }}>
      {/* 左:实时表单 */}
      <div
        style={{
          width: 320, flexShrink: 0, display: 'flex', flexDirection: 'column',
          borderRight: '1px solid var(--border)', background: 'var(--bg-accent)',
        }}
      >
        <div style={{
          fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase',
          letterSpacing: 0.5, padding: '10px 16px', borderBottom: '1px solid var(--border)',
        }}>
          画布节点预览{runnable ? '' : ' · 预览'}
        </div>
        <div style={{ flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
          <SchemaDrivenForm
            inputs={value.inputs}
            submitting={running}
            submitLabel={runnable ? '▶ 运行测试' : '发布后可运行'}
            onSubmit={runnable && onRun ? onRun : () => {}}
          />
          {formFooter}
        </div>
      </div>

      {/* 右:只读节点图 */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <ReactFlow
          nodes={rfNodes}
          edges={rfEdges}
          nodeTypes={nodeTypes}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          fitView
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={16} color="var(--border)" />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
    </div>
  )
}

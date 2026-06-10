// 工作流 → WebUI 应用编辑器(spec 2026-06-09 PR-2 → 2026-06-10 节点弹窗化)。
// 复刻 Infinite-Canvas 测试画布:右侧只读 nous 节点图,**点节点 → 弹出属性编辑窗**
// (逐 widget 勾选 + 改名 + 控件类型),左侧从勾选项实时生成表单。
// 被发布弹窗(预览态)和服务详情页「应用编辑」tab(可运行)复用。
import { useCallback, useMemo, useState, type MouseEvent as ReactMouseEvent, type ReactNode } from 'react'
import { ReactFlow, Background, Controls, type Edge, type Node } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { ChevronUp, ChevronDown, X } from 'lucide-react'
import type { ExposedParam } from '../../api/services'
import { paramSlot } from '../../api/services'
import { DECLARATIVE_NODES } from '../../models/nodeRegistry'
import SchemaDrivenForm from '../playground/SchemaDrivenForm'
import AppEditorNode, { type AppEditorNodeData } from './AppEditorNode'
import AppEditorNodePopup from './AppEditorNodePopup'
import { applyKind, type WidgetKind } from './widgetKind'
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

// 输出参数的 type:让 Playground/SchemaDrivenOutput 直接按类型渲 <img>/<audio>
// (不必再靠值推断)。slot=image_url→image,audio→audio,text→string。
function outputTypeForSlot(slot: string): string {
  if (slot === 'image_url' || slot === 'image') return 'image'
  if (slot === 'audio') return 'audio'
  return 'string'
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
  const [popupNodeId, setPopupNodeId] = useState<string | null>(null)

  const outputChecked = useMemo(
    () => new Set(value.outputs.map((p) => String(p.node_id))),
    [value.outputs],
  )
  // 每节点已暴露字段数(节点卡「N 已用」badge)。
  const exposedCountByNode = useMemo(() => {
    const m = new Map<string, number>()
    for (const p of value.inputs) m.set(String(p.node_id), (m.get(String(p.node_id)) ?? 0) + 1)
    return m
  }, [value.inputs])

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
        type: outputTypeForSlot(slot),
      }
      onChange({ ...value, outputs: dedupeKeys([...value.outputs, param]) })
    },
    [value, onChange],
  )

  // 弹窗内按 input_name 改名 / 改控件类型(身份 = node_id::input_name)。
  const renameByInput = useCallback(
    (nodeId: string, inputName: string, label: string) => {
      const id = paramId({ node_id: nodeId, input_name: inputName })
      onChange({ ...value, inputs: value.inputs.map((p) => (paramId(p) === id ? { ...p, label } : p)) })
    },
    [value, onChange],
  )
  const changeKind = useCallback(
    (nodeId: string, inputName: string, kind: WidgetKind) => {
      const id = paramId({ node_id: nodeId, input_name: inputName })
      onChange({ ...value, inputs: value.inputs.map((p) => (paramId(p) === id ? applyKind(p, kind) : p)) })
    },
    [value, onChange],
  )

  // 左侧字段列表的改名 / 上下移 / 移除(按数组顺序渲表单)。
  const renameInput = useCallback(
    (i: number, label: string) =>
      onChange({ ...value, inputs: value.inputs.map((p, j) => (j === i ? { ...p, label } : p)) }),
    [value, onChange],
  )
  const moveInput = useCallback(
    (i: number, dir: -1 | 1) => {
      const j = i + dir
      if (j < 0 || j >= value.inputs.length) return
      const arr = [...value.inputs]
      ;[arr[i], arr[j]] = [arr[j], arr[i]]
      onChange({ ...value, inputs: arr })
    },
    [value, onChange],
  )
  const removeInput = useCallback(
    (i: number) => onChange({ ...value, inputs: value.inputs.filter((_, j) => j !== i) }),
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
          exposedCount: exposedCountByNode.get(n.id) ?? 0,
          active: popupNodeId === n.id,
          isOutput: out,
          outputChecked: outputChecked.has(n.id),
        },
      }
    })
  }, [nodes, edges, exposedCountByNode, popupNodeId, outputChecked])

  // 节点点击统一走 React Flow 的 onNodeClick(节点卡内 DOM onClick 在 React Flow
  // 里不可靠触发 —— 真机验证逮到:卡片 onClick 点不开弹窗。onNodeClick 是 React Flow
  // 自己派发的,稳定。输出节点→切「暴露为输出」,其余→开属性弹窗。
  const onNodeClick = useCallback(
    (_e: ReactMouseEvent, rfNode: Node) => {
      const n = nodes.find((x) => x.id === rfNode.id)
      if (!n) return
      if (isOutputNode(n.type)) toggleOutput(n)
      else setPopupNodeId(n.id)
    },
    [nodes, toggleOutput],
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

  // 当前打开弹窗的节点 + 它的可暴露行 + 已暴露映射。
  const popupNode = popupNodeId ? nodes.find((n) => n.id === popupNodeId) ?? null : null
  const popupRows = useMemo(() => (popupNode ? exposableRowsFor(popupNode) : []), [popupNode])
  const popupExposed = useMemo(() => {
    const m = new Map<string, ExposedParam>()
    if (popupNode) {
      for (const p of value.inputs) if (String(p.node_id) === popupNode.id) m.set(paramId(p), p)
    }
    return m
  }, [popupNode, value.inputs])

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
          {value.inputs.length > 0 && (
            <div style={{ padding: '10px 12px 4px', borderBottom: '1px solid var(--border)' }}>
              <div style={{ fontSize: 10, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 6 }}>
                字段 · 可改名/排序
              </div>
              {value.inputs.map((p, i) => (
                <div key={paramId(p)} style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 4 }}>
                  <div style={{ display: 'flex', flexDirection: 'column' }}>
                    <FieldBtn title="上移" disabled={i === 0} onClick={() => moveInput(i, -1)}><ChevronUp size={11} /></FieldBtn>
                    <FieldBtn title="下移" disabled={i === value.inputs.length - 1} onClick={() => moveInput(i, 1)}><ChevronDown size={11} /></FieldBtn>
                  </div>
                  <input
                    value={p.label ?? ''}
                    onChange={(e) => renameInput(i, e.target.value)}
                    placeholder={paramSlot(p) ?? 'field'}
                    title={`node=${p.node_id} · ${paramSlot(p) ?? ''}`}
                    style={{
                      flex: 1, minWidth: 0, fontSize: 12, padding: '4px 7px',
                      background: 'var(--bg)', color: 'var(--text)',
                      border: '1px solid var(--border)', borderRadius: 4,
                    }}
                  />
                  <FieldBtn title="移除暴露" onClick={() => removeInput(i)}><X size={12} /></FieldBtn>
                </div>
              ))}
            </div>
          )}
          <SchemaDrivenForm
            inputs={value.inputs}
            submitting={running}
            submitLabel={runnable ? '▶ 运行测试' : '发布后可运行'}
            onSubmit={runnable && onRun ? onRun : () => {}}
          />
          {formFooter}
        </div>
      </div>

      {/* 右:只读节点图 + 节点属性弹窗(position:relative 容纳弹窗 absolute 定位)。
          画布底色用 --bg(对齐真工作流画布 NodeEditor),让白色 --card 节点卡拉开对比。 */}
      <div style={{ flex: 1, minWidth: 0, position: 'relative' }}>
        <ReactFlow
          nodes={rfNodes}
          edges={rfEdges}
          nodeTypes={nodeTypes}
          onNodeClick={onNodeClick}
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
        {popupNode && (
          <AppEditorNodePopup
            title={DECLARATIVE_NODES[popupNode.type]?.label || popupNode.type}
            sub={`${popupNode.type} · #${popupNode.id}`}
            rows={popupRows}
            exposed={popupExposed}
            onToggle={(inputName) => toggleInput(popupNode, inputName)}
            onRename={(inputName, label) => renameByInput(popupNode.id, inputName, label)}
            onChangeKind={(inputName, kind) => changeKind(popupNode.id, inputName, kind)}
            onClose={() => setPopupNodeId(null)}
          />
        )}
      </div>
    </div>
  )
}

function FieldBtn({
  children,
  onClick,
  title,
  disabled,
}: {
  children: ReactNode
  onClick: () => void
  title: string
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      disabled={disabled}
      style={{
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
        width: 20, height: 18, padding: 0, flexShrink: 0,
        background: 'transparent', border: 'none', borderRadius: 3,
        color: 'var(--muted)', cursor: disabled ? 'default' : 'pointer',
        opacity: disabled ? 0.3 : 1,
      }}
    >
      {children}
    </button>
  )
}

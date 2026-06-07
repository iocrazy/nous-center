import { useCallback, useRef, useMemo, useEffect, useState } from 'react'
import { Copy, Ban, Trash2 } from 'lucide-react'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  type Connection,
  type Node,
  type Edge,
  type NodeChange,
  type EdgeChange,
  type ReactFlowInstance,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import { nodeTypes } from './nodeTypes'
import PortTypedEdge from '../edges/PortTypedEdge'
import { PORT_TYPE_COLORS } from './portColors'
import { useWorkspaceStore } from '../../stores/workspace'
import { usePanelStore } from '../../stores/panel'
import { useExecutionStore } from '../../stores/execution'
import { NODE_DEFS, type NodeType, type PortType, type WorkflowNode, type WorkflowEdge } from '../../models/workflow'
import { buildPastedGraph } from '../../utils/pasteGraph'
import NodeLibraryPanel from '../panels/NodeLibraryPanel'
import NodePropertyPanel from '../panels/NodePropertyPanel'
import WorkflowsPanel from '../panels/WorkflowsPanel'
import PresetsPanel from '../panels/PresetsPanel'
import { useSelectionStore } from '../../stores/selection'
import DashboardOverlay from '../overlays/DashboardOverlay'
import ModelsOverlay from '../overlays/ModelsOverlay'
import SettingsOverlay from '../overlays/SettingsOverlay'
import PresetDetailOverlay from '../overlays/PresetDetailOverlay'
import AgentManagementOverlay from '../overlays/AgentManagementOverlay'
import LogsOverlay from '../overlays/LogsOverlay'
import NodePackagesOverlay from '../overlays/NodePackagesOverlay'
import ServicesList from '../../pages/ServicesList'
import ServiceDetailRoute from '../../pages/ServiceDetailRoute'
import WorkflowsList from '../../pages/WorkflowsList'
import UsagePage from '../../pages/UsagePage'
import ApiKeysList from '../../pages/ApiKeysList'
import ApiKeyDetail from '../../pages/ApiKeyDetail'
function getPortType(nodeType: string, handleId: string | null | undefined): PortType | null {
  const def = NODE_DEFS[nodeType as NodeType]
  if (!def || !handleId) return null
  const allPorts = [...def.inputs, ...def.outputs]
  const port = allPorts.find((p) => p.id === handleId)
  return port?.type ?? null
}

const PANEL_MAP: Record<string, React.FC> = {
  nodes: NodeLibraryPanel,
  workflows: WorkflowsPanel,
  presets: PresetsPanel,
}

export default function NodeEditor() {
  const workflow = useWorkspaceStore((s) => s.getActiveWorkflow())
  const setWorkflow = useWorkspaceStore((s) => s.setWorkflow)
  const storeAddEdge = useWorkspaceStore((s) => s.addEdge)
  const storeRemoveEdge = useWorkspaceStore((s) => s.removeEdge)
  const storeAddNode = useWorkspaceStore((s) => s.addNode)
  const storeAddNodesWithEdges = useWorkspaceStore((s) => s.addNodesWithEdges)
  const storeRemoveNode = useWorkspaceStore((s) => s.removeNode)
  const updateNode = useWorkspaceStore((s) => s.updateNode)
  const undo = useWorkspaceStore((s) => s.undo)
  const redo = useWorkspaceStore((s) => s.redo)
  const { activePanel, activeOverlay, panelWidth } = usePanelStore()
  const nodeStates = useExecutionStore((s) => s.nodeStates)

  const reactFlowWrapper = useRef<HTMLDivElement>(null)
  const reactFlowInstance = useRef<ReactFlowInstance | null>(null)
  const setSelectedNodeId = useSelectionStore((s) => s.setSelectedNodeId)
  // 节点右键菜单(承载 旁路/复制/删除,让快捷键功能可发现)。坐标为画布容器内像素。
  const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number; nodeId: string } | null>(null)

  // Cmd/Ctrl+Z undo; Cmd/Ctrl+Shift+Z (or Ctrl+Y) redo. Swallow when the
  // event target is an input/textarea/contenteditable so native text edit
  // history keeps working inside the portal editor, node inputs, etc.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey
      if (!mod) return
      const tgt = e.target as HTMLElement | null
      if (tgt) {
        const tag = tgt.tagName
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tgt.isContentEditable) return
      }
      const k = e.key.toLowerCase()
      if (k === 'z' && !e.shiftKey) {
        e.preventDefault()
        undo()
      } else if ((k === 'z' && e.shiftKey) || k === 'y') {
        e.preventDefault()
        redo()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [undo, redo])

  const NODE_STATE_CLASS: Record<string, string> = {
    pending: 'node-pending',
    running: 'node-running',
    completed: 'node-completed',
    error: 'node-error',
  }

  const rfNodes: Node[] = useMemo(
    () =>
      workflow.nodes.map((n) => ({
        id: n.id,
        type: n.type,
        position: n.position,
        data: n.data,
        style: (n as any).style ?? { width: 320 },
        ...((n as any).width != null ? { width: (n as any).width } : {}),
        ...((n as any).height != null ? { height: (n as any).height } : {}),
        // 旁路态(node-bypassed)叠加在执行态之上 —— 旁路节点不会进入 running,
        // 故一般只会单独出现;两者都在时类名都挂,CSS 旁路优先(灰显)。
        className: [
          nodeStates[n.id] ? NODE_STATE_CLASS[nodeStates[n.id]] : '',
          n.data?.bypassed ? 'node-bypassed' : '',
        ].filter(Boolean).join(' ') || undefined,
      })),
    [workflow.nodes, nodeStates],
  )

  // 节点 id → type:给 PortTypedEdge 按 source 端口推 PortType 着色用。
  const nodeTypeById = useMemo(
    () => Object.fromEntries(workflow.nodes.map((n) => [n.id, n.type])) as Record<string, string>,
    [workflow.nodes],
  )

  const rfEdges: Edge[] = useMemo(
    () =>
      workflow.edges.map((e) => {
        // source 端口类型 → 颜色(复用端口圆点配色)。推不出则回退 muted。
        const portType = getPortType(nodeTypeById[e.source] ?? '', e.sourceHandle)
        const color = portType
          ? PORT_TYPE_COLORS[portType] ?? 'var(--muted-strong)'
          : 'var(--muted-strong)'
        return {
          id: e.id,
          source: e.source,
          sourceHandle: e.sourceHandle,
          target: e.target,
          targetHandle: e.targetHandle,
          type: 'portTyped',
          data: { color, portType },
        }
      }),
    [workflow.edges, nodeTypeById],
  )

  const edgeTypes = useMemo(() => ({ portTyped: PortTypedEdge }), [])

  const [nodes, setNodes, onNodesChangeInternal] = useNodesState(rfNodes)
  const [edges, setEdges, onEdgesChangeInternal] = useEdgesState(rfEdges)

  // Keep a ref to always access latest React Flow nodes/edges (avoids stale closures)
  const nodesRef = useRef(nodes)
  nodesRef.current = nodes
  const edgesRef = useRef(edges)
  edgesRef.current = edges

  // 复制/粘贴剪贴板(模块外不共享 —— 仅本会话内存;跨 tab 粘贴可用,因为是 ref
  // 持有的纯数据)。pasteSeq 让连续粘贴递增偏移,避免叠在同一处。
  const clipboardRef = useRef<{
    nodes: Array<{ id: string; type: string; data: Record<string, unknown>; position: { x: number; y: number }; style?: unknown; width?: number; height?: number }>
    edges: Array<{ source: string; sourceHandle: string; target: string; targetHandle: string }>
  } | null>(null)
  const pasteSeqRef = useRef(0)

  // Sync Zustand store changes back to React Flow, preserving resize dimensions
  // AND selection state. rfNodes 从 store 重建时不带 `selected`,若不在这里保留,
  // 任何 store 更新(如在右侧属性面板编辑字段触发的 updateNode)都会让 RF 丢掉选中
  // → onSelectionChange([]) → 属性面板退回空态(面板"消失",无法持续编辑)。保留
  // `selected`/`dragging` 让选中跨 store 更新稳定,面板可固定编辑。
  useEffect(() => {
    setNodes((prev) =>
      rfNodes.map((rfn) => {
        const existing = prev.find((p) => p.id === rfn.id)
        if (!existing) return rfn
        // Preserve React Flow's resize state (width/height/style set by NodeResizer)
        // + 交互态(selected/dragging),否则属性面板编辑会清掉选中。
        return {
          ...rfn,
          ...(existing.width != null ? { width: existing.width } : {}),
          ...(existing.height != null ? { height: existing.height } : {}),
          ...(existing.style ? { style: existing.style } : {}),
          selected: existing.selected,
          ...(existing.dragging != null ? { dragging: existing.dragging } : {}),
        }
      }),
    )
  }, [rfNodes, setNodes])
  // round5:跟节点同理(#237 修过节点、边漏了)——`setEdges(rfEdges)` 全量覆盖会把
  // 用户单击高亮的边(onEdgeClick 写的 selected/accent style)在任意 store 更新(改字段
  // → updateNode → rfEdges 重算)后冲回 muted 默认,边高亮「闪一下就没」。保留 selected/style。
  useEffect(() => {
    setEdges((prev) =>
      rfEdges.map((rfe) => {
        const existing = prev.find((p) => p.id === rfe.id)
        if (!existing) return rfe
        return { ...rfe, selected: existing.selected, ...(existing.style ? { style: existing.style } : {}) }
      }),
    )
  }, [rfEdges, setEdges])

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      onNodesChangeInternal(changes)
      for (const change of changes) {
        if (change.type === 'remove') storeRemoveNode(change.id)
      }
    },
    [onNodesChangeInternal, storeRemoveNode],
  )

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      onEdgesChangeInternal(changes)
      for (const change of changes) {
        if (change.type === 'remove') storeRemoveEdge(change.id)
      }
    },
    [onEdgesChangeInternal, storeRemoveEdge],
  )

  const onConnect = useCallback(
    (params: Connection) => {
      setEdges((eds) => addEdge({ ...params, type: 'portTyped' }, eds))
      const edgeId = crypto.randomUUID().slice(0, 8)
      storeAddEdge({
        id: edgeId,
        source: params.source,
        sourceHandle: params.sourceHandle ?? '',
        target: params.target,
        targetHandle: params.targetHandle ?? '',
      })
    },
    [setEdges, storeAddEdge],
  )

  const isValidConnection = useCallback(
    (connection: Edge | Connection) => {
      // round5:挡自连(自环)——拖到节点自己的同类型输入会建自环边,要等执行时
      // topoSort 才报「循环依赖」;backend 执行路径前端更无守卫。提前挡掉。
      if (connection.source === connection.target) return false
      const currentNodes = nodesRef.current
      const sourceNode = currentNodes.find((n) => n.id === connection.source)
      const targetNode = currentNodes.find((n) => n.id === connection.target)
      if (!sourceNode || !targetNode) return false
      const sourceType = getPortType(sourceNode.type ?? '', connection.sourceHandle)
      const targetType = getPortType(targetNode.type ?? '', connection.targetHandle)
      if (!sourceType || !targetType) return false
      return sourceType === targetType
    },
    [],
  )

  // Sync React Flow positions/sizes to Zustand store (uses ref to avoid stale closures)
  const syncToStore = useCallback(() => {
    const currentNodes = nodesRef.current
    setWorkflow({
      ...workflow,
      nodes: workflow.nodes.map((wn) => {
        const rfNode = currentNodes.find((n) => n.id === wn.id)
        if (!rfNode) return wn
        const updated: any = { ...wn, position: rfNode.position }
        if (rfNode.style) updated.style = rfNode.style
        if (rfNode.width != null) updated.width = rfNode.width
        if (rfNode.height != null) updated.height = rfNode.height
        return updated
      }),
    })
  }, [workflow, setWorkflow])

  // Sync resize changes to store
  useEffect(() => {
    const handler = () => syncToStore()
    window.addEventListener('node-resize-end', handler)
    return () => window.removeEventListener('node-resize-end', handler)
  }, [syncToStore])

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
  }, [])

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault()
      const type = event.dataTransfer.getData('application/reactflow') as NodeType
      if (!type || !NODE_DEFS[type]) return
      const rfi = reactFlowInstance.current
      if (!rfi || !reactFlowWrapper.current) return
      const bounds = reactFlowWrapper.current.getBoundingClientRect()
      const position = rfi.screenToFlowPosition({
        x: event.clientX - bounds.left,
        y: event.clientY - bounds.top,
      })
      const id = crypto.randomUUID().slice(0, 8)
      const newNode: Node = { id, type, position, data: {}, style: { width: 320 } }
      setNodes((nds) => [...nds, newNode])
      storeAddNode({ id, type, position, data: {} })
    },
    [setNodes, storeAddNode],
  )

  // 复制选中节点(+ 它们之间的内部连线)到剪贴板。深拷贝 data 防共享引用;
  // 保留原 id 以便粘贴时按 id 映射重连内部边。
  const copySelection = useCallback(() => {
    const selected = nodesRef.current.filter((n) => n.selected)
    if (selected.length === 0) return false
    const selectedIds = new Set(selected.map((n) => n.id))
    clipboardRef.current = {
      nodes: selected.map((n) => ({
        id: n.id,
        type: n.type ?? '',
        data: structuredClone(n.data ?? {}) as Record<string, unknown>,
        position: { ...n.position },
        style: (n as any).style,
        width: (n as any).width,
        height: (n as any).height,
      })),
      // 只带「两端都在选区内」的边 —— 跨选区边粘贴后无对应端点。
      edges: edgesRef.current
        .filter((e) => selectedIds.has(e.source) && selectedIds.has(e.target))
        .map((e) => ({
          source: e.source,
          sourceHandle: e.sourceHandle ?? '',
          target: e.target,
          targetHandle: e.targetHandle ?? '',
        })),
    }
    pasteSeqRef.current = 0
    return true
  }, [])

  // 粘贴:用纯函数 buildPastedGraph 发新 id、偏移落位、内部边重连(逻辑在
  // utils/pasteGraph.ts,有单测)。走 store 批量 addNodesWithEdges(单次 undo),
  // 并即时 setNodes/setEdges 选中新节点。
  const pasteClipboard = useCallback(() => {
    const clip = clipboardRef.current
    if (!clip || clip.nodes.length === 0) return
    pasteSeqRef.current += 1
    const { nodes: pn, edges: pe } = buildPastedGraph(clip, 40 * pasteSeqRef.current, () =>
      crypto.randomUUID().slice(0, 8),
    )
    const newNodes = pn.map((n) => {
      const node: any = { id: n.id, type: n.type, position: n.position, data: n.data, style: n.style ?? { width: 320 } }
      if (n.width != null) node.width = n.width
      if (n.height != null) node.height = n.height
      return node as WorkflowNode
    })
    const newEdges: WorkflowEdge[] = pe.map((e) => ({
      id: e.id, source: e.source, sourceHandle: e.sourceHandle, target: e.target, targetHandle: e.targetHandle,
    }))

    storeAddNodesWithEdges(newNodes, newEdges)
    // 即时渲染 + 选中粘贴出来的节点(取消原选区),方便接着拖动。
    setNodes((nds) => [
      ...nds.map((n) => ({ ...n, selected: false })),
      ...newNodes.map((n) => ({
        id: n.id,
        type: n.type,
        position: n.position,
        data: n.data,
        style: (n as any).style ?? { width: 320 },
        selected: true,
      } as Node)),
    ])
    setEdges((eds) => [
      ...eds,
      ...newEdges.map((e) => ({ id: e.id, source: e.source, sourceHandle: e.sourceHandle, target: e.target, targetHandle: e.targetHandle, type: 'portTyped' } as Edge)),
    ])
  }, [setNodes, setEdges, storeAddNodesWithEdges])

  // 复制 Ctrl/Cmd+C / 粘贴 Ctrl/Cmd+V / 原地复制 Ctrl/Cmd+D。
  // 与 undo/redo 同样:focus 在 input/textarea/contenteditable 时放行原生行为。
  // Ctrl+V 仅在 in-app 剪贴板有节点时接管 —— 否则放行(让 MultimodalInputNode
  // 的图片粘贴等原生 paste 正常工作)。
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey
      if (!mod) return
      const tgt = e.target as HTMLElement | null
      if (tgt) {
        const tag = tgt.tagName
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tgt.isContentEditable) return
      }
      const k = e.key.toLowerCase()
      if (k === 'c') {
        copySelection()
      } else if (k === 'v') {
        if (clipboardRef.current && clipboardRef.current.nodes.length > 0) {
          e.preventDefault()
          pasteClipboard()
        }
      } else if (k === 'd') {
        if (nodesRef.current.some((n) => n.selected)) {
          e.preventDefault()
          copySelection()
          pasteClipboard()
        }
      } else if (k === 'b') {
        // 旁路/取消旁路选中节点(对齐 ComfyUI Ctrl+B)。整组按「是否全已旁路」翻转:
        // 有任一未旁路 → 全部旁路;否则全部取消旁路。flag 落 node.data.bypassed。
        const selected = nodesRef.current.filter((n) => n.selected)
        if (selected.length > 0) {
          e.preventDefault()
          const anyOn = selected.some((n) => !(n.data as any)?.bypassed)
          for (const n of selected) updateNode(n.id, { bypassed: anyOn })
        }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [copySelection, pasteClipboard, updateNode])

  // m09: 画布模式下左节点库 + 右属性面板**常驻**。overlay 视图
  // （dashboard / services / 设置 等）下两边都隐藏。
  // v3 没有 panel 切换 — PanelStore.PANEL_ITEMS 已空 — activePanel
  // 系统作为 legacy workflows/presets 兜底（如有，仍然 PANEL_MAP 渲染）。
  const showPropertyPanel = !activeOverlay
  const PROPERTY_PANEL_WIDTH = 300

  const LegacyPanel = activePanel && activePanel !== 'nodes' ? PANEL_MAP[activePanel] : null
  const showLegacyPanel = !!LegacyPanel && !activeOverlay
  const showNodeLibrary = !activeOverlay && !showLegacyPanel
  const NodeLibraryComponent = PANEL_MAP.nodes

  return (
    <div className="relative flex-1 overflow-hidden" ref={reactFlowWrapper}>
      {/* Floating side panel — m09: 常驻节点库；legacy workflows/presets 仍兼容 */}
      {showNodeLibrary && <NodeLibraryComponent />}
      {showLegacyPanel && LegacyPanel && <LegacyPanel />}

      {/* Canvas area — offset when panel is open */}
      <div
        className="absolute inset-0 transition-[left] duration-200"
        style={{
          left: showNodeLibrary || showLegacyPanel ? panelWidth : 0,
          right: showPropertyPanel ? PROPERTY_PANEL_WIDTH : 0,
        }}
      >
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          isValidConnection={isValidConnection}
          onInit={(instance) => { reactFlowInstance.current = instance }}
          onDragOver={onDragOver}
          onDrop={onDrop}
          onSelectionChange={({ nodes: sel }) => {
            // Only single-select drives the property panel; multi-select / deselect → null.
            setSelectedNodeId(sel.length === 1 ? sel[0].id : null)
          }}
          onNodeDragStop={syncToStore}
          onEdgeDoubleClick={(_event, edge) => {
            setEdges((eds) => eds.filter((e) => e.id !== edge.id))
            storeRemoveEdge(edge.id)
          }}
          onEdgeContextMenu={(event, edge) => {
            event.preventDefault()
            setEdges((eds) => eds.filter((e) => e.id !== edge.id))
            storeRemoveEdge(edge.id)
          }}
          onNodeContextMenu={(event, node) => {
            event.preventDefault()
            // 右键即选中该节点(便于「复制」读选区),并在指针处弹菜单。
            setNodes((nds) => nds.map((n) => ({ ...n, selected: n.id === node.id })))
            const bounds = reactFlowWrapper.current?.getBoundingClientRect()
            setCtxMenu({
              x: event.clientX - (bounds?.left ?? 0),
              y: event.clientY - (bounds?.top ?? 0),
              nodeId: node.id,
            })
          }}
          onPaneClick={() => setCtxMenu(null)}
          onMoveStart={() => setCtxMenu(null)}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          deleteKeyCode={['Backspace', 'Delete']}
          edgesReconnectable
          minZoom={0.15}
          maxZoom={8}
          fitView
          fitViewOptions={{ padding: 0.1, minZoom: 0.5, maxZoom: 1.5 }}
          defaultEdgeOptions={{
            type: 'portTyped',
            focusable: true,
            interactionWidth: 20,
          }}
          style={{ background: 'var(--bg)' }}
        >
          <Background color="rgba(255,255,255,0.03)" gap={20} />
          <Controls />
          <MiniMap
            nodeColor={() => 'var(--muted-strong)'}
            style={{ background: 'var(--bg-accent)' }}
            pannable
            zoomable
          />
        </ReactFlow>
      </div>

      {/* 节点右键菜单(旁路/复制/删除)*/}
      {ctxMenu && (() => {
        const ctxNode = workflow.nodes.find((n) => n.id === ctxMenu.nodeId)
        const isBypassed = !!ctxNode?.data?.bypassed
        const close = () => setCtxMenu(null)
        const item = (icon: React.ReactNode, label: string, onClick: () => void, danger?: boolean) => (
          <button
            type="button"
            onClick={() => { onClick(); close() }}
            className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-left"
            style={{ color: danger ? 'var(--err, #ef4444)' : 'var(--text)', background: 'transparent', border: 'none', cursor: 'pointer' }}
            onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-hover)')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
          >
            {icon}
            <span>{label}</span>
          </button>
        )
        return (
          <>
            {/* 点击空白关闭(覆盖全画布的透明层)*/}
            <div className="absolute inset-0 z-30" onClick={close} onContextMenu={(e) => { e.preventDefault(); close() }} />
            <div
              className="absolute z-40 py-1 rounded-md"
              style={{
                top: ctxMenu.y, left: ctxMenu.x, minWidth: 140,
                background: 'var(--bg-elevated)', border: '1px solid var(--border)', boxShadow: 'var(--shadow-md)',
              }}
            >
              {item(<Ban size={13} />, isBypassed ? '取消旁路' : '旁路 (Ctrl+B)', () => updateNode(ctxMenu.nodeId, { bypassed: !isBypassed }))}
              {item(<Copy size={13} />, '复制 (Ctrl+D)', () => {
                setNodes((nds) => nds.map((n) => ({ ...n, selected: n.id === ctxMenu.nodeId })))
                // 选中态在下一帧才稳定 → 延后一拍再 copy+paste
                setTimeout(() => { copySelection(); pasteClipboard() }, 0)
              })}
              {item(<Trash2 size={13} />, '删除', () => {
                setNodes((nds) => nds.filter((n) => n.id !== ctxMenu.nodeId))
                storeRemoveNode(ctxMenu.nodeId)
              }, true)}
            </div>
          </>
        )
      })()}

      {/* m09: 节点属性面板（画布模式常驻右侧；overlay 视图下隐藏） */}
      {showPropertyPanel && <NodePropertyPanel />}

      {/* Page overlays */}
      {activeOverlay === 'dashboard' && <DashboardOverlay />}
      {activeOverlay === 'models' && <ModelsOverlay />}
      {activeOverlay === 'settings' && <SettingsOverlay />}
      {activeOverlay === 'preset-detail' && <PresetDetailOverlay />}
      {activeOverlay === 'api-keys-list' && <ApiKeysList />}
      {activeOverlay === 'api-key-detail' && <ApiKeyDetail />}
      {activeOverlay === 'agents' && <AgentManagementOverlay />}
      {activeOverlay === 'logs' && <LogsOverlay />}
      {activeOverlay === 'node-packages' && <NodePackagesOverlay />}
      {activeOverlay === 'services' && <ServicesList />}
      {activeOverlay === 'apps' && <ServicesList />}
      {activeOverlay === 'service-detail' && <ServiceDetailRoute />}
      {activeOverlay === 'workflows-list' && <WorkflowsList />}
      {activeOverlay === 'usage' && <UsagePage />}

      {/* PR-3g(2026-05-28 任务面板重置收尾):删 QueueProgressOverlay 老画布右上浮窗
          (PR-170 时代方案)。任务进度统一由 GlobalTopbar 任务下拉 + Active/History
          dropdown panel 承担(PR-3a-3e)。旧 overlay 文件留着只为兼容 TaskPanel 单测,
          后续如果完全摘除可再删 QueueProgressOverlay.tsx + TaskMenuButton.tsx。 */}
    </div>
  )
}

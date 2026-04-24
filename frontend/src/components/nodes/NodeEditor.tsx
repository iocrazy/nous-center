import { useCallback, useRef, useMemo, useEffect } from 'react'
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
import { useWorkspaceStore } from '../../stores/workspace'
import { usePanelStore } from '../../stores/panel'
import { useExecutionStore } from '../../stores/execution'
import { NODE_DEFS, type NodeType, type PortType } from '../../models/workflow'
import NodeLibraryPanel from '../panels/NodeLibraryPanel'
import WorkflowsPanel from '../panels/WorkflowsPanel'
import PresetsPanel from '../panels/PresetsPanel'
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
import TaskPanel from '../panels/TaskPanel'

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
  const storeRemoveNode = useWorkspaceStore((s) => s.removeNode)
  const undo = useWorkspaceStore((s) => s.undo)
  const redo = useWorkspaceStore((s) => s.redo)
  const { activePanel, activeOverlay, panelWidth } = usePanelStore()
  const nodeStates = useExecutionStore((s) => s.nodeStates)
  const taskPanelOpen = useExecutionStore((s) => s.taskPanelOpen)
  const toggleTaskPanel = useExecutionStore((s) => s.toggleTaskPanel)

  const reactFlowWrapper = useRef<HTMLDivElement>(null)
  const reactFlowInstance = useRef<ReactFlowInstance | null>(null)

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
        className: nodeStates[n.id] ? NODE_STATE_CLASS[nodeStates[n.id]] : undefined,
      })),
    [workflow.nodes, nodeStates],
  )

  const rfEdges: Edge[] = useMemo(
    () =>
      workflow.edges.map((e) => ({
        id: e.id,
        source: e.source,
        sourceHandle: e.sourceHandle,
        target: e.target,
        targetHandle: e.targetHandle,
        style: { stroke: 'var(--muted-strong)' },
        animated: true,
      })),
    [workflow.edges],
  )

  const [nodes, setNodes, onNodesChangeInternal] = useNodesState(rfNodes)
  const [edges, setEdges, onEdgesChangeInternal] = useEdgesState(rfEdges)

  // Keep a ref to always access latest React Flow nodes (avoids stale closures)
  const nodesRef = useRef(nodes)
  nodesRef.current = nodes

  // Sync Zustand store changes back to React Flow, preserving resize dimensions
  useEffect(() => {
    setNodes((prev) =>
      rfNodes.map((rfn) => {
        const existing = prev.find((p) => p.id === rfn.id)
        if (!existing) return rfn
        // Preserve React Flow's resize state (width/height/style set by NodeResizer)
        return {
          ...rfn,
          ...(existing.width != null ? { width: existing.width } : {}),
          ...(existing.height != null ? { height: existing.height } : {}),
          ...(existing.style ? { style: existing.style } : {}),
        }
      }),
    )
  }, [rfNodes, setNodes])
  useEffect(() => { setEdges(rfEdges) }, [rfEdges, setEdges])

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
      setEdges((eds) =>
        addEdge({ ...params, animated: true, style: { stroke: 'var(--muted-strong)' } }, eds),
      )
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

  // Determine which floating panel to show
  const PanelComponent = activePanel && !activeOverlay ? PANEL_MAP[activePanel] : null

  const showPanel = PanelComponent && !activeOverlay

  return (
    <div className="relative flex-1 overflow-hidden" ref={reactFlowWrapper}>
      {/* Floating side panel */}
      {showPanel && <PanelComponent />}

      {/* Canvas area — offset when panel is open */}
      <div
        className="absolute inset-0 transition-[left] duration-200"
        style={{ left: showPanel ? panelWidth : 0 }}
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
          nodeTypes={nodeTypes}
          deleteKeyCode={['Backspace', 'Delete']}
          edgesReconnectable
          fitView
          fitViewOptions={{ padding: 0.1, minZoom: 0.5, maxZoom: 1.5 }}
          defaultEdgeOptions={{
            animated: true,
            style: { stroke: 'var(--muted-strong)', strokeWidth: 2 },
            focusable: true,
            interactionWidth: 20,
          }}
          onEdgeClick={(_event, edge) => {
            setEdges((eds) => eds.map((e) =>
              e.id === edge.id
                ? { ...e, selected: true, style: { stroke: 'var(--accent)', strokeWidth: 3 } }
                : { ...e, selected: false, style: { stroke: 'var(--muted-strong)', strokeWidth: 2 } }
            ))
          }}
          style={{ background: 'var(--bg)' }}
        >
          <Background color="rgba(255,255,255,0.03)" gap={20} />
          <Controls />
          <MiniMap
            nodeColor={() => 'var(--muted-strong)'}
            style={{ background: 'var(--bg-accent)' }}
          />
        </ReactFlow>
      </div>

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

      {/* Task panel */}
      <TaskPanel open={taskPanelOpen} onClose={toggleTaskPanel} />
    </div>
  )
}

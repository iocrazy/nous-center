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
import { NODE_DEFS, type NodeType, type PortType } from '../../models/workflow'
import NodeLibraryPanel from '../panels/NodeLibraryPanel'
import WorkflowsPanel from '../panels/WorkflowsPanel'
import PresetsPanel from '../panels/PresetsPanel'
import DashboardOverlay from '../overlays/DashboardOverlay'
import ModelsOverlay from '../overlays/ModelsOverlay'
import SettingsOverlay from '../overlays/SettingsOverlay'
import PresetDetailOverlay from '../overlays/PresetDetailOverlay'
import ApiManagementOverlay from '../overlays/ApiManagementOverlay'

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
  const { activePanel, activeOverlay, panelWidth } = usePanelStore()

  const reactFlowWrapper = useRef<HTMLDivElement>(null)
  const reactFlowInstance = useRef<ReactFlowInstance | null>(null)

  const rfNodes: Node[] = useMemo(
    () =>
      workflow.nodes.map((n) => ({
        id: n.id,
        type: n.type,
        position: n.position,
        data: n.data,
      })),
    [workflow.nodes],
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

  // Sync Zustand store changes (e.g. widget edits) back to React Flow's internal state
  useEffect(() => { setNodes(rfNodes) }, [rfNodes, setNodes])
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
      const sourceNode = nodes.find((n) => n.id === connection.source)
      const targetNode = nodes.find((n) => n.id === connection.target)
      if (!sourceNode || !targetNode) return false
      const sourceType = getPortType(sourceNode.type ?? '', connection.sourceHandle)
      const targetType = getPortType(targetNode.type ?? '', connection.targetHandle)
      if (!sourceType || !targetType) return false
      return sourceType === targetType
    },
    [nodes],
  )

  const syncPositionsToStore = useCallback(() => {
    setWorkflow({
      ...workflow,
      nodes: workflow.nodes.map((wn) => {
        const rfNode = nodes.find((n) => n.id === wn.id)
        return rfNode ? { ...wn, position: rfNode.position } : wn
      }),
    })
  }, [nodes, workflow, setWorkflow])

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
      const newNode: Node = { id, type, position, data: {} }
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
          onNodeDragStop={syncPositionsToStore}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.1, minZoom: 0.5, maxZoom: 1.5 }}
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
      {activeOverlay === 'api-management' && <ApiManagementOverlay />}
    </div>
  )
}

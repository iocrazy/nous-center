import { create } from 'zustand'

export type NodeExecState = 'pending' | 'running' | 'completed' | 'error'

export interface ExecutionState {
  isRunning: boolean
  taskId: string | null
  progress: number
  error: string | null
  result: { audioBase64: string; sampleRate: number; duration: number } | null
  _ws: WebSocket | null

  // Node-level execution state
  nodeStates: Record<string, NodeExecState>
  currentNodeId: string | null
  currentNodeType: string | null

  start: (taskId?: string) => void
  setProgress: (progress: number) => void
  succeed: (result: ExecutionState['result']) => void
  fail: (error: string) => void
  reset: () => void
  wsConnect: (instanceId: string) => void
  wsDisconnect: () => void

  // Node-level actions
  setNodeState: (nodeId: string, state: NodeExecState) => void
  clearNodeState: (nodeId: string) => void
  setCurrentNode: (nodeId: string | null, nodeType: string | null) => void
  resetNodeStates: () => void

  // Task panel
  taskPanelOpen: boolean
  toggleTaskPanel: () => void
}

export const useExecutionStore = create<ExecutionState>((set, get) => ({
  isRunning: false,
  taskId: null,
  progress: 0,
  error: null,
  result: null,
  _ws: null,
  nodeStates: {},
  currentNodeId: null,
  currentNodeType: null,
  taskPanelOpen: false,

  start: (taskId) =>
    set({ isRunning: true, taskId: taskId ?? null, progress: 0, error: null, result: null, nodeStates: {}, currentNodeId: null, currentNodeType: null }),

  setProgress: (progress) => set({ progress }),

  succeed: (result) =>
    set({ isRunning: false, progress: 100, result, error: null, currentNodeId: null, currentNodeType: null }),

  fail: (error) =>
    set({ isRunning: false, error, currentNodeId: null, currentNodeType: null }),

  reset: () =>
    set({ isRunning: false, taskId: null, progress: 0, error: null, result: null, nodeStates: {}, currentNodeId: null, currentNodeType: null }),

  setNodeState: (nodeId, state) =>
    set((s) => ({ nodeStates: { ...s.nodeStates, [nodeId]: state } })),

  clearNodeState: (nodeId) =>
    set((s) => {
      const { [nodeId]: _drop, ...rest } = s.nodeStates
      return { nodeStates: rest }
    }),

  setCurrentNode: (nodeId, nodeType) =>
    set({ currentNodeId: nodeId, currentNodeType: nodeType }),

  resetNodeStates: () =>
    set({ nodeStates: {}, currentNodeId: null, currentNodeType: null }),

  toggleTaskPanel: () =>
    set((s) => ({ taskPanelOpen: !s.taskPanelOpen })),

  wsConnect: (instanceId: string) => {
    const existing = get()._ws
    if (existing) existing.close()

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const ws = new WebSocket(`${protocol}//${host}/ws/workflow/${instanceId}`)

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data)
      if (data.type === 'node_start') {
        set((s) => ({
          progress: data.progress ?? s.progress,
          nodeStates: { ...s.nodeStates, [data.node_id]: 'running' },
          currentNodeId: data.node_id ?? null,
          currentNodeType: data.node_type ?? null,
        }))
      }
      if (data.type === 'node_complete') {
        set((s) => ({
          progress: data.progress ?? s.progress,
          nodeStates: { ...s.nodeStates, [data.node_id]: 'completed' },
        }))
      }
      if (data.type === 'node_error') {
        set((s) => ({
          error: data.error,
          nodeStates: { ...s.nodeStates, [data.node_id]: 'error' },
        }))
      }
      if (data.type === 'complete') {
        set({ progress: 100, currentNodeId: null, currentNodeType: null })
      }
    }
    set({ _ws: ws })
  },

  wsDisconnect: () => {
    const ws = get()._ws
    if (ws) ws.close()
    set({ _ws: null })
  },
}))

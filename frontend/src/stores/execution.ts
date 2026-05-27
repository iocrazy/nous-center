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

  // PR-E2/F2:对齐 ComfyUI「全部:N% / 节点:M%」+ live preview thumbnail。
  // currentNodeProgress = 当前节点内步级进度(0-100,例如 KSampler step 12/25 = 48%);
  // latestPreviewUrl = 最近一帧 latent live preview(WS node_progress.preview_url)。
  currentNodeProgress: number | null
  currentNodeStep: { done: number; total: number } | null
  latestPreviewUrl: string | null

  start: (taskId?: string) => void
  setProgress: (progress: number) => void
  setCurrentNodeProgress: (
    percent: number | null,
    step?: { done: number; total: number } | null,
    previewUrl?: string | null,
  ) => void
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
  /** Tasks badge(后续 sidebar PR-3 接管) 计数（DD4）：点 Run 累加，打开面板清零。
   * 与「running task 数」解耦 —— 它表达的是「有未查看的新提交」。 */
  taskIconBadge: number
  bumpTaskBadge: () => void
  clearTaskBadge: () => void
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
  currentNodeProgress: null,
  currentNodeStep: null,
  latestPreviewUrl: null,
  taskPanelOpen: false,
  taskIconBadge: 0,

  start: (taskId) =>
    set({ isRunning: true, taskId: taskId ?? null, progress: 0, error: null, result: null, nodeStates: {}, currentNodeId: null, currentNodeType: null, currentNodeProgress: null, currentNodeStep: null, latestPreviewUrl: null }),

  setProgress: (progress) => set({ progress }),

  setCurrentNodeProgress: (percent, step = null, previewUrl) =>
    set((s) => ({
      currentNodeProgress: percent,
      currentNodeStep: step,
      latestPreviewUrl: previewUrl !== undefined ? previewUrl : s.latestPreviewUrl,
    })),

  succeed: (result) =>
    set({ isRunning: false, progress: 100, result, error: null, currentNodeId: null, currentNodeType: null, currentNodeProgress: null, currentNodeStep: null, latestPreviewUrl: null }),

  fail: (error) =>
    set({ isRunning: false, error, currentNodeId: null, currentNodeType: null, currentNodeProgress: null, currentNodeStep: null }),

  reset: () =>
    set({ isRunning: false, taskId: null, progress: 0, error: null, result: null, nodeStates: {}, currentNodeId: null, currentNodeType: null, currentNodeProgress: null, currentNodeStep: null, latestPreviewUrl: null }),

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

  bumpTaskBadge: () => set((s) => ({ taskIconBadge: s.taskIconBadge + 1 })),

  clearTaskBadge: () => set({ taskIconBadge: 0 }),

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
          // 节点完成 → 清当前节点进度(下个 node_start/node_progress 会重置)。
          currentNodeProgress: null,
          currentNodeStep: null,
        }))
      }
      // PR-E2/F2:每步 node_progress 更新「节点内 %」+ latest preview thumbnail。
      if (data.type === 'node_progress') {
        const m = typeof data.detail === 'string' ? /step\s+(\d+)\s*\/\s*(\d+)/.exec(data.detail) : null
        const percent = typeof data.progress === 'number' ? Math.round(data.progress * 100)
          : (m ? Math.round((Number(m[1]) / Number(m[2])) * 100) : null)
        set((s) => ({
          currentNodeProgress: percent,
          currentNodeStep: m ? { done: Number(m[1]), total: Number(m[2]) } : s.currentNodeStep,
          latestPreviewUrl: typeof data.preview_url === 'string' && data.preview_url
            ? data.preview_url
            : s.latestPreviewUrl,
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

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
  // PR-3b(任务面板重置 L3 callout):接 PR-1a 后端发的 stage/step_latency_ms/eta_ms。
  // 用于 ActiveTaskRow task-callout 渲染「⚡ dit denoise · step 27/50 · 240ms/step · ETA 5.5s」。
  currentNodeStage: string | null          // 'text_encode' / 'dit_denoise' / 'vae_decode' / 'tts_synth' / 'llm_gen' / 'vision_inference'
  currentNodeStepLatencyMs: number | null  // per-step 平均 latency
  currentNodeEtaMs: number | null          // 估计剩余 ms

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

  /** PR-3c:history list 行级展开态(按 task.id key 入 Set)。默认空 → 按 task_type 决定:
   * image/tts 默认展开;llm/vision 默认折叠(对齐 mockup variant-final 示例形态)。
   * 用户点 chevron 强制覆盖默认。 */
  expandedHistoryRowIds: Set<string>
  toggleHistoryRowExpanded: (id: string) => void

  /** PR-3e:任务详情 modal。点任务缩略图 / 「点击放大→」打开,820×600 居中浮层,
   * 按 service type 切换内部布局(image/tts/llm/vision)。 */
  detailModalTaskId: string | null
  openDetailModal: (taskId: string) => void
  closeDetailModal: () => void
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
  currentNodeStage: null,
  currentNodeStepLatencyMs: null,
  currentNodeEtaMs: null,
  taskPanelOpen: false,
  taskIconBadge: 0,
  expandedHistoryRowIds: new Set<string>(),
  detailModalTaskId: null,

  start: (taskId) =>
    set({ isRunning: true, taskId: taskId ?? null, progress: 0, error: null, result: null, nodeStates: {}, currentNodeId: null, currentNodeType: null, currentNodeProgress: null, currentNodeStep: null, latestPreviewUrl: null, currentNodeStage: null, currentNodeStepLatencyMs: null, currentNodeEtaMs: null }),

  setProgress: (progress) => set({ progress }),

  setCurrentNodeProgress: (percent, step = null, previewUrl) =>
    set((s) => ({
      currentNodeProgress: percent,
      currentNodeStep: step,
      latestPreviewUrl: previewUrl !== undefined ? previewUrl : s.latestPreviewUrl,
    })),

  succeed: (result) =>
    set({ isRunning: false, progress: 100, result, error: null, currentNodeId: null, currentNodeType: null, currentNodeProgress: null, currentNodeStep: null, latestPreviewUrl: null, currentNodeStage: null, currentNodeStepLatencyMs: null, currentNodeEtaMs: null }),

  fail: (error) =>
    set({ isRunning: false, error, currentNodeId: null, currentNodeType: null, currentNodeProgress: null, currentNodeStep: null, currentNodeStage: null, currentNodeStepLatencyMs: null, currentNodeEtaMs: null }),

  reset: () =>
    set({ isRunning: false, taskId: null, progress: 0, error: null, result: null, nodeStates: {}, currentNodeId: null, currentNodeType: null, currentNodeProgress: null, currentNodeStep: null, latestPreviewUrl: null, currentNodeStage: null, currentNodeStepLatencyMs: null, currentNodeEtaMs: null }),

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
    // round3 #6:打开面板时一并清掉「N 个新任务」徽章。早先只有 IconRail 入口清,
    // topbar chip / 任务菜单 / queue overlay「查看全部」都只 toggle、不清 → 看了任务
    // 徽章仍残留。收进 toggle → 所有入口统一受益(IconRail 旧的清除变冗余但无害)。
    set((s) => ({
      taskPanelOpen: !s.taskPanelOpen,
      taskIconBadge: !s.taskPanelOpen ? 0 : s.taskIconBadge,
    })),

  bumpTaskBadge: () => set((s) => ({ taskIconBadge: s.taskIconBadge + 1 })),

  clearTaskBadge: () => set({ taskIconBadge: 0 }),

  toggleHistoryRowExpanded: (id) =>
    set((s) => {
      const next = new Set(s.expandedHistoryRowIds)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return { expandedHistoryRowIds: next }
    }),

  openDetailModal: (taskId) => set({ detailModalTaskId: taskId }),
  closeDetailModal: () => set({ detailModalTaskId: null }),

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
      // PR-3b 扩展:接 L3 stage / step_latency_ms / eta_ms(后端 PR-1a/1b/1c/1d 发的字段)。
      if (data.type === 'node_progress') {
        const m = typeof data.detail === 'string' ? /step\s+(\d+)\s*\/\s*(\d+)/.exec(data.detail) : null
        const stepFromFields = typeof data.step === 'number' && typeof data.total_steps === 'number'
          ? { done: data.step, total: data.total_steps }
          : (m ? { done: Number(m[1]), total: Number(m[2]) } : null)
        const percent = typeof data.progress === 'number' ? Math.round(data.progress * 100)
          : (m ? Math.round((Number(m[1]) / Number(m[2])) * 100) : null)
        set((s) => ({
          currentNodeProgress: percent,
          currentNodeStep: stepFromFields ?? s.currentNodeStep,
          latestPreviewUrl: typeof data.preview_url === 'string' && data.preview_url
            ? data.preview_url
            : s.latestPreviewUrl,
          currentNodeStage: typeof data.stage === 'string' ? data.stage : s.currentNodeStage,
          currentNodeStepLatencyMs: typeof data.step_latency_ms === 'number'
            ? data.step_latency_ms : s.currentNodeStepLatencyMs,
          currentNodeEtaMs: typeof data.eta_ms === 'number' ? data.eta_ms : s.currentNodeEtaMs,
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

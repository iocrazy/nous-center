import { create } from 'zustand'

export interface ExecutionState {
  isRunning: boolean
  taskId: string | null
  progress: number
  error: string | null
  result: { audioBase64: string; sampleRate: number; duration: number } | null
  _ws: WebSocket | null

  start: (taskId?: string) => void
  setProgress: (progress: number) => void
  succeed: (result: ExecutionState['result']) => void
  fail: (error: string) => void
  reset: () => void
  wsConnect: (instanceId: string) => void
  wsDisconnect: () => void
}

export const useExecutionStore = create<ExecutionState>((set, get) => ({
  isRunning: false,
  taskId: null,
  progress: 0,
  error: null,
  result: null,
  _ws: null,

  start: (taskId) =>
    set({ isRunning: true, taskId: taskId ?? null, progress: 0, error: null, result: null }),

  setProgress: (progress) => set({ progress }),

  succeed: (result) =>
    set({ isRunning: false, progress: 100, result, error: null }),

  fail: (error) =>
    set({ isRunning: false, error }),

  reset: () =>
    set({ isRunning: false, taskId: null, progress: 0, error: null, result: null }),

  wsConnect: (instanceId: string) => {
    const existing = get()._ws
    if (existing) existing.close()

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const ws = new WebSocket(`${protocol}//${host}/ws/workflow/${instanceId}`)

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data)
      if (data.type === 'node_start' || data.type === 'node_complete') {
        set({ progress: data.progress })
      }
      if (data.type === 'node_error') {
        set({ error: data.error })
      }
      if (data.type === 'complete') {
        set({ progress: 100 })
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

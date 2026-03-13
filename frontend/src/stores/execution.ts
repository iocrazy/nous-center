import { create } from 'zustand'

export interface ExecutionState {
  isRunning: boolean
  taskId: string | null
  progress: number
  error: string | null
  result: { audioBase64: string; sampleRate: number; duration: number } | null

  start: (taskId?: string) => void
  setProgress: (progress: number) => void
  succeed: (result: ExecutionState['result']) => void
  fail: (error: string) => void
  reset: () => void
}

export const useExecutionStore = create<ExecutionState>((set) => ({
  isRunning: false,
  taskId: null,
  progress: 0,
  error: null,
  result: null,

  start: (taskId) =>
    set({ isRunning: true, taskId: taskId ?? null, progress: 0, error: null, result: null }),

  setProgress: (progress) => set({ progress }),

  succeed: (result) =>
    set({ isRunning: false, progress: 100, result, error: null }),

  fail: (error) =>
    set({ isRunning: false, error }),

  reset: () =>
    set({ isRunning: false, taskId: null, progress: 0, error: null, result: null }),
}))

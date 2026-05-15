import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook } from '@testing-library/react'
import { useTaskCompletionNotifier } from './useTaskCompletionNotifier'

const notifyOnce = vi.fn()
vi.mock('../stores/notifications', () => ({
  useNotificationStore: () => ({ notifyOnce }),
}))
const toastAdd = vi.fn()
vi.mock('../stores/toast', () => ({
  useToastStore: (sel: (s: { add: typeof toastAdd }) => unknown) => sel({ add: toastAdd }),
}))

let mockTasks: any[] = []
vi.mock('../api/tasks', () => ({
  useTasks: () => ({ data: mockTasks }),
}))

describe('useTaskCompletionNotifier', () => {
  beforeEach(() => {
    notifyOnce.mockClear()
    mockTasks = []
  })

  it('does NOT notify for tasks already terminal on first load', () => {
    mockTasks = [{ id: 't1', status: 'completed', workflow_name: 'wf', duration_ms: 1000 }]
    renderHook(() => useTaskCompletionNotifier())
    expect(notifyOnce).not.toHaveBeenCalled()
  })

  it('notifies when a running task transitions to completed', () => {
    mockTasks = [{ id: 't1', status: 'running', workflow_name: 'flux2-人物立绘', duration_ms: null }]
    const { rerender } = renderHook(() => useTaskCompletionNotifier())
    expect(notifyOnce).not.toHaveBeenCalled()

    mockTasks = [{ id: 't1', status: 'completed', workflow_name: 'flux2-人物立绘', duration_ms: 34000 }]
    rerender()
    expect(notifyOnce).toHaveBeenCalledTimes(1)
    expect(notifyOnce).toHaveBeenCalledWith(
      't1',
      expect.stringContaining('flux2-人物立绘'),
      'success',
      toastAdd,
    )
  })

  it('notifies with error type when a task transitions to failed', () => {
    mockTasks = [{ id: 't2', status: 'running', workflow_name: 'sd-bg', duration_ms: null }]
    const { rerender } = renderHook(() => useTaskCompletionNotifier())
    mockTasks = [{ id: 't2', status: 'failed', workflow_name: 'sd-bg', duration_ms: 2000 }]
    rerender()
    expect(notifyOnce).toHaveBeenCalledWith('t2', expect.stringContaining('失败'), 'error', toastAdd)
  })
})

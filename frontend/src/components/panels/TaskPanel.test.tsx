import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import TaskPanel from './TaskPanel'

// m15 mockup 对齐：3 tab 切换、汇总数字、空态。

const mockTasks = [
  {
    id: 'wf_exec_7k2m',
    workflow_name: '多角色播客合成',
    status: 'running',
    nodes_done: 3,
    nodes_total: 7,
    duration_ms: null,
    created_at: new Date().toISOString(),
    error: null,
    result: null,
    current_node: 'tts_engine',
  },
  {
    id: 'wf_done_1a2b',
    workflow_name: 'podcast-short',
    status: 'completed',
    nodes_done: 5,
    nodes_total: 5,
    duration_ms: 106000,
    created_at: new Date().toISOString(),
    error: null,
    result: 'ok',
    current_node: null,
  },
  {
    id: 'wf_failed_x',
    workflow_name: 'failed-tts',
    status: 'failed',
    nodes_done: 1,
    nodes_total: 5,
    duration_ms: 2300,
    created_at: new Date().toISOString(),
    error: 'CUDA out of memory',
    result: null,
    current_node: null,
  },
]

vi.mock('../../api/tasks', () => ({
  useTasks: () => ({ data: mockTasks }),
  useCancelTask: () => ({ mutate: vi.fn(), isPending: false }),
  useRetryTask: () => ({ mutate: vi.fn(), isPending: false }),
  useDeleteTask: () => ({ mutate: vi.fn(), isPending: false }),
}))

function withQuery(ui: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return <QueryClientProvider client={qc}>{ui}</QueryClientProvider>
}

describe('TaskPanel m15 alignment', () => {
  it('default tab 活跃 shows running task only', () => {
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    expect(screen.getByText('多角色播客合成')).toBeTruthy()
    expect(screen.queryByText('podcast-short')).toBeNull()
    expect(screen.queryByText('failed-tts')).toBeNull()
  })

  it('header summary shows correct counts', () => {
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    expect(screen.getByText(/1 运行中.*0 排队.*1 已完成/)).toBeTruthy()
  })

  it('switching to 失败 tab shows only failed task', () => {
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    fireEvent.click(screen.getByText('失败 1'))
    expect(screen.getByText('failed-tts')).toBeTruthy()
    expect(screen.getByText('CUDA out of memory')).toBeTruthy()
    expect(screen.queryByText('多角色播客合成')).toBeNull()
  })

  it('switching to 最近 tab shows completed', () => {
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    fireEvent.click(screen.getByText('最近 1'))
    expect(screen.getByText('podcast-short')).toBeTruthy()
  })

  it('hidden when open=false', () => {
    render(withQuery(<TaskPanel open={false} onClose={() => {}} />))
    expect(screen.queryByText('任务面板')).toBeNull()
  })
})

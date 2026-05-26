import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import TaskPanel, { sortTasks } from './TaskPanel'
import { ALL_TASK_STATUSES, usePanelStore } from '../../stores/panel'
import type { ExecutionTask } from '../../api/tasks'

// PR-5 ComfyUI 对齐:dock/float + 任务列表 + 缩略图 + 右键菜单。
const mockTasks = [
  {
    id: 'r1', workflow_name: 'flux2-人物立绘', status: 'running',
    nodes_done: 1, nodes_total: 3, duration_ms: null,
    created_at: new Date().toISOString(),
    error: null, result: null, current_node: 'flux2_ksampler', task_type: null,
    image_width: null, image_height: null,
  },
  {
    id: 'c1', workflow_name: 'flux2-狐狸', status: 'completed',
    nodes_done: 3, nodes_total: 3, duration_ms: 14025,
    created_at: new Date().toISOString(),
    error: null, result: null, current_node: null, task_type: 'image',
    image_width: 1024, image_height: 1024,
    output_thumbnails: ['/files/outputs/c1/0.webp'],
  },
  {
    id: 'f1', workflow_name: 'tts-旁白', status: 'failed',
    nodes_done: 0, nodes_total: 2, duration_ms: 500,
    created_at: new Date().toISOString(),
    error: 'OOM', result: null, current_node: null, task_type: null,
    image_width: null, image_height: null,
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

describe('TaskPanel — ComfyUI 对齐(PR-5)', () => {
  // 每个 case 跑前把 filter/sort 复位,避免上一个 case 漏掉的 state 污染。
  beforeEach(() => {
    usePanelStore.setState({
      taskFilterStatuses: new Set(ALL_TASK_STATUSES),
      taskSortKey: 'created',
      taskSortDir: 'desc',
      taskPanelMode: 'dock',
    })
  })

  it('header 运行中时显示「N 个正在运行」(对齐 ComfyUI 截图)', () => {
    render(withQuery(<TaskPanel open onClose={() => {}} />))
    expect(screen.getByRole('heading', { name: /1 个正在运行/ })).toBeTruthy()
  })

  it('「全部」tab 显示所有任务,「已完成」tab 过滤掉 running', () => {
    render(withQuery(<TaskPanel open onClose={() => {}} />))
    // 全部:3 项
    expect(screen.getByText('flux2-人物立绘')).toBeTruthy()
    expect(screen.getByText('flux2-狐狸')).toBeTruthy()
    expect(screen.getByText('tts-旁白')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: '已完成' }))
    // 已完成:running 不见,completed/failed 还在
    expect(screen.queryByText('flux2-人物立绘')).toBeNull()
    expect(screen.getByText('flux2-狐狸')).toBeTruthy()
    expect(screen.getByText('tts-旁白')).toBeTruthy()
  })

  it('运行中任务有 中止 按钮(端到端真生效经 #148 PR-3 cancel 桥修复)', () => {
    render(withQuery(<TaskPanel open onClose={() => {}} />))
    expect(screen.getByRole('button', { name: '中止任务' })).toBeTruthy()
  })

  it('已完成 image 任务渲染缩略图', () => {
    render(withQuery(<TaskPanel open onClose={() => {}} />))
    const imgs = document.querySelectorAll('img[src*="/files/outputs/c1/"]')
    expect(imgs.length).toBeGreaterThan(0)
  })

  it('右键已完成任务弹出 ContextMenu(查看图片/复制ID/重试/删除)', () => {
    render(withQuery(<TaskPanel open onClose={() => {}} />))
    const completedCard = screen.getByText('flux2-狐狸').closest('div[style*="cursor: context-menu"]') as HTMLElement
    expect(completedCard).toBeTruthy()
    fireEvent.contextMenu(completedCard)
    expect(screen.getByText('查看图片')).toBeTruthy()
    expect(screen.getByText('复制任务 ID')).toBeTruthy()
    expect(screen.getByText('删除')).toBeTruthy()
  })

  it('mode 切换按钮(dock↔float)更新 panel store', () => {
    usePanelStore.setState({ taskPanelMode: 'dock' })
    render(withQuery(<TaskPanel open onClose={() => {}} />))
    const toggle = screen.getByRole('button', { name: '切换浮窗' })
    fireEvent.click(toggle)
    expect(usePanelStore.getState().taskPanelMode).toBe('float')
    // 此时按钮 aria-label 应变为「切换停靠」
    expect(screen.getByRole('button', { name: '切换停靠' })).toBeTruthy()
  })

  it('open=false 不渲染', () => {
    const { container } = render(withQuery(<TaskPanel open={false} onClose={() => {}} />))
    expect(container.textContent).toBe('')
  })

  it('failed 任务有重试按钮', () => {
    render(withQuery(<TaskPanel open onClose={() => {}} />))
    expect(screen.getByRole('button', { name: '重试任务' })).toBeTruthy()
  })

  it('筛选 popover:取消勾「失败」后 failed 任务被过滤掉', () => {
    render(withQuery(<TaskPanel open onClose={() => {}} />))
    expect(screen.getByText('tts-旁白')).toBeTruthy()
    // 打开筛选 popover,点「失败」复选框反勾。
    fireEvent.click(screen.getByRole('button', { name: '筛选状态' }))
    fireEvent.click(screen.getByRole('menuitemcheckbox', { name: /失败/ }))
    expect(screen.queryByText('tts-旁白')).toBeNull()
    expect(screen.getByText('flux2-人物立绘')).toBeTruthy() // running 还在
    expect(screen.getByText('flux2-狐狸')).toBeTruthy() // completed 还在
  })

  it('筛选 popover「全选」按钮恢复所有状态', () => {
    usePanelStore.setState({ taskFilterStatuses: new Set(['running']) })
    render(withQuery(<TaskPanel open onClose={() => {}} />))
    expect(screen.queryByText('tts-旁白')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /筛选/ }))
    fireEvent.click(screen.getByRole('button', { name: '全选' }))
    expect(screen.getByText('tts-旁白')).toBeTruthy()
    expect(screen.getByText('flux2-狐狸')).toBeTruthy()
  })

  it('排序 popover:选「耗时(长→短)」更新 store', () => {
    render(withQuery(<TaskPanel open onClose={() => {}} />))
    fireEvent.click(screen.getByRole('button', { name: '排序' }))
    fireEvent.click(screen.getByRole('menuitemradio', { name: /耗时\(长→短\)/ }))
    expect(usePanelStore.getState().taskSortKey).toBe('duration')
    expect(usePanelStore.getState().taskSortDir).toBe('desc')
  })
})

describe('sortTasks — 纯函数排序', () => {
  const mk = (id: string, created_at: string, duration_ms: number | null): ExecutionTask =>
    ({
      id, workflow_id: null, workflow_name: id, status: 'completed',
      nodes_total: 1, nodes_done: 1, current_node: null,
      result: null, error: null, duration_ms,
      created_at, updated_at: created_at,
      task_type: null, image_width: null, image_height: null,
    }) as ExecutionTask

  it('created desc:新的在前', () => {
    const a = mk('a', '2026-01-01T00:00:00Z', 100)
    const b = mk('b', '2026-01-02T00:00:00Z', 200)
    expect(sortTasks([a, b], 'created', 'desc').map((t) => t.id)).toEqual(['b', 'a'])
  })

  it('created asc:旧的在前', () => {
    const a = mk('a', '2026-01-01T00:00:00Z', 100)
    const b = mk('b', '2026-01-02T00:00:00Z', 200)
    expect(sortTasks([a, b], 'created', 'asc').map((t) => t.id)).toEqual(['a', 'b'])
  })

  it('duration desc:长的在前,null duration 永远排末尾', () => {
    const a = mk('a', '2026-01-01T00:00:00Z', 100)
    const b = mk('b', '2026-01-01T00:00:00Z', 500)
    const c = mk('c', '2026-01-01T00:00:00Z', null)
    expect(sortTasks([a, b, c], 'duration', 'desc').map((t) => t.id)).toEqual(['b', 'a', 'c'])
  })

  it('duration asc:短的在前,null duration 仍排末尾(不被翻动)', () => {
    const a = mk('a', '2026-01-01T00:00:00Z', 100)
    const b = mk('b', '2026-01-01T00:00:00Z', 500)
    const c = mk('c', '2026-01-01T00:00:00Z', null)
    expect(sortTasks([a, b, c], 'duration', 'asc').map((t) => t.id)).toEqual(['a', 'b', 'c'])
  })
})

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import TaskPanel from './TaskPanel'

// Lane I: Buildkite 风结构 — per-runner 泳道区（hero）+ 最近完成列表。

const mockTasks = [
  {
    id: 'wf_done_1', workflow_name: 'flux2-人物立绘', status: 'completed',
    nodes_done: 2, nodes_total: 2, duration_ms: 34000, created_at: new Date().toISOString(),
    error: null, result: null, current_node: null, task_type: 'image',
    image_width: 1024, image_height: 1024, output_thumbnails: ['/files/outputs/wf_done_1/0.webp'],
  },
  {
    id: 'wf_done_2', workflow_name: 'cosy-旁白', status: 'completed',
    nodes_done: 1, nodes_total: 1, duration_ms: 8000, created_at: new Date().toISOString(),
    error: null, result: null, current_node: null, task_type: null,
    image_width: null, image_height: null,
  },
]

const mockRunners = [
  {
    id: 'runner-i', label: 'Runner-I', role: 'image', state: 'busy',
    current_task: { task_id: 'wf_run_x', workflow_name: 'flux2-人物立绘', progress: 0.6, detail: 'step 18/30' },
    queue: [{ task_id: 'q1', workflow_name: 'sd-背景', position: 1 }],
    restart_attempt: null, load_error: null, gpus: [0],
  },
  {
    id: 'runner-l', label: 'Runner-L', role: 'llm', state: 'idle',
    current_task: null, queue: [], restart_attempt: null, load_error: null, gpus: [1],
  },
]

vi.mock('../../api/tasks', () => ({
  useTasks: () => ({ data: mockTasks }),
  useCancelTask: () => ({ mutate: vi.fn(), isPending: false }),
  useRetryTask: () => ({ mutate: vi.fn(), isPending: false }),
  useDeleteTask: () => ({ mutate: vi.fn(), isPending: false }),
}))

let runnersData: unknown = mockRunners
const retryRunnerMutate = vi.fn()
vi.mock('../../api/runners', () => ({
  useRunners: () => ({ data: runnersData }),
  useRetryRunner: () => ({ mutate: retryRunnerMutate, isPending: false }),
}))

function withQuery(ui: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return <QueryClientProvider client={qc}>{ui}</QueryClientProvider>
}

describe('TaskPanel — Buildkite-style structure (DD3)', () => {
  beforeEach(() => {
    runnersData = mockRunners
    window.innerWidth = 1280
  })

  it('hidden when open=false', () => {
    render(withQuery(<TaskPanel open={false} onClose={() => {}} />))
    expect(screen.queryByText('任务面板')).toBeNull()
  })

  it('renders one lane per runner with label + role', () => {
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    expect(screen.getByText('Runner-I')).toBeTruthy()
    expect(screen.getByText('Runner-L')).toBeTruthy()
    expect(screen.getByText(/image/)).toBeTruthy()
    expect(screen.getByText(/llm/)).toBeTruthy()
  })

  it('busy lane shows current task name + progress detail', () => {
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    // 同名 task 同时出现在「正在跑」泳道 + 「最近完成」（mock 数据巧合），用 All。
    expect(screen.getAllByText('flux2-人物立绘').length).toBeGreaterThan(0)
    expect(screen.getByText('step 18/30')).toBeTruthy()
  })

  it('idle lane shows the "idle" text label (a11y: text not just color)', () => {
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    expect(screen.getByText('idle')).toBeTruthy()
  })

  it('renders a "最近完成" section with completed tasks', () => {
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    expect(screen.getByText('最近完成')).toBeTruthy()
    expect(screen.getByText('cosy-旁白')).toBeTruthy()
  })

  it('shows an empty-runner hint when /api/v1/runners returned []', () => {
    runnersData = []
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    expect(screen.getByText(/暂无 runner/)).toBeTruthy()
  })

  it('close button is a real button with aria-label', () => {
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    const closeBtn = screen.getByLabelText('关闭')
    expect(closeBtn.tagName).toBe('BUTTON')
  })

  it('drawer is fullscreen-width under 768px (DD7 responsive)', () => {
    window.innerWidth = 600
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    const drawer = screen.getByRole('complementary') // <aside>
    expect(drawer.style.width).toBe('100vw')
  })
})

describe('TaskPanel — runner abnormal states (DD5)', () => {
  beforeEach(() => {
    window.innerWidth = 1280
  })

  it('restarting lane shows "重启中 N/M" with attempt numbers', () => {
    runnersData = [
      {
        id: 'runner-t', label: 'Runner-T', role: 'tts', state: 'restarting',
        current_task: null, queue: [], restart_attempt: [2, 4], load_error: null, gpus: [0],
      },
    ]
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    expect(screen.getByText('重启中 2/4')).toBeTruthy()
  })

  it('load_failed lane shows the error text + a keyboard-reachable Retry button', () => {
    runnersData = [
      {
        id: 'runner-l', label: 'Runner-L', role: 'llm', state: 'load_failed',
        current_task: null, queue: [], restart_attempt: null,
        load_error: 'qwen3-35b OOM', gpus: [1],
      },
    ]
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    expect(screen.getByText(/qwen3-35b OOM/)).toBeTruthy()
    const retry = screen.getByRole('button', { name: '重试加载' })
    expect(retry.tagName).toBe('BUTTON')
  })
})

describe('TaskPanel — queue position expand (DD8)', () => {
  beforeEach(() => {
    window.innerWidth = 1280
    runnersData = [
      {
        id: 'runner-i', label: 'Runner-I', role: 'image', state: 'busy',
        current_task: { task_id: 'cur', workflow_name: 'cur-wf', progress: 0.3, detail: null },
        queue: [
          { task_id: 'q1', workflow_name: 'sd-背景', position: 1 },
          { task_id: 'q2', workflow_name: 'flux-头像', position: 2 },
        ],
        restart_attempt: null, load_error: null, gpus: [0],
      },
    ]
  })

  it('queue toggle is a real button with aria-expanded, collapsed by default', () => {
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    const toggle = screen.getByRole('button', { name: /排队 2/ })
    expect(toggle.getAttribute('aria-expanded')).toBe('false')
    expect(screen.queryByText('sd-背景')).toBeNull()
  })

  it('clicking the toggle expands an ordered list with #position numbers', () => {
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    const toggle = screen.getByRole('button', { name: /排队 2/ })
    fireEvent.click(toggle)
    expect(toggle.getAttribute('aria-expanded')).toBe('true')
    expect(screen.getByText('sd-背景')).toBeTruthy()
    expect(screen.getByText('flux-头像')).toBeTruthy()
    expect(screen.getByText('#1')).toBeTruthy()
    expect(screen.getByText('#2')).toBeTruthy()
  })
})

describe('TaskPanel — image thumbnail history (DD9)', () => {
  beforeEach(() => {
    window.innerWidth = 1280
    runnersData = []
  })

  it('image task with output_thumbnails renders an <img> thumbnail', () => {
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    // mockTasks[0] = flux2-人物立绘，task_type image，带 output_thumbnails
    const thumb = screen.getByAltText(/flux2-人物立绘/) as HTMLImageElement
    expect(thumb.tagName).toBe('IMG')
    expect(thumb.src).toContain('/files/outputs/wf_done_1/0.webp')
  })

  it('non-image completed task falls back to the status icon (no img)', () => {
    render(withQuery(<TaskPanel open={true} onClose={() => {}} />))
    // mockTasks[1] = cosy-旁白，task_type null —— 不应有 img alt 含它的名字
    expect(screen.queryByAltText(/cosy-旁白/)).toBeNull()
  })
})

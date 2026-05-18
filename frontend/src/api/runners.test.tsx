import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useRunners } from './runners'

vi.mock('./client', () => ({
  apiFetch: vi.fn(),
}))
import { apiFetch } from './client'

function wrapper() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={qc}>{children}</QueryClientProvider>
  )
}

describe('useRunners', () => {
  beforeEach(() => vi.clearAllMocks())

  it('adapts backend snapshot from /api/v1/monitor/runners to RunnerInfo', async () => {
    // 后端 shape:Lane H Task 6 + Lane K LLMRunner.health_snapshot +
    // Lane K follow-up RunnerClient.current_dispatch。
    vi.mocked(apiFetch).mockResolvedValue({
      runners: [
        // image: 存活 + 正在跑(current_task 非空) → busy
        { group_id: 'image', gpus: [0], running: true, restart_count: 0, pid: 12345,
          current_task: { task_id: 7001, workflow_name: 'flux2-人物立绘', node_id: 'n1',
                          node_type: 'image_generate', started_at: 0, progress: 0.6, detail: 'step 18/30' } },
        // tts: 不存活 + restart_count>0 → restarting
        { group_id: 'tts', gpus: [2], running: false, restart_count: 2, pid: null,
          current_task: null },
        // llm: 不存活 + restart_count=0 + 无 current_task → idle
        { group_id: 'llm', gpus: [1], running: false, restart_count: 0, pid: null,
          current_task: null },
        // image-idle: 存活但无 current_task → idle(不能误标 busy)
        { group_id: 'image', gpus: [0], running: true, restart_count: 0, pid: 99,
          current_task: null },
      ],
    })
    const { result } = renderHook(() => useRunners(), { wrapper: wrapper() })
    await waitFor(() => expect(result.current.data).toBeDefined())
    expect(result.current.data?.length).toBe(4)
    // image 真在跑 → busy + current_task
    expect(result.current.data?.[0]).toMatchObject({
      id: 'image', state: 'busy',
      current_task: { workflow_name: 'flux2-人物立绘', progress: 0.6, detail: 'step 18/30' },
    })
    // tts 不存活 + 重启 → restarting
    expect(result.current.data?.[1]).toMatchObject({
      id: 'tts', state: 'restarting', restart_attempt: [2, 4],
    })
    // llm 不存活 + 无重启 → idle
    expect(result.current.data?.[2]).toMatchObject({
      id: 'llm', state: 'idle', current_task: null,
    })
    // image 存活但无 task → idle(不是 busy!)
    expect(result.current.data?.[3]).toMatchObject({
      id: 'image', state: 'idle', current_task: null,
    })
  })

  it('degrades to empty array when endpoint 404s', async () => {
    vi.mocked(apiFetch).mockRejectedValue(
      Object.assign(new Error('Not Found'), { status: 404 }),
    )
    const { result } = renderHook(() => useRunners(), { wrapper: wrapper() })
    await waitFor(() => expect(result.current.data).toEqual([]))
  })
})

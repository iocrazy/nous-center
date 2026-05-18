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
    // 后端实际返回的 shape (Lane H Task 6 + Lane K LLMRunner.health_snapshot)。
    // 关键:后端 running=进程存活 ≠ 「正在跑任务」 —— 适配器把存活映射成 idle,
    // 而不是 busy(busy 需要 current_task 信号,GroupScheduler 尚未暴露)。
    vi.mocked(apiFetch).mockResolvedValue({
      runners: [
        { group_id: 'image', gpus: [0], running: true, restart_count: 0, pid: 12345 },
        { group_id: 'tts', gpus: [2], running: false, restart_count: 2, pid: null },
        { group_id: 'llm', gpus: [1], running: false, restart_count: 0, pid: null },
      ],
    })
    const { result } = renderHook(() => useRunners(), { wrapper: wrapper() })
    await waitFor(() => expect(result.current.data).toBeDefined())
    expect(result.current.data?.length).toBe(3)
    // image: 存活但无 current_task → idle(不是 busy!)
    expect(result.current.data?.[0]).toMatchObject({
      id: 'image', role: 'image', state: 'idle', gpus: [0],
    })
    // tts: 不存活 + restart_count>0 → restarting
    expect(result.current.data?.[1]).toMatchObject({
      id: 'tts', state: 'restarting', restart_attempt: [2, 4],
    })
    // llm: 不存活 + restart_count=0 → idle(未 preload,不是 failed)
    expect(result.current.data?.[2]).toMatchObject({
      id: 'llm', role: 'llm', state: 'idle', current_task: null, queue: [],
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

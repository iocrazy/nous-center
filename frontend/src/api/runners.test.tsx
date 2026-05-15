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

  it('returns runner list from /api/v1/runners', async () => {
    vi.mocked(apiFetch).mockResolvedValue([
      {
        id: 'runner-i',
        label: 'Runner-I',
        role: 'image',
        state: 'busy',
        current_task: { task_id: '7k2m', workflow_name: 'flux2-人物立绘', progress: 0.6, detail: 'step 18/30' },
        queue: [{ task_id: '9p1q', workflow_name: 'sd-背景', position: 1 }],
        restart_attempt: null,
        load_error: null,
      },
    ])
    const { result } = renderHook(() => useRunners(), { wrapper: wrapper() })
    await waitFor(() => expect(result.current.data).toBeDefined())
    expect(result.current.data?.[0].id).toBe('runner-i')
    expect(result.current.data?.[0].current_task?.detail).toBe('step 18/30')
  })

  it('degrades to empty array when endpoint 404s', async () => {
    vi.mocked(apiFetch).mockRejectedValue(
      Object.assign(new Error('Not Found'), { status: 404 }),
    )
    const { result } = renderHook(() => useRunners(), { wrapper: wrapper() })
    await waitFor(() => expect(result.current.data).toEqual([]))
  })
})

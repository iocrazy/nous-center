import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { apiFetch } from './client'
import {
  fetchAutostartPreview,
  preloadModelDetail,
  useSetServiceAutostart,
} from './services'

vi.mock('./client', () => ({ apiFetch: vi.fn() }))

function wrapper() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={qc}>{children}</QueryClientProvider>
  )
}

beforeEach(() => {
  vi.mocked(apiFetch).mockReset()
})

describe('autostart api', () => {
  it('fetchAutostartPreview GETs the preview endpoint', async () => {
    vi.mocked(apiFetch).mockResolvedValue({
      service_id: '7', name: 'svc', autostart: false, preload_models: [],
    })
    await fetchAutostartPreview('7')
    expect(vi.mocked(apiFetch).mock.calls[0][0]).toBe(
      '/api/v1/services/7/autostart-preview',
    )
  })

  it('useSetServiceAutostart POSTs /autostart with {enabled}', async () => {
    vi.mocked(apiFetch).mockResolvedValue({ autostart: true, preload_models: [] })
    const { result } = renderHook(() => useSetServiceAutostart(), { wrapper: wrapper() })
    result.current.mutate({ serviceId: '42', enabled: true })
    await waitFor(() => expect(apiFetch).toHaveBeenCalled())
    const [url, opts] = vi.mocked(apiFetch).mock.calls[0]
    expect(url).toBe('/api/v1/services/42/autostart')
    expect((opts as RequestInit).method).toBe('POST')
    expect(JSON.parse((opts as RequestInit).body as string)).toEqual({ enabled: true })
  })

  it('preloadModelDetail 拼显存/GPU,两者都没有就空串', () => {
    expect(preloadModelDetail({ name: 'a', vram_gb: 35, gpu: 1 })).toBe('35 GB · GPU 1')
    expect(preloadModelDetail({ name: 'a', gpu: [0, 1] })).toBe('GPU 0,1')
    expect(preloadModelDetail({ name: 'a' })).toBe('')
  })
})

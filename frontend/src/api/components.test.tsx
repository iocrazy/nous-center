import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor, act } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

vi.mock('./client', () => ({ apiFetch: vi.fn() }))
vi.mock('./useLiveChannel', () => ({ useLiveChannel: vi.fn() }))
import { apiFetch } from './client'
import { useComponents, componentStateKey, useComponentStateStore } from './components'

function wrapper() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={qc}>{children}</QueryClientProvider>
  )
}

describe('components api', () => {
  beforeEach(() => vi.clearAllMocks())

  it('componentStateKey matches backend format (file|device|dtype|)', () => {
    expect(componentStateKey({ file: '/m/u.safe', device: 'cuda:1', dtype: 'bfloat16' }))
      .toBe('/m/u.safe|cuda:1|bfloat16|')
  })

  it('useComponents fetches by role', async () => {
    vi.mocked(apiFetch).mockResolvedValue({ components: [{ filename: 'u.safe', abs_path: '/m/u.safe', size_mb: 1, quant_type: 'bf16', mtime: 0 }] })
    const { result } = renderHook(() => useComponents('unet'), { wrapper: wrapper() })
    await waitFor(() => expect(result.current.data).toBeDefined())
    expect(apiFetch).toHaveBeenCalledWith('/api/v1/components?role=unet')
    expect(result.current.data?.[0].abs_path).toBe('/m/u.safe')
  })

  it('store update is read back', () => {
    act(() => useComponentStateStore.getState().set('k1', 'loaded', null))
    expect(useComponentStateStore.getState().states['k1']).toEqual({ state: 'loaded', error: null })
  })
})

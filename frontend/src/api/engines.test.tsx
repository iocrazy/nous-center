import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { usePreloadComponent, useSetComponentResident, useSetSeedvr2Resident, useUnloadComponent, useSetGpu, useGpuGroups } from './engines'

vi.mock('./client', () => ({ apiFetch: vi.fn() }))
import { apiFetch } from './client'

vi.mock('../stores/toast', () => ({
  useToastStore: { getState: () => ({ add: vi.fn() }) },
}))

function wrapper() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={qc}>{children}</QueryClientProvider>
  )
}

describe('组件 L1 引擎库 hooks (PR-3b)', () => {
  beforeEach(() => vi.clearAllMocks())

  it('usePreloadComponent POST /component/preload，默认 bfloat16 非常驻', async () => {
    vi.mocked(apiFetch).mockResolvedValue({ status: 'accepted' })
    const { result } = renderHook(() => usePreloadComponent(), { wrapper: wrapper() })
    result.current.mutate({ name: 'component:clip:/m/qwen.safetensors' })
    await waitFor(() => expect(apiFetch).toHaveBeenCalled())
    const [url, opts] = vi.mocked(apiFetch).mock.calls[0]
    expect(url).toBe('/api/v1/engines/component/preload')
    expect(JSON.parse((opts as any).body)).toEqual({
      name: 'component:clip:/m/qwen.safetensors', dtype: 'bfloat16', resident: false,
    })
  })

  it('useUnloadComponent POST /component/unload，优先 state_key 精确匹配', async () => {
    vi.mocked(apiFetch).mockResolvedValue({ status: 'accepted' })
    const { result } = renderHook(() => useUnloadComponent(), { wrapper: wrapper() })
    result.current.mutate({ state_key: '/m/z.safetensors|cuda:1|bfloat16|' })
    await waitFor(() => expect(apiFetch).toHaveBeenCalled())
    const [url, opts] = vi.mocked(apiFetch).mock.calls[0]
    expect(url).toBe('/api/v1/engines/component/unload')
    expect(JSON.parse((opts as any).body)).toEqual({ state_key: '/m/z.safetensors|cuda:1|bfloat16|' })
  })

  it('useUnloadComponent 无 state_key → 回退 name+device/dtype', async () => {
    vi.mocked(apiFetch).mockResolvedValue({ status: 'accepted' })
    const { result } = renderHook(() => useUnloadComponent(), { wrapper: wrapper() })
    result.current.mutate({ name: 'component:vae:/m/vae.safetensors' })
    await waitFor(() => expect(apiFetch).toHaveBeenCalled())
    expect(JSON.parse((vi.mocked(apiFetch).mock.calls[0][1] as any).body)).toEqual({
      name: 'component:vae:/m/vae.safetensors', dtype: 'bfloat16',
    })
  })

  it('usePreloadComponent 传精度 + 常驻', async () => {
    vi.mocked(apiFetch).mockResolvedValue({ status: 'accepted' })
    const { result } = renderHook(() => usePreloadComponent(), { wrapper: wrapper() })
    result.current.mutate({ name: 'component:vae:/m/vae.safetensors', dtype: 'fp8_e4m3', resident: true })
    await waitFor(() => expect(apiFetch).toHaveBeenCalled())
    expect(JSON.parse((vi.mocked(apiFetch).mock.calls[0][1] as any).body)).toMatchObject({
      dtype: 'fp8_e4m3', resident: true,
    })
  })

  it('usePreloadComponent 指定 device → body 带 device(预加载到选定 GPU)', async () => {
    vi.mocked(apiFetch).mockResolvedValue({ status: 'accepted' })
    const { result } = renderHook(() => usePreloadComponent(), { wrapper: wrapper() })
    result.current.mutate({ name: 'component:clip:/m/x.safetensors', device: 'cuda:1' })
    await waitFor(() => expect(apiFetch).toHaveBeenCalled())
    const body = JSON.parse((vi.mocked(apiFetch).mock.calls[0][1] as any).body)
    expect(body).toMatchObject({ device: 'cuda:1', dtype: 'bfloat16' })
  })

  it('usePreloadComponent 不传 device → body 不含 device(后端 auto 选卡)', async () => {
    vi.mocked(apiFetch).mockResolvedValue({ status: 'accepted' })
    const { result } = renderHook(() => usePreloadComponent(), { wrapper: wrapper() })
    result.current.mutate({ name: 'component:clip:/m/x.safetensors' })
    await waitFor(() => expect(apiFetch).toHaveBeenCalled())
    const body = JSON.parse((vi.mocked(apiFetch).mock.calls[0][1] as any).body)
    expect(body).not.toHaveProperty('device')
  })

  it('useSetComponentResident 有 state_key 时优先用它(精确匹配)', async () => {
    vi.mocked(apiFetch).mockResolvedValue({ status: 'accepted' })
    const { result } = renderHook(() => useSetComponentResident(), { wrapper: wrapper() })
    result.current.mutate({ state_key: '/m/qwen.safetensors|cuda:1|bfloat16|', resident: true })
    await waitFor(() => expect(apiFetch).toHaveBeenCalled())
    const [url, opts] = vi.mocked(apiFetch).mock.calls[0]
    expect(url).toBe('/api/v1/engines/component/resident')
    expect(JSON.parse((opts as any).body)).toEqual({
      state_key: '/m/qwen.safetensors|cuda:1|bfloat16|', resident: true,
    })
  })

  it('useSetComponentResident 无 state_key 时回退 name+device/dtype', async () => {
    vi.mocked(apiFetch).mockResolvedValue({ status: 'accepted' })
    const { result } = renderHook(() => useSetComponentResident(), { wrapper: wrapper() })
    result.current.mutate({ name: 'component:clip:/m/x.safetensors', resident: false })
    await waitFor(() => expect(apiFetch).toHaveBeenCalled())
    expect(JSON.parse((vi.mocked(apiFetch).mock.calls[0][1] as any).body)).toEqual({
      name: 'component:clip:/m/x.safetensors', device: undefined, dtype: 'bfloat16', resident: false,
    })
  })

  it('useSetSeedvr2Resident POST /seedvr2/resident', async () => {
    vi.mocked(apiFetch).mockResolvedValue({ status: 'accepted' })
    const { result } = renderHook(() => useSetSeedvr2Resident(), { wrapper: wrapper() })
    result.current.mutate({ name: 'seedvr2:dit.safetensors', resident: true })
    await waitFor(() => expect(apiFetch).toHaveBeenCalled())
    const [url, opts] = vi.mocked(apiFetch).mock.calls[0]
    expect(url).toBe('/api/v1/engines/seedvr2/resident')
    expect(JSON.parse((opts as any).body)).toEqual({ name: 'seedvr2:dit.safetensors', resident: true })
  })
})

describe('GPU 分配 hooks（单卡 / 组）', () => {
  beforeEach(() => vi.clearAllMocks())

  it('useSetGpu 传 gpus 时走请求体 {gpus}（张量并行组）', async () => {
    vi.mocked(apiFetch).mockResolvedValue({ applied: false })
    const { result } = renderHook(() => useSetGpu(), { wrapper: wrapper() })
    result.current.mutate({ name: 'qwen3-35b', gpus: [0, 2] })
    await waitFor(() => expect(apiFetch).toHaveBeenCalled())
    const [url, opts] = vi.mocked(apiFetch).mock.calls[0]
    expect(url).toBe('/api/v1/engines/qwen3-35b/gpu')
    expect((opts as any).method).toBe('PATCH')
    expect(JSON.parse((opts as any).body)).toEqual({ gpus: [0, 2] })
  })

  it('useSetGpu 传单卡时仍走老的查询参数路径', async () => {
    vi.mocked(apiFetch).mockResolvedValue({ applied: false })
    const { result } = renderHook(() => useSetGpu(), { wrapper: wrapper() })
    result.current.mutate({ name: 'qwen3-35b', gpu: 1 })
    await waitFor(() => expect(apiFetch).toHaveBeenCalled())
    const [url, opts] = vi.mocked(apiFetch).mock.calls[0]
    expect(url).toBe('/api/v1/engines/qwen3-35b/gpu?gpu=1')
    expect((opts as any).body).toBeUndefined()
  })

  it('useSetGpu 只有一张卡的 gpus 退化成单卡路径（组至少两张）', async () => {
    vi.mocked(apiFetch).mockResolvedValue({ applied: false })
    const { result } = renderHook(() => useSetGpu(), { wrapper: wrapper() })
    result.current.mutate({ name: 'm', gpu: 2, gpus: [2] })
    await waitFor(() => expect(apiFetch).toHaveBeenCalled())
    expect(vi.mocked(apiFetch).mock.calls[0][0]).toBe('/api/v1/engines/m/gpu?gpu=2')
  })

  it('useGpuGroups 读 /api/v1/gpu/groups', async () => {
    vi.mocked(apiFetch).mockResolvedValue({ groups: [] })
    renderHook(() => useGpuGroups(), { wrapper: wrapper() })
    await waitFor(() => expect(apiFetch).toHaveBeenCalledWith('/api/v1/gpu/groups'))
  })
})

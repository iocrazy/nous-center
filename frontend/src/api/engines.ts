import { useQuery, useMutation, useQueryClient, keepPreviousData } from '@tanstack/react-query'
import { apiFetch } from './client'
import { useToastStore } from '../stores/toast'
import { useLiveChannel } from './useLiveChannel'

export interface EngineInfo {
  name: string
  display_name: string
  type: string
  status: 'loaded' | 'unloaded' | 'loading' | 'failed'
  gpu: number | number[]
  vram_gb: number
  resident: boolean
  local_path: string | null
  local_exists: boolean
  // Remote metadata
  organization: string | null
  model_size: string | null
  frameworks: string[] | null
  libraries: string[] | null
  license: string | null
  languages: string[] | null
  tags: string[] | null
  tensor_types: string[] | null
  description: string | null
  has_metadata: boolean
  auto_detected: boolean
  /**
   * False = the model was discovered on disk but no adapter is wired up
   * (image / video diffusers right now). UI must disable the load
   * button — the backend will 422 with a config hint anyway, but it's
   * cleaner to gate the button than to let users click a doomed action.
   */
  has_adapter: boolean
  loaded_gpu: number | null
  loaded_gpus: number[] | null
  status_detail: string | null
  /** image engines only: how many LoRAs the adapter knows about (loaded
   * value when the model is loaded, scanner total when unloaded). null
   * for non-image engines. */
  lora_count: number | null
}

/**
 * Subscribe to /ws/models and invalidate the ['engines'] query family.
 * Pure sync — no toasts. Use from canvas dropdowns that render inside
 * pages where useEngines() isn't mounted but still need live updates.
 *
 * The shared channel is URL-deduped so calling this from many sites
 * doesn't multiply socket connections.
 */
export function useEnginesLiveSync(): void {
  const qc = useQueryClient()
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const url = `${proto}//${window.location.host}/ws/models`
  useLiveChannel(url, {
    onMessage: (data) => {
      if (data.event === 'model_status') {
        qc.invalidateQueries({ queryKey: ['engines'] })
      }
    },
    onReconnect: () => qc.invalidateQueries({ queryKey: ['engines'] }),
  })
}

export function useEngines() {
  const qc = useQueryClient()
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const url = `${proto}//${window.location.host}/ws/models`

  useLiveChannel(url, {
    onMessage: (data) => {
      if (data.event !== 'model_status') return
      qc.invalidateQueries({ queryKey: ['engines'] })
      if (data.status === 'loaded') {
        useToastStore.getState().add(`${data.model} ${data.detail || '加载完成'}`, 'success')
      } else if (data.status === 'failed') {
        useToastStore.getState().add(`${data.model} 加载失败: ${data.detail}`, 'error')
      } else if (data.status === 'installed') {
        useToastStore.getState().add(`${data.model} 依赖安装完成`, 'success')
      } else if (data.status === 'install_failed') {
        useToastStore.getState().add(`${data.model} 依赖安装失败: ${data.detail}`, 'error')
      }
    },
    onReconnect: () => qc.invalidateQueries({ queryKey: ['engines'] }),
  })

  return useQuery({
    queryKey: ['engines'],
    queryFn: () => apiFetch<EngineInfo[]>('/api/v1/engines'),
    // Every status transition arrives via /ws/models; the periodic
    // refetch is a safety net (60s) for the rare case the socket is
    // wedged in some half-open state the browser hides from us.
    refetchInterval: (query) => query.state.status === 'error' ? 10_000 : 60_000,
    refetchOnWindowFocus: false,
    retry: false,
    // Show last-known engines instantly when navigating back to the page;
    // background refetch keeps them fresh. Backend serves cached body in <50ms
    // when warm, so the visible flicker window collapses.
    placeholderData: keepPreviousData,
    staleTime: 30_000,
  })
}

export function useLoadEngine() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) =>
      apiFetch(`/api/v1/engines/${name}/load`, { method: 'POST' }),
    onSuccess: (_, name) => {
      // Immediately invalidate to show "loading" status; terminal toast comes from WebSocket
      qc.invalidateQueries({ queryKey: ['engines'] })
      useToastStore.getState().add(`${name} 开始加载...`, 'info')
    },
    onError: (error: Error) => {
      useToastStore.getState().add(`加载失败: ${error.message}`, 'error')
    },
  })
}

export function useUnloadEngine() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) =>
      apiFetch(`/api/v1/engines/${name}/unload?force=true`, { method: 'POST' }),
    onSuccess: (_, name) => {
      qc.invalidateQueries({ queryKey: ['engines'] })
      useToastStore.getState().add(`${name} 已卸载`, 'success')
    },
    onError: (error: Error) => {
      useToastStore.getState().add(`卸载失败: ${error.message}`, 'error')
    },
  })
}

export function useSyncMetadata() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () =>
      apiFetch('/api/v1/engines/sync-metadata', { method: 'POST' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['engines'] }),
  })
}

export function useSetResident() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ name, resident }: { name: string; resident: boolean }) =>
      apiFetch(`/api/v1/engines/${name}/resident?resident=${resident}`, { method: 'PATCH' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['engines'] }),
    onError: (error: Error) => {
      useToastStore.getState().add(`设置失败: ${error.message}`, 'error')
    },
  })
}

export function useScanModels() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () =>
      apiFetch<{ count: number; models: string[] }>('/api/v1/engines/scan', { method: 'POST' }),
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ['engines'] })
      useToastStore.getState().add(`扫描完成，共 ${data.count} 个模型`, 'success')
    },
    onError: (error: Error) => {
      useToastStore.getState().add(`扫描失败: ${error.message}`, 'error')
    },
  })
}

export function useRefreshMetadata() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) =>
      apiFetch(`/api/v1/engines/${name}/refresh-metadata`, { method: 'POST' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['engines'] }),
  })
}

export interface GpuDevice {
  index: number
  name: string
  vram_gb: number
}

export function useGpus() {
  return useQuery({
    queryKey: ['gpus'],
    queryFn: () =>
      apiFetch<{ count: number; devices: GpuDevice[] }>('/api/v1/engines/gpus'),
  })
}

export function useSetGpu() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ name, gpu }: { name: string; gpu: number }) =>
      apiFetch(`/api/v1/engines/${name}/gpu?gpu=${gpu}`, { method: 'PATCH' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['engines'] }),
    onError: (error: Error) => {
      useToastStore.getState().add(`GPU 分配失败: ${error.message}`, 'error')
    },
  })
}

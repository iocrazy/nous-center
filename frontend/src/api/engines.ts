import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'
import { useToastStore } from '../stores/toast'

export interface EngineInfo {
  name: string
  display_name: string
  type: string
  status: 'loaded' | 'unloaded'
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
}

export function useEngines() {
  return useQuery({
    queryKey: ['engines'],
    queryFn: () => apiFetch<EngineInfo[]>('/api/v1/engines'),
    refetchInterval: (query) => query.state.status === 'error' ? 10_000 : 5000,
    refetchOnWindowFocus: false,
    retry: false,
  })
}

export function useLoadEngine() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) =>
      apiFetch(`/api/v1/engines/${name}/load`, { method: 'POST' }),
    onSuccess: (_, name) => {
      qc.invalidateQueries({ queryKey: ['engines'] })
      useToastStore.getState().add(`${name} 加载成功`, 'success')
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
      apiFetch(`/api/v1/engines/${name}/unload`, { method: 'POST' }),
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

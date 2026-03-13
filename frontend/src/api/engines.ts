import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface EngineInfo {
  name: string
  display_name: string
  type: string
  status: 'loaded' | 'unloaded'
  gpu: number
  vram_gb: number
  resident: boolean
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
    onSuccess: () => qc.invalidateQueries({ queryKey: ['engines'] }),
  })
}

export function useUnloadEngine() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) =>
      apiFetch(`/api/v1/engines/${name}/unload`, { method: 'POST' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['engines'] }),
  })
}

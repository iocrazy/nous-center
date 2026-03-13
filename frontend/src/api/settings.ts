import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface ServerSettings {
  local_models_path: string
  cosyvoice_repo_path: string
  indextts_repo_path: string
  gpu_image: number
  gpu_tts: number
  redis_url: string
  api_base_url: string
}

export function useServerSettings() {
  return useQuery({
    queryKey: ['settings'],
    queryFn: () => apiFetch<ServerSettings>('/api/v1/settings'),
    refetchOnWindowFocus: false,
    retry: false,
  })
}

export function useUpdateServerSettings() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: Partial<ServerSettings>) =>
      apiFetch<ServerSettings>('/api/v1/settings', {
        method: 'PUT',
        body: JSON.stringify(data),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['settings'] }),
  })
}

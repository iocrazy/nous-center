import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface VoicePreset {
  id: string
  name: string
  engine: string
  params: Record<string, unknown>
  reference_audio_path: string | null
  reference_text: string | null
  tags: string[]
  status: string
  endpoint_path: string | null
}

export function useVoicePresets() {
  return useQuery({
    queryKey: ['voices'],
    queryFn: () => apiFetch<VoicePreset[]>('/api/v1/voices'),
    refetchOnWindowFocus: false,
    retry: false,
  })
}

export function useCreatePreset() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: Omit<VoicePreset, 'id'>) =>
      apiFetch('/api/v1/voices', { method: 'POST', body: JSON.stringify(data) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['voices'] }),
  })
}

export function useDeletePreset() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch(`/api/v1/voices/${id}`, { method: 'DELETE' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['voices'] }),
  })
}

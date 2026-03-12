import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface PresetApiKey {
  id: string
  preset_id: string
  label: string
  key_prefix: string
  is_active: boolean
  usage_calls: number
  usage_chars: number
  last_used_at: string | null
  created_at: string
}

export interface PresetApiKeyCreated extends PresetApiKey {
  key: string
}

export function usePresetKeys(presetId: string | null) {
  return useQuery({
    queryKey: ['preset-keys', presetId],
    queryFn: () => apiFetch<PresetApiKey[]>(`/api/v1/presets/${presetId}/keys`),
    enabled: !!presetId,
    refetchOnWindowFocus: false,
    retry: false,
  })
}

export function useCreatePresetKey(presetId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (label: string) =>
      apiFetch<PresetApiKeyCreated>(`/api/v1/presets/${presetId}/keys`, {
        method: 'POST',
        body: JSON.stringify({ label }),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['preset-keys', presetId] }),
  })
}

export function useDeletePresetKey(presetId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (keyId: string) =>
      apiFetch(`/api/v1/presets/${presetId}/keys/${keyId}`, { method: 'DELETE' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['preset-keys', presetId] }),
  })
}

export function useUpdatePresetStatus(presetId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (status: 'active' | 'inactive') =>
      apiFetch(`/api/v1/presets/${presetId}/status`, {
        method: 'PATCH',
        body: JSON.stringify({ status }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['voices'] })
    },
  })
}

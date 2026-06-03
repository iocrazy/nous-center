import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface ServiceInstance {
  id: string
  source_type: string
  source_id: string
  source_name: string
  name: string
  type: string
  status: string
  endpoint_path: string | null
  params_override: Record<string, unknown>
  created_at: string
  updated_at: string
}

// --- Instance CRUD ---

export function useInstances(type?: string) {
  return useQuery({
    queryKey: ['instances', type],
    queryFn: () =>
      apiFetch<ServiceInstance[]>(
        type
          ? `/api/v1/instances?type=${type}`
          : '/api/v1/instances',
      ),
    refetchOnWindowFocus: false,
    retry: false,
  })
}

export function useInstance(instanceId: string | null) {
  return useQuery({
    queryKey: ['instance', instanceId],
    queryFn: () => apiFetch<ServiceInstance>(`/api/v1/instances/${instanceId}`),
    enabled: !!instanceId,
    refetchOnWindowFocus: false,
    retry: false,
  })
}

export function useCreateInstance() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: { source_type: string; source_id: string; name: string; type?: string; params_override?: Record<string, unknown> }) =>
      apiFetch<ServiceInstance>('/api/v1/instances', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['instances'] })
    },
  })
}

export function useDeleteInstance() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (instanceId: string) =>
      apiFetch(`/api/v1/instances/${instanceId}`, { method: 'DELETE' }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['instances'] })
    },
  })
}

export function useUpdateInstanceStatus(instanceId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (status: 'active' | 'inactive') =>
      apiFetch(`/api/v1/instances/${instanceId}/status`, {
        method: 'PATCH',
        body: JSON.stringify({ status }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['instance', instanceId] })
      qc.invalidateQueries({ queryKey: ['instances'] })
    },
  })
}


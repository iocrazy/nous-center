import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface WorkflowSummary {
  id: string
  name: string
  description: string | null
  is_template: boolean
  status: string
  created_at: string
  updated_at: string
}

export interface WorkflowFull extends WorkflowSummary {
  nodes: any[]
  edges: any[]
}

export function useWorkflows(isTemplate?: boolean) {
  const params = isTemplate != null ? `?is_template=${isTemplate}` : ''
  return useQuery({
    queryKey: ['workflows', isTemplate],
    queryFn: () => apiFetch<WorkflowSummary[]>(`/api/v1/workflows${params}`),
  })
}

export function useWorkflow(id: string | null) {
  return useQuery({
    queryKey: ['workflow', id],
    queryFn: () => apiFetch<WorkflowFull>(`/api/v1/workflows/${id}`),
    enabled: !!id,
  })
}

export function useCreateWorkflow() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: { name: string; nodes?: any[]; edges?: any[]; is_template?: boolean }) =>
      apiFetch<WorkflowFull>('/api/v1/workflows', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['workflows'] }),
  })
}

export function useSaveWorkflow() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, ...data }: { id: string; name?: string; nodes?: any[]; edges?: any[] }) =>
      apiFetch<WorkflowFull>(`/api/v1/workflows/${id}`, {
        method: 'PATCH',
        body: JSON.stringify(data),
      }),
    onSuccess: (_, vars) => {
      qc.invalidateQueries({ queryKey: ['workflow', vars.id] })
      qc.invalidateQueries({ queryKey: ['workflows'] })
    },
  })
}

export function useDeleteWorkflow() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch(`/api/v1/workflows/${id}`, { method: 'DELETE' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['workflows'] }),
  })
}

export function usePublishWorkflow() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<{ instance_id: string; endpoint: string }>(
        `/api/v1/workflows/${id}/publish`,
        { method: 'POST' }
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['workflows'] })
    },
  })
}

export function useUnpublishWorkflow() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch(`/api/v1/workflows/${id}/unpublish`, { method: 'POST' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['workflows'] }),
  })
}

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface ExecutionTask {
  id: string
  workflow_id: string | null
  workflow_name: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  nodes_total: number
  nodes_done: number
  current_node: string | null
  result: any
  error: string | null
  duration_ms: number | null
  created_at: string
  updated_at: string
}

export function useTasks(enabled = true) {
  return useQuery({
    queryKey: ['tasks'],
    queryFn: () => apiFetch<ExecutionTask[]>('/api/v1/tasks?limit=50'),
    refetchInterval: enabled ? 10000 : false,
    enabled,
  })
}

export function useCancelTask() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => apiFetch(`/api/v1/tasks/${id}/cancel`, { method: 'POST' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['tasks'] }),
  })
}

export function useRetryTask() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => apiFetch(`/api/v1/tasks/${id}/retry`, { method: 'POST' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['tasks'] }),
  })
}

export function useDeleteTask() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => apiFetch(`/api/v1/tasks/${id}`, { method: 'DELETE' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['tasks'] }),
  })
}

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'
import { useLiveChannel } from './useLiveChannel'

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
  /** Server-derived from result envelope. null until the run completes
   * with an image_output, then 'image'. Used for the task card badge. */
  task_type: 'image' | null
  image_width: number | null
  image_height: number | null
}

/** Fetch tasks once, then rely on WebSocket for updates. */
export function useTasks() {
  const qc = useQueryClient()
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const url = `${protocol}//${window.location.host}/ws/tasks`

  useLiveChannel(url, {
    onMessage: () => qc.invalidateQueries({ queryKey: ['tasks'] }),
    onReconnect: () => qc.invalidateQueries({ queryKey: ['tasks'] }),
  })

  return useQuery({
    queryKey: ['tasks'],
    queryFn: () => apiFetch<ExecutionTask[]>('/api/v1/tasks?limit=50'),
    // WebSocket handles real-time updates; this is a safety net for the
    // half-open-socket case the browser may not surface.
    refetchInterval: 60_000,
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

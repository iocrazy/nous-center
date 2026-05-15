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

  // —— V1.5 新增（Lane I，对齐 Lane B execution_tasks schema）——
  // 全部 optional：旧后端 payload 不带这些字段时为 undefined，
  // 组件按 undefined 降级（缩略图 → ImageIcon，runner 标识 → 不显示）。
  /** 落到哪个 hardware.yaml group（"image" / "llm-tp" / "tts"）。 */
  gpu_group?: string | null
  /** 实际执行的 runner 实例 id。 */
  runner_id?: string | null
  /** queued 态时的排队序号（1-based）；非 queued 态为 null/undefined。 */
  queue_position?: number | null
  /** image 任务的输出缩略图 URL 列表（数据源 outputs/{task_id}/，
   * 后端 Lane D 落盘 + /tasks 序列化时带出）。空 → 降级 ImageIcon。 */
  output_thumbnails?: string[] | null
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

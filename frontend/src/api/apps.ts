import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface ExposedParam {
  node_id: string
  param_key: string
  api_name: string
  param_type: string
  description: string
  required: boolean
  default: unknown
}

export interface WorkflowApp {
  id: string
  name: string
  display_name: string
  description: string
  workflow_id: string
  active: boolean
  exposed_inputs: ExposedParam[]
  exposed_outputs: ExposedParam[]
  call_count: number
  created_at: string | null
  updated_at: string | null
}

export interface PublishAppRequest {
  name: string
  display_name: string
  description: string
  exposed_inputs: ExposedParam[]
  exposed_outputs: ExposedParam[]
}

export function useApps() {
  return useQuery({
    queryKey: ['apps'],
    queryFn: () => apiFetch<WorkflowApp[]>('/api/v1/apps'),
  })
}

export function usePublishApp() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ workflowId, body }: { workflowId: string; body: PublishAppRequest }) =>
      apiFetch<WorkflowApp>(`/api/v1/workflows/${workflowId}/publish-app`, {
        method: 'POST',
        body: JSON.stringify(body),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['apps'] }),
  })
}

export function useUnpublishApp() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (appName: string) =>
      apiFetch(`/api/v1/apps/${appName}`, { method: 'DELETE' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['apps'] }),
  })
}

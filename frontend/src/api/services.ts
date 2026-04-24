import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'

// ---------- shared types ----------

export type ServiceStatus = 'active' | 'paused' | 'deprecated' | 'retired'
export type ServiceCategory = 'llm' | 'tts' | 'vl' | 'app'

export interface ExposedParam {
  key?: string
  input_name?: string
  node_id: string
  label?: string
  type?: string
  required?: boolean
  default?: unknown
  constraints?: Record<string, unknown>
  // legacy aliases (from rows backfilled from workflow_apps)
  api_name?: string
  param_key?: string
}

export interface ServiceRow {
  id: string
  name: string
  type: string
  status: ServiceStatus
  source_type: 'workflow' | 'preset' | 'model'
  source_id: string | null
  source_name: string | null
  category: ServiceCategory | null
  meter_dim: string | null
  workflow_id: string | null
  workflow_name: string | null  // 来自 backend LEFT JOIN workflows.name
  snapshot_hash: string | null
  snapshot_schema_version: number
  version: number
  created_at: string
  updated_at: string
}

export interface ServiceDetail extends ServiceRow {
  workflow_snapshot: Record<string, unknown>
  exposed_inputs: ExposedParam[]
  exposed_outputs: ExposedParam[]
}

export interface QuickProvisionBody {
  name: string
  category: ServiceCategory
  engine: string
  label?: string
  params?: Record<string, unknown>
}

export interface PublishBody {
  name: string
  label?: string
  category?: ServiceCategory
  meter_dim?: string
  exposed_inputs: ExposedParam[]
  exposed_outputs: ExposedParam[]
}

// ---------- queries ----------

export function useServices(params?: { category?: ServiceCategory; status?: ServiceStatus }) {
  const qs = new URLSearchParams()
  if (params?.category) qs.set('category', params.category)
  if (params?.status) qs.set('status', params.status)
  const q = qs.toString()
  return useQuery<ServiceRow[]>({
    queryKey: ['services', params?.category ?? null, params?.status ?? null],
    queryFn: () => apiFetch<ServiceRow[]>(`/api/v1/services${q ? `?${q}` : ''}`),
  })
}

export function useService(serviceId: string | number | null) {
  return useQuery<ServiceDetail>({
    queryKey: ['service', String(serviceId)],
    queryFn: () => apiFetch<ServiceDetail>(`/api/v1/services/${serviceId}`),
    enabled: serviceId != null,
  })
}

// ---------- mutations ----------

export function useQuickProvision() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: QuickProvisionBody) =>
      apiFetch<ServiceDetail>('/api/v1/services/quick-provision', {
        method: 'POST',
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['services'] })
    },
  })
}

export function usePublishWorkflow() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ workflowId, body }: { workflowId: string | number; body: PublishBody }) =>
      apiFetch<ServiceDetail>(`/api/v1/workflows/${workflowId}/publish`, {
        method: 'POST',
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['services'] })
    },
  })
}

export function usePatchService() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ serviceId, status }: { serviceId: string | number; status: ServiceStatus }) =>
      apiFetch<ServiceRow>(`/api/v1/services/${serviceId}`, {
        method: 'PATCH',
        body: JSON.stringify({ status }),
      }),
    onSuccess: (_, vars) => {
      qc.invalidateQueries({ queryKey: ['services'] })
      qc.invalidateQueries({ queryKey: ['service', String(vars.serviceId)] })
    },
  })
}

export function useDeleteService() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (serviceId: string | number) =>
      apiFetch(`/api/v1/services/${serviceId}`, { method: 'DELETE' }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['services'] })
    },
  })
}

// ---------- helpers ----------

export function paramKey(p: ExposedParam): string | undefined {
  return p.key ?? p.api_name
}

export function paramSlot(p: ExposedParam): string | undefined {
  return p.input_name ?? p.param_key
}

export function endpointFor(svc: Pick<ServiceRow, 'name' | 'category'>): string {
  return svc.category === 'llm'
    ? `POST /v1/chat/completions · model=${svc.name}`
    : `POST /v1/apps/${svc.name}/run`
}

export const NAME_RE = /^[a-z][a-z0-9-]{1,62}$/

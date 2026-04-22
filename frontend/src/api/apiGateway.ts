import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface MyService {
  instance_id: number
  instance_name: string
  category: string | null
  meter_dim: string | null
  grant_status: string
  total_units: number
  used_units: number
  remaining_units: number
}

export interface CatalogService {
  instance_id: number
  instance_name: string
  type: string
  category: string | null
  meter_dim: string | null
  status: string
  total_grants: number
  active_grants: number
  total_units: number
  used_units: number
  remaining_units: number
}

export interface Grant {
  id: number
  api_key_id: number
  instance_id: number
  instance_name: string
  status: string
  activated_at: string
  paused_at: string | null
  retired_at: string | null
}

export interface Pack {
  id: number
  grant_id: number
  name: string
  total_units: number
  used_units: number
  remaining_units: number
  expires_at: string | null
  purchased_at: string
  source: string
}

export interface AlertRule {
  id: number
  grant_id: number
  threshold_percent: number
  pack_id: number | null
  enabled: boolean
  last_notified_at: string | null
  created_at: string
}

// ---------- queries ----------

export function useMyServices() {
  return useQuery<MyService[]>({
    queryKey: ['my-services'],
    queryFn: () => apiFetch('/api/v1/services/me'),
  })
}

export function useServicesCatalog() {
  return useQuery<CatalogService[]>({
    queryKey: ['services-catalog'],
    queryFn: () => apiFetch('/api/v1/services/catalog'),
  })
}

export function useGrantsForKey(keyId: number | string | null) {
  return useQuery<Grant[]>({
    queryKey: ['grants', keyId],
    queryFn: () => apiFetch(`/api/v1/keys/${keyId}/grants`),
    enabled: keyId != null,
  })
}

export function usePacksForGrant(grantId: number | null) {
  return useQuery<Pack[]>({
    queryKey: ['packs', grantId],
    queryFn: () => apiFetch(`/api/v1/grants/${grantId}/packs`),
    enabled: grantId != null,
  })
}

export function useAlertsForGrant(grantId: number | null) {
  return useQuery<AlertRule[]>({
    queryKey: ['alerts', grantId],
    queryFn: () => apiFetch(`/api/v1/grants/${grantId}/alerts`),
    enabled: grantId != null,
  })
}

// ---------- mutations ----------

export function useCreateGrant() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ keyId, instanceId }: { keyId: number | string; instanceId: number }) =>
      apiFetch<Grant>(`/api/v1/keys/${keyId}/grants`, {
        method: 'POST',
        body: JSON.stringify({ instance_id: instanceId }),
      }),
    onSuccess: (_, vars) => {
      qc.invalidateQueries({ queryKey: ['grants', vars.keyId] })
      qc.invalidateQueries({ queryKey: ['my-services'] })
    },
  })
}

export function useUpdateGrantStatus() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ grantId, status }: { grantId: number; status: 'active' | 'paused' | 'retired' }) =>
      apiFetch<Grant>(`/api/v1/grants/${grantId}`, {
        method: 'PATCH',
        body: JSON.stringify({ status }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['grants'] })
      qc.invalidateQueries({ queryKey: ['my-services'] })
    },
  })
}

export function useCreatePack() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({
      grantId, name, totalUnits, expiresAt,
    }: {
      grantId: number; name: string; totalUnits: number; expiresAt?: string | null
    }) =>
      apiFetch<Pack>(`/api/v1/grants/${grantId}/packs`, {
        method: 'POST',
        body: JSON.stringify({
          name,
          total_units: totalUnits,
          expires_at: expiresAt ?? null,
        }),
      }),
    onSuccess: (_, vars) => {
      qc.invalidateQueries({ queryKey: ['packs', vars.grantId] })
      qc.invalidateQueries({ queryKey: ['my-services'] })
    },
  })
}

export function useCreateAlert() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({
      grantId, thresholdPercent, packId,
    }: {
      grantId: number; thresholdPercent: number; packId?: number | null
    }) =>
      apiFetch<AlertRule>(`/api/v1/grants/${grantId}/alerts`, {
        method: 'POST',
        body: JSON.stringify({
          threshold_percent: thresholdPercent,
          pack_id: packId ?? null,
        }),
      }),
    onSuccess: (_, vars) => {
      qc.invalidateQueries({ queryKey: ['alerts', vars.grantId] })
    },
  })
}

export function useToggleAlert() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ ruleId, enabled }: { ruleId: number; enabled: boolean }) =>
      apiFetch<AlertRule>(`/api/v1/alerts/${ruleId}`, {
        method: 'PATCH',
        body: JSON.stringify({ enabled }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['alerts'] })
    },
  })
}

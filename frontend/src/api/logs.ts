import { useQuery } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface LogItem {
  id: number
  timestamp: string
  [key: string]: unknown
}

export interface LogResponse {
  total: number
  items: LogItem[]
}

export interface LogQueryParams {
  limit?: number
  offset?: number
  search?: string
  since?: string
  level?: string
  type?: string
  method?: string
  status?: string
}

function buildParams(params: LogQueryParams): string {
  const sp = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v != null && v !== '') sp.set(k, String(v))
  }
  const str = sp.toString()
  return str ? `?${str}` : ''
}

/** Stable key excludes `since` (which changes every render) to avoid infinite refetch loops. */
function stableKey(params: LogQueryParams): Omit<LogQueryParams, 'since'> {
  const { since: _, ...rest } = params
  return rest
}

export function useRequestLogs(params: LogQueryParams = {}, enabled = true) {
  return useQuery({
    queryKey: ['logs', 'requests', stableKey(params)],
    queryFn: () => apiFetch<LogResponse>(`/api/v1/logs/requests${buildParams(params)}`),
    refetchInterval: enabled ? 3000 : false,
    enabled,
  })
}

export function useAppLogs(params: LogQueryParams = {}, enabled = true) {
  return useQuery({
    queryKey: ['logs', 'app', stableKey(params)],
    queryFn: () => apiFetch<LogResponse>(`/api/v1/logs/app${buildParams(params)}`),
    refetchInterval: enabled ? 3000 : false,
    enabled,
  })
}

export function useFrontendLogs(params: LogQueryParams = {}, enabled = true) {
  return useQuery({
    queryKey: ['logs', 'frontend', stableKey(params)],
    queryFn: () => apiFetch<LogResponse>(`/api/v1/logs/frontend${buildParams(params)}`),
    refetchInterval: enabled ? 3000 : false,
    enabled,
  })
}

export function useAuditLogs(params: LogQueryParams = {}, enabled = true) {
  return useQuery({
    queryKey: ['logs', 'audit', stableKey(params)],
    queryFn: () => apiFetch<LogResponse>(`/api/v1/logs/audit${buildParams(params)}`),
    refetchInterval: enabled ? 3000 : false,
    enabled,
  })
}

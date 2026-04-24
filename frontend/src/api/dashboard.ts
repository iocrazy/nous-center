import { useQuery } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface TopServiceRow {
  service_name: string
  calls: number
  percent: number
}

export interface AlertItem {
  id: number
  grant_id: number
  service_name: string | null
  threshold_percent: number
  last_notified_at: string | null
  severity: 'warn' | 'err'
}

export interface DashboardSummary {
  today_calls: number
  today_calls_delta_pct: number | null
  month_tokens: number
  month_tokens_quota: number | null
  month_tokens_used_pct: number | null
  active_alerts_count: number
  active_alerts_top_label: string | null
  api_key_count: number
  service_count: number
  unbound_key_count: number
  top_services_today: TopServiceRow[]
  recent_alerts: AlertItem[]
}

export function useDashboardSummary() {
  return useQuery<DashboardSummary>({
    queryKey: ['dashboard-summary'],
    queryFn: () => apiFetch('/api/v1/dashboard/summary'),
    refetchInterval: 30_000,
  })
}

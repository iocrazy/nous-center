import { useQuery } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface UsageSummary {
  days: number
  period_start: string
  period_end: string
  total_calls: number
  total_tokens: number
  prompt_tokens: number
  completion_tokens: number
  tts_characters: number
  avg_latency_ms: number | null
  error_rate: number | null
  prev_total_calls: number
  prev_total_tokens: number
}

export interface TimeseriesPoint {
  date: string
  by_service: Record<string, number>
}

export interface Timeseries {
  days: number
  points: TimeseriesPoint[]
  top_services: string[]
}

export interface TopKey {
  api_key_id: number
  label: string | null
  key_prefix: string | null
  mode: 'legacy' | 'm:n'
  calls: number
  tokens: number
  avg_latency_ms: number | null
}

export interface TopKeys {
  days: number
  rows: TopKey[]
}

export function useUsageSummary(days: number) {
  return useQuery<UsageSummary>({
    queryKey: ['usage-summary', days],
    queryFn: () => apiFetch(`/api/v1/usage/summary?days=${days}`),
  })
}

export function useUsageTimeseries(days: number) {
  return useQuery<Timeseries>({
    queryKey: ['usage-timeseries', days],
    queryFn: () => apiFetch(`/api/v1/usage/timeseries?days=${days}`),
  })
}

export function useUsageTopKeys(days: number, limit = 10) {
  return useQuery<TopKeys>({
    queryKey: ['usage-top-keys', days, limit],
    queryFn: () => apiFetch(`/api/v1/usage/top-keys?days=${days}&limit=${limit}`),
  })
}

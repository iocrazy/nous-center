import { useQuery } from '@tanstack/react-query'
import { apiFetch } from './client'

export type ComponentStatus = 'operational' | 'idle' | 'degraded' | 'down'
export type DayStatus = ComponentStatus | 'nodata'

export interface StatusDay {
  date: string
  uptime_pct: number | null
  status: DayStatus
  samples: number
}

export interface StatusComponent {
  key: string
  name: string
  status: ComponentStatus
  uptime_7d: number | null
  days: StatusDay[]
}

export interface StatusSnapshot {
  overall: ComponentStatus
  updated_at: string
  components: StatusComponent[]
}

export function useStatus() {
  return useQuery<StatusSnapshot>({
    queryKey: ['status-snapshot'],
    queryFn: () => apiFetch('/api/v1/status'),
    refetchInterval: 15_000, // 状态页 15s 自动刷新
  })
}

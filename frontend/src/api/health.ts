import { useQuery } from '@tanstack/react-query'
import { apiFetch } from './client'

export interface HealthStartup {
  resident_total: number
  resident_loaded: number
  preloading: boolean
}

export interface Health {
  status: string
  database?: string
  gpus?: number
  models_loaded?: number
  load_failures?: Record<string, string>
  startup?: HealthStartup
}

/** 轮询 /health(unauth)。预加载中 3s 一次(给「模型加载中」横幅及时更新),
 *  否则 20s 一次(低开销)。 */
export function useHealth() {
  return useQuery<Health>({
    queryKey: ['health'],
    queryFn: () => apiFetch('/health'),
    refetchInterval: (q) => (q.state.data?.startup?.preloading ? 3000 : 20000),
  })
}

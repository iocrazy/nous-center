import { useQuery } from '@tanstack/react-query'
import { apiFetch } from './client'

// 双轨收敛(#3):legacy /api/v1/instances 已删,统一走 v3 /api/v1/services。
// 保留本文件的 hook 名/形状,组件零改动。原有 create/delete/updateStatus 三个
// hook 无任何组件使用(死代码),一并删除。
export interface ServiceInstance {
  id: string
  source_type: string
  source_id: string | null
  source_name: string | null
  name: string
  type: string
  status: string
  created_at: string
  updated_at: string
}

// --- 列表/详情(读)统一到 v3 /services ---

export function useInstances(type?: string) {
  return useQuery({
    queryKey: ['instances', type],
    queryFn: async () => {
      // v3 /services 不按 type 过滤 → 客户端过滤,精确复刻 legacy ?type=<t>
      // (按 ServiceInstance.type 匹配;preset id 之类匹配不到即空,行为不变)。
      const all = await apiFetch<ServiceInstance[]>('/api/v1/services')
      return type ? all.filter((s) => s.type === type) : all
    },
    refetchOnWindowFocus: false,
    retry: false,
  })
}

export function useInstance(instanceId: string | null) {
  return useQuery({
    queryKey: ['instance', instanceId],
    queryFn: () => apiFetch<ServiceInstance>(`/api/v1/services/${instanceId}`),
    enabled: !!instanceId,
    refetchOnWindowFocus: false,
    retry: false,
  })
}

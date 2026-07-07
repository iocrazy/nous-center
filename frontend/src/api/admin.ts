import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'

export type AdminMe = {
  login_required: boolean
  authenticated: boolean
}

export const ADMIN_ME_KEY = ['admin', 'me'] as const

export function useAdminMe() {
  return useQuery({
    queryKey: ADMIN_ME_KEY,
    queryFn: () => apiFetch<AdminMe>('/sys/admin/me'),
    // /sys/admin/me is light and the gate is the source of truth — keep it
    // fresh enough that a 30-day cookie expiring mid-session flips us to login
    // within a minute, without polling so often it shows up in the network tab.
    staleTime: 60_000,
    refetchOnWindowFocus: true,
    retry: false,
  })
}

export function useAdminLogin() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (password: string) =>
      apiFetch<{ ok: true }>('/sys/admin/login', {
        method: 'POST',
        body: JSON.stringify({ password }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ADMIN_ME_KEY })
    },
  })
}

export function useAdminLogout() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => apiFetch<{ ok: true }>('/sys/admin/logout', { method: 'POST' }),
    onSuccess: () => {
      // Drop every cached query so the next admin (or post-relogin session)
      // can't see stale list data via placeholderData.
      qc.clear()
      // 显式把 admin/me 置为未认证 —— AuthGate 立刻翻回登录页。原先只 clear()+invalidate
      // 靠 refetch 翻页,clear() 后 observer 刷新有时序竞态、不保证同步翻 authenticated,
      // 导致「点登出仍停在主页面」。setQueryData 同步写入,确定性触发 AuthGate 重渲染。
      qc.setQueryData<AdminMe>(ADMIN_ME_KEY, { login_required: true, authenticated: false })
      qc.invalidateQueries({ queryKey: ADMIN_ME_KEY })
    },
  })
}

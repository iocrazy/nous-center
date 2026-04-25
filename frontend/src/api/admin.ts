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
      qc.invalidateQueries({ queryKey: ADMIN_ME_KEY })
    },
  })
}

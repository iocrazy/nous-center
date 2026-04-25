import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'
import { ADMIN_ME_KEY } from './admin'

export type TotpStatus = { enabled: boolean; has_verified: boolean }

export type TotpRow = {
  id: number
  label: string
  created_at: string
  verified_at: string | null
  last_used_at: string | null
}

export type TotpSetupOut = {
  id: number
  label: string
  secret: string
  otpauth_url: string
}

const STATUS_KEY = ['totp', 'status'] as const
const LIST_KEY = ['totp', 'list'] as const

export function useTotpStatus() {
  return useQuery({
    queryKey: STATUS_KEY,
    queryFn: () => apiFetch<TotpStatus>('/sys/admin/totp/status'),
    staleTime: 60_000,
    retry: false,
  })
}

export function useTotpList() {
  return useQuery({
    queryKey: LIST_KEY,
    queryFn: () => apiFetch<TotpRow[]>('/sys/admin/totp'),
    staleTime: 30_000,
    retry: false,
  })
}

export function useTotpSetup() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (label: string) =>
      apiFetch<TotpSetupOut>('/sys/admin/totp/setup', {
        method: 'POST',
        body: JSON.stringify({ label }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: LIST_KEY })
    },
  })
}

export function useTotpVerify() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (params: { id: number; code: string }) =>
      apiFetch<{ ok: true }>('/sys/admin/totp/verify', {
        method: 'POST',
        body: JSON.stringify(params),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: LIST_KEY })
      qc.invalidateQueries({ queryKey: STATUS_KEY })
    },
  })
}

export function useDeleteTotp() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: number) =>
      apiFetch<void>(`/sys/admin/totp/${id}`, { method: 'DELETE' }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: LIST_KEY })
      qc.invalidateQueries({ queryKey: STATUS_KEY })
    },
  })
}

export function useTotpLogin() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (code: string) =>
      apiFetch<{ ok: true }>('/sys/admin/totp/login', {
        method: 'POST',
        body: JSON.stringify({ code }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ADMIN_ME_KEY })
    },
  })
}

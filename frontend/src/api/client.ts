import { NousApiError } from './errors'

const BASE = '' // proxied via vite config

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  })
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}))
    const reqId = resp.headers.get('x-request-id') ?? undefined
    throw new NousApiError(body, resp.status, reqId)
  }
  if (resp.status === 204) return undefined as T
  return resp.json()
}

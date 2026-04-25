import { NousApiError } from './errors'

const BASE = '' // proxied via vite config

// Set by main.tsx so apiFetch can poke React Query when the cookie expires
// mid-session and the gate starts handing back 401s.
type OnUnauthorized = () => void
let onUnauthorized: OnUnauthorized = () => {}
export function setUnauthorizedHandler(handler: OnUnauthorized) {
  onUnauthorized = handler
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    credentials: 'same-origin',
    ...init,
  })
  if (!resp.ok) {
    if (resp.status === 401) onUnauthorized()
    const body = await resp.json().catch(() => ({}))
    const reqId = resp.headers.get('x-request-id') ?? undefined
    throw new NousApiError(body, resp.status, reqId)
  }
  if (resp.status === 204) return undefined as T
  return resp.json()
}

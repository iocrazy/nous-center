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
  // spread 顺序是要害:`...init` 必须在前,headers/credentials 在后。反过来写的话,
  // 任何自带 headers 的调用方会把默认头整块顶掉 —— createPredictionAsync 只传了
  // `Prefer` 就丢掉 Content-Type,body 以 text/plain 发出,后端 422。
  const headers = new Headers(init?.headers)
  // FormData 的 Content-Type 必须由浏览器自己写(要带 multipart boundary),硬塞 JSON
  // 会让后端解不出 form 字段。
  if (!headers.has('Content-Type') && !(init?.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json')
  }
  const resp = await fetch(`${BASE}${path}`, {
    ...init,
    headers,
    credentials: init?.credentials ?? 'same-origin',
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

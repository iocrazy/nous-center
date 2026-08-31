import { afterEach, describe, expect, it, vi } from 'vitest'
import { apiFetch, setUnauthorizedHandler } from './client'
import { NousApiError } from './errors'

/**
 * 这些用例锁的是 apiFetch 的 init 合并契约。曾经 `...init` 展开在最后,导致任何
 * 自带 headers 的调用方(createPredictionAsync 的 `Prefer: respond-async`)丢掉
 * Content-Type,JSON body 被当 text/plain 发出去 → 后端 422。
 */

function mockFetch(resp: Partial<Response> = {}) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    headers: new Headers(),
    json: async () => ({}),
    ...resp,
  })
  globalThis.fetch = fetchMock as unknown as typeof fetch
  return fetchMock
}

/** 取出真正传给 fetch 的 init。 */
function initOf(fetchMock: ReturnType<typeof mockFetch>): RequestInit {
  return fetchMock.mock.calls[0][1] as RequestInit
}

const realFetch = globalThis.fetch

afterEach(() => {
  globalThis.fetch = realFetch
  setUnauthorizedHandler(() => {})
  vi.restoreAllMocks()
})

describe('apiFetch header 合并', () => {
  it('调用方自带 headers 时仍然带上默认 Content-Type', async () => {
    const fetchMock = mockFetch()

    await apiFetch('/v1/services/x/predictions', {
      method: 'POST',
      headers: { Prefer: 'respond-async' },
      body: JSON.stringify({ input: {} }),
    })

    const headers = new Headers(initOf(fetchMock).headers)
    expect(headers.get('Content-Type')).toBe('application/json')
  })

  it('调用方自带的 headers 本身不丢', async () => {
    const fetchMock = mockFetch()

    await apiFetch('/v1/services/x/predictions', {
      method: 'POST',
      headers: { Prefer: 'respond-async' },
      body: JSON.stringify({ input: {} }),
    })

    const headers = new Headers(initOf(fetchMock).headers)
    expect(headers.get('Prefer')).toBe('respond-async')
  })

  it('调用方显式指定 Content-Type 时以调用方为准', async () => {
    const fetchMock = mockFetch()

    // agents.ts / skills.ts 存 prompt 走的就是 text/plain。
    await apiFetch('/api/v1/agents/a/prompts/p.md', {
      method: 'PUT',
      headers: { 'Content-Type': 'text/plain' },
      body: 'hello',
    })

    const headers = new Headers(initOf(fetchMock).headers)
    expect(headers.get('Content-Type')).toBe('text/plain')
  })

  it('body 是 FormData 时不塞 Content-Type — 留给浏览器写 multipart boundary', async () => {
    const fetchMock = mockFetch()

    const fd = new FormData()
    fd.append('repo_url', 'https://example.com/pkg.git')
    await apiFetch('/api/v1/nodes/packages/install_git', { method: 'POST', body: fd })

    const headers = new Headers(initOf(fetchMock).headers)
    expect(headers.get('Content-Type')).toBeNull()
  })

  it('init 的其余字段照常透传', async () => {
    const fetchMock = mockFetch()

    await apiFetch('/api/v1/keys', { method: 'DELETE', body: '{"id":1}' })

    const init = initOf(fetchMock)
    expect(init.method).toBe('DELETE')
    expect(init.body).toBe('{"id":1}')
  })
})

describe('apiFetch credentials', () => {
  it('默认带 same-origin(后台鉴权靠 cookie)', async () => {
    const fetchMock = mockFetch()

    await apiFetch('/api/v1/keys')

    expect(initOf(fetchMock).credentials).toBe('same-origin')
  })

  it('调用方显式指定时不被覆盖', async () => {
    const fetchMock = mockFetch()

    await apiFetch('/api/v1/keys', { credentials: 'omit' })

    expect(initOf(fetchMock).credentials).toBe('omit')
  })
})

describe('apiFetch 响应处理', () => {
  it('204 返回 undefined 且不解析 body', async () => {
    const json = vi.fn()
    mockFetch({ status: 204, json })

    await expect(apiFetch('/api/v1/keys/1')).resolves.toBeUndefined()
    expect(json).not.toHaveBeenCalled()
  })

  it('非 2xx 抛 NousApiError 并带上 x-request-id', async () => {
    mockFetch({
      ok: false,
      status: 422,
      headers: new Headers({ 'x-request-id': 'req-1' }),
      json: async () => ({ error: { message: 'boom', type: 'invalid_request_error' } }),
    })

    await expect(apiFetch('/v1/services/x/predictions', { method: 'POST' })).rejects.toMatchObject({
      message: 'boom',
      httpStatus: 422,
      requestId: 'req-1',
    })
  })

  it('401 触发 unauthorized 回调', async () => {
    mockFetch({ ok: false, status: 401, headers: new Headers(), json: async () => ({}) })
    const onUnauthorized = vi.fn()
    setUnauthorizedHandler(onUnauthorized)

    await expect(apiFetch('/api/v1/keys')).rejects.toBeInstanceOf(NousApiError)
    expect(onUnauthorized).toHaveBeenCalledOnce()
  })
})

import { describe, it, expect } from 'vitest'
import { shouldRetry, MAX_RETRIES } from './retryPolicy'
import { NousApiError } from './errors'

function apiErr(status: number) {
  return new NousApiError({ error: { message: 'boom' } }, status)
}

describe('shouldRetry', () => {
  it('4xx 一律不重试 —— 请求本身不对,重发改变不了结果', () => {
    for (const s of [400, 401, 403, 404, 409, 422, 429, 499]) {
      expect(shouldRetry(0, apiErr(s)), `HTTP ${s}`).toBe(false)
    }
  })

  it('5xx 照旧重试到上限 —— 服务端故障可能自愈', () => {
    expect(shouldRetry(0, apiErr(500))).toBe(true)
    expect(shouldRetry(MAX_RETRIES - 1, apiErr(503))).toBe(true)
    expect(shouldRetry(MAX_RETRIES, apiErr(500))).toBe(false)
  })

  it('网络错误(没有 httpStatus)照旧重试 —— 断网/超时可能恢复', () => {
    expect(shouldRetry(0, new TypeError('Failed to fetch'))).toBe(true)
    expect(shouldRetry(MAX_RETRIES, new TypeError('Failed to fetch'))).toBe(false)
  })

  it('鸭子类型的错误对象也认 httpStatus(跨 bundle / 测试替身)', () => {
    expect(shouldRetry(0, { httpStatus: 404 })).toBe(false)
    expect(shouldRetry(0, { httpStatus: 502 })).toBe(true)
  })

  it('null / undefined / 字符串状态码不当成 HTTP 错误,走默认重试', () => {
    expect(shouldRetry(0, null)).toBe(true)
    expect(shouldRetry(0, undefined)).toBe(true)
    expect(shouldRetry(0, { httpStatus: '404' })).toBe(true)
  })
})

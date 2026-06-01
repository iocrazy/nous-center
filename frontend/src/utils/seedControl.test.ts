import { describe, it, expect, vi, afterEach } from 'vitest'
import { nextSeed } from './seedControl'

afterEach(() => vi.restoreAllMocks())

describe('nextSeed (ComfyUI control_after_generate)', () => {
  it('fixed: seed 不变', () => {
    expect(nextSeed('12345', 'fixed')).toBe('12345')
    expect(nextSeed('', 'fixed')).toBe('')
  })

  it('increment: +1(普通 seed)', () => {
    expect(nextSeed('100', 'increment')).toBe('101')
  })

  it('decrement: -1,不低于 0', () => {
    expect(nextSeed('100', 'decrement')).toBe('99')
    expect(nextSeed('0', 'decrement')).toBe('0')
  })

  it('increment 大 seed 不丢精度(BigInt,> 2^53)', () => {
    // 用户截图 seed=99611110155462212 > Number.MAX_SAFE_INTEGER(9007199254740991)
    expect(nextSeed('99611110155462212', 'increment')).toBe('99611110155462213')
    expect(nextSeed('99611110155462212', 'decrement')).toBe('99611110155462211')
  })

  it('randomize: 落在 [0, 2^53) 安全范围', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0.5)
    const s = nextSeed('100', 'randomize')
    const n = Number(s)
    expect(n).toBeGreaterThanOrEqual(0)
    expect(n).toBeLessThan(Number.MAX_SAFE_INTEGER)
    expect(Number.isInteger(n)).toBe(true)
  })

  it('空/非法 seed 时 increment 从 0 起算', () => {
    expect(nextSeed('', 'increment')).toBe('1')
    expect(nextSeed('abc', 'increment')).toBe('1')
    expect(nextSeed(null, 'increment')).toBe('1')
  })
})

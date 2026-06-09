import { describe, it, expect } from 'vitest'
import {
  formatCharCount,
  isOverLimit,
  filterTemplates,
  PROMPT_TEMPLATES,
  PROMPT_DEFAULT_MAX,
} from './promptLibrary'

describe('promptLibrary', () => {
  it('formatCharCount 千分位 + 默认上限', () => {
    expect(formatCharCount(0)).toBe('0 / 20,000')
    expect(formatCharCount(92)).toBe('92 / 20,000')
    expect(formatCharCount(1234)).toBe('1,234 / 20,000')
    expect(formatCharCount(5, 100)).toBe('5 / 100')
  })

  it('isOverLimit 边界', () => {
    expect(isOverLimit(PROMPT_DEFAULT_MAX)).toBe(false)
    expect(isOverLimit(PROMPT_DEFAULT_MAX + 1)).toBe(true)
    expect(isOverLimit(5, 5)).toBe(false)
    expect(isOverLimit(6, 5)).toBe(true)
  })

  it('filterTemplates 空查询返回全部', () => {
    expect(filterTemplates('')).toHaveLength(PROMPT_TEMPLATES.length)
    expect(filterTemplates('   ')).toHaveLength(PROMPT_TEMPLATES.length)
  })

  it('filterTemplates 按名称/场景匹配(大小写无关)', () => {
    const byName = filterTemplates('九宫格')
    expect(byName.some((t) => t.id === 'multi-angle-grid')).toBe(true)
    const byHint = filterTemplates('电商')
    expect(byHint.some((t) => t.id === 'product-clean')).toBe(true)
    expect(filterTemplates('zzz-no-match')).toHaveLength(0)
  })

  it('每个模板字段非空', () => {
    for (const t of PROMPT_TEMPLATES) {
      expect(t.id).toBeTruthy()
      expect(t.name).toBeTruthy()
      expect(t.hint).toBeTruthy()
      expect(t.prompt.length).toBeGreaterThan(10)
    }
  })
})

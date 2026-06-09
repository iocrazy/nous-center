import { describe, it, expect } from 'vitest'
import {
  computeGroupBounds,
  countGroupMembers,
  groupSubtitle,
  classifyNodeType,
  GROUP_PAD,
  GROUP_HEADER,
} from './groupGeometry'

describe('groupGeometry', () => {
  it('computeGroupBounds: 空成员返回 null', () => {
    expect(computeGroupBounds([])).toBeNull()
  })

  it('computeGroupBounds: 包围盒 + padding + 顶部留头', () => {
    const r = computeGroupBounds([
      { x: 100, y: 100, width: 200, height: 100 },
      { x: 400, y: 250, width: 200, height: 100 },
    ])!
    // minX 100, minY 100, maxX 600, maxY 350
    expect(r.x).toBe(100 - GROUP_PAD)
    expect(r.y).toBe(100 - GROUP_PAD - GROUP_HEADER)
    expect(r.width).toBe(500 + GROUP_PAD * 2)
    expect(r.height).toBe(250 + GROUP_PAD * 2 + GROUP_HEADER)
  })

  it('computeGroupBounds: 单成员', () => {
    const r = computeGroupBounds([{ x: 0, y: 0, width: 320, height: 160 }], 10, 20)!
    expect(r).toEqual({ x: -10, y: -30, width: 340, height: 200 })
  })

  it('classifyNodeType: text_input = 提示词', () => {
    expect(classifyNodeType('text_input')).toBe('prompt')
  })

  it('classifyNodeType: 未知类型 = other', () => {
    expect(classifyNodeType('___nope___')).toBe('other')
  })

  it('countGroupMembers + groupSubtitle: 混合', () => {
    // 用真实类型:text_input(prompt) + 未知(other)
    const counts = countGroupMembers(['text_input', 'text_input', '___nope___'])
    expect(counts.prompts).toBe(2)
    expect(counts.others).toBe(1)
    expect(groupSubtitle(counts)).toBe('2个提示词 · 1个节点 已成组')
  })

  it('groupSubtitle: 零项省略', () => {
    expect(groupSubtitle({ images: 1, prompts: 0, others: 0 })).toBe('1张图片 已成组')
    expect(groupSubtitle({ images: 2, prompts: 1, others: 0 })).toBe('2张图片 · 1个提示词 已成组')
    expect(groupSubtitle({ images: 0, prompts: 0, others: 0 })).toBe('空 已成组')
  })
})

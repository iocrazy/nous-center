import { describe, it, expect } from 'vitest'
import type { ExposedParam } from '../../api/services'
import { kindOf, applyKind } from './widgetKind'

const p = (over: Partial<ExposedParam>): ExposedParam => ({ node_id: 'n', input_name: 'x', ...over })

describe('widgetKind.kindOf', () => {
  it('detects select from enum', () => {
    expect(kindOf(p({ type: 'string', constraints: { enum: ['a', 'b'] } }))).toBe('select')
  })
  it('detects slider from numeric + min/max', () => {
    expect(kindOf(p({ type: 'number', constraints: { min: 0, max: 1 } }))).toBe('slider')
  })
  it('plain number / integer', () => {
    expect(kindOf(p({ type: 'number' }))).toBe('number')
    expect(kindOf(p({ type: 'integer' }))).toBe('integer')
  })
  it('boolean / image', () => {
    expect(kindOf(p({ type: 'boolean' }))).toBe('boolean')
    expect(kindOf(p({ type: 'image' }))).toBe('image')
  })
  it('string → text (single_line) vs textarea', () => {
    expect(kindOf(p({ type: 'string', constraints: { format: 'single_line' } }))).toBe('text')
    expect(kindOf(p({ type: 'string' }))).toBe('textarea')
  })
})

describe('widgetKind.applyKind', () => {
  it('text sets single_line, textarea clears it', () => {
    expect(applyKind(p({}), 'text').constraints).toEqual({ format: 'single_line' })
    expect(applyKind(p({}), 'textarea').constraints).toEqual({})
  })
  it('slider keeps existing range, defaults when missing', () => {
    expect(applyKind(p({ type: 'number', constraints: { min: 1, max: 9, step: 2 } }), 'slider').constraints).toEqual({ min: 1, max: 9, step: 2 })
    expect(applyKind(p({}), 'slider').constraints).toEqual({ min: 0, max: 100, step: 1 })
  })
  it('select preserves enum', () => {
    expect(applyKind(p({ constraints: { enum: ['a'] } }), 'select').constraints).toEqual({ enum: ['a'] })
  })
  it('round-trips type through kindOf', () => {
    expect(kindOf(applyKind(p({}), 'boolean'))).toBe('boolean')
    expect(kindOf(applyKind(p({}), 'integer'))).toBe('integer')
    expect(kindOf(applyKind(p({}), 'image'))).toBe('image')
  })
})

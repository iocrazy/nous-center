import { describe, it, expect } from 'vitest'
import {
  widgetToExposed,
  fallbackExposed,
  exposableRowsFor,
  dedupeKeys,
  layeredLayout,
  paramId,
} from './appEditorSchema'
import { DECLARATIVE_NODES } from '../../models/nodeRegistry'

describe('widgetToExposed', () => {
  it('slider → number with min/max/step; precision 0 → integer', () => {
    const p = widgetToExposed('n1', {
      name: 'temperature', label: '温度', widget: 'slider',
      min: 0, max: 2, step: 0.1, precision: 1, default: 0.7,
    })
    expect(p.type).toBe('number')
    expect(p.constraints).toEqual({ min: 0, max: 2, step: 0.1 })
    expect(p.input_name).toBe('temperature')

    const i = widgetToExposed('n1', {
      name: 'steps', label: '步数', widget: 'slider', min: 1, max: 100, step: 1, precision: 0,
    })
    expect(i.type).toBe('integer')
  })

  it('select → enum (+ enum_labels when labels differ)', () => {
    const p = widgetToExposed('n1', {
      name: 'mode', label: '模式', widget: 'select',
      options: [{ value: 'a', label: '甲' }, { value: 'b', label: '乙' }],
    })
    expect(p.constraints).toEqual({ enum: ['a', 'b'], enum_labels: { a: '甲', b: '乙' } })

    const plain = widgetToExposed('n1', {
      name: 'x', label: 'x', widget: 'select', options: ['p', 'q'],
    })
    expect(plain.constraints).toEqual({ enum: ['p', 'q'] })
  })

  it('checkbox/image/input/textarea map to expected types', () => {
    expect(widgetToExposed('n', { name: 'a', label: 'a', widget: 'checkbox' }).type).toBe('boolean')
    expect(widgetToExposed('n', { name: 'a', label: 'a', widget: 'image_upload' }).type).toBe('image')
    expect(widgetToExposed('n', { name: 'a', label: 'a', widget: 'input' }).constraints)
      .toEqual({ format: 'single_line' })
    expect(widgetToExposed('n', { name: 'a', label: 'a', widget: 'textarea' }).type).toBe('string')
  })

  it('prefers the node current value as default', () => {
    const p = widgetToExposed('n1', { name: 'system', label: 'sys', widget: 'textarea', default: 'd' },
      { id: 'n1', type: 'llm', data: { system: 'live' } })
    expect(p.default).toBe('live')
  })
})

describe('fallbackExposed', () => {
  it('infers type from value', () => {
    expect(fallbackExposed('n', 'k', true).type).toBe('boolean')
    expect(fallbackExposed('n', 'k', 3).type).toBe('integer')
    expect(fallbackExposed('n', 'k', 3.5).type).toBe('number')
    expect(fallbackExposed('n', 'k', 'hi').type).toBe('string')
  })
})

describe('exposableRowsFor', () => {
  it('uses DECLARATIVE_NODES widgets when defined (llm exists)', () => {
    expect(DECLARATIVE_NODES.llm).toBeTruthy()
    const rows = exposableRowsFor({ id: 'n1', type: 'llm', data: {} })
    const names = rows.map((r) => r.input_name)
    expect(names).toContain('system')
    expect(names).toContain('temperature')
  })

  it('falls back to data keys (skipping wired arrays) for unknown types', () => {
    const rows = exposableRowsFor({
      id: 'x', type: 'totally_unknown_node',
      data: { prompt: 'hi', upstream: ['n0', 0], count: 2 },
    })
    const names = rows.map((r) => r.input_name).sort()
    expect(names).toEqual(['count', 'prompt'])
  })
})

describe('dedupeKeys', () => {
  it('prefixes colliding keys with short node id', () => {
    const out = dedupeKeys([
      { node_id: 'aaaa1111', key: 'seed', input_name: 'seed' },
      { node_id: 'bbbb2222', key: 'seed', input_name: 'seed' },
    ])
    expect(out[0].key).toBe('seed')
    expect(out[1].key).toBe('bbbb_seed')
  })
})

describe('layeredLayout', () => {
  it('columns by longest-path depth', () => {
    const pos = layeredLayout(
      [{ id: 'a' }, { id: 'b' }, { id: 'c' }],
      [{ source: 'a', target: 'b' }, { source: 'b', target: 'c' }],
      { colGap: 100, rowGap: 50 },
    )
    expect(pos.a.x).toBe(0)
    expect(pos.b.x).toBe(100)
    expect(pos.c.x).toBe(200)
  })

  it('survives cycles without infinite loop', () => {
    const pos = layeredLayout(
      [{ id: 'a' }, { id: 'b' }],
      [{ source: 'a', target: 'b' }, { source: 'b', target: 'a' }],
    )
    expect(Object.keys(pos).sort()).toEqual(['a', 'b'])
  })
})

describe('paramId', () => {
  it('is node_id + input_name', () => {
    expect(paramId({ node_id: 'n1', input_name: 'seed' })).toBe('n1::seed')
  })
})

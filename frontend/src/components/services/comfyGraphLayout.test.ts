import { describe, it, expect } from 'vitest'
import { layoutComfyGraph, isComfyNodeRef, COL_WIDTH, ROW_HEIGHT } from './comfyGraphLayout'

const WF = {
  '1': { class_type: 'LoadImage', inputs: { image: 'a.png' } },
  '2': { class_type: 'Encode', inputs: { pixels: ['1', 0] } },
  '3': { class_type: 'SaveVideo', inputs: { latent: ['2', 0] } },
}

describe('layoutComfyGraph', () => {
  it('按拓扑深度分列,引用值成边', () => {
    const { nodes, edges } = layoutComfyGraph(WF)
    const byId = Object.fromEntries(nodes.map((n) => [n.id, n]))
    expect(byId['1'].x).toBeLessThan(byId['2'].x)
    expect(byId['2'].x).toBeLessThan(byId['3'].x)
    expect(edges).toContainEqual({ source: '1', target: '2' })
    expect(edges).toContainEqual({ source: '2', target: '3' })
  })

  it('x = 深度 * COL_WIDTH, y = 列内序 * ROW_HEIGHT', () => {
    const { nodes } = layoutComfyGraph(WF)
    const byId = Object.fromEntries(nodes.map((n) => [n.id, n]))
    expect(byId['1']).toMatchObject({ x: 0, y: 0 })
    expect(byId['2']).toMatchObject({ x: COL_WIDTH, y: 0 })
    expect(byId['3']).toMatchObject({ x: COL_WIDTH * 2, y: 0 })
  })

  it('同列多节点按序排 y', () => {
    const wf = {
      a: { class_type: 'A', inputs: {} },
      b: { class_type: 'B', inputs: {} },
    }
    const { nodes } = layoutComfyGraph(wf)
    const ys = nodes.map((n) => n.y).sort((x, y) => x - y)
    expect(ys).toEqual([0, ROW_HEIGHT])
  })

  it('每个节点带 class_type,usedCount 默认 0(由调用方回填)', () => {
    const { nodes } = layoutComfyGraph(WF)
    const n1 = nodes.find((n) => n.id === '1')!
    expect(n1.class_type).toBe('LoadImage')
    expect(n1.usedCount).toBe(0)
  })

  it('孤立节点(无引用)全部落在第 0 列', () => {
    const wf = {
      x: { class_type: 'X', inputs: { v: 1 } },
      y: { class_type: 'Y', inputs: { v: 2 } },
    }
    const { nodes, edges } = layoutComfyGraph(wf)
    expect(edges).toEqual([])
    expect(nodes.every((n) => n.x === 0)).toBe(true)
  })
})

describe('isComfyNodeRef', () => {
  const idSet = new Set(['1', '2', '3'])
  it('识别 [nodeId, slot] 引用', () => {
    expect(isComfyNodeRef(['1', 0], idSet)).toBe(true)
  })
  it('普通数值/字符串/数组值不算引用', () => {
    expect(isComfyNodeRef('a.png', idSet)).toBe(false)
    expect(isComfyNodeRef(42, idSet)).toBe(false)
    expect(isComfyNodeRef(['not-a-node', 0], idSet)).toBe(false)
    expect(isComfyNodeRef([1, 2, 3], idSet)).toBe(false)
  })
})

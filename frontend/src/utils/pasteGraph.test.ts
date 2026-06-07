import { describe, it, expect } from 'vitest'
import { buildPastedGraph } from './pasteGraph'

describe('buildPastedGraph', () => {
  // 确定性发号:n1, n2, ... 便于断言映射结果。
  function seqIds() {
    let i = 0
    return () => `n${++i}`
  }

  it('assigns new ids, offsets positions, deep-clones data', () => {
    const clip = {
      nodes: [
        { id: 'a', type: 'clip', data: { k: 1 }, position: { x: 10, y: 20 } },
      ],
      edges: [],
    }
    const out = buildPastedGraph(clip, 40, seqIds())
    expect(out.nodes).toHaveLength(1)
    expect(out.nodes[0].id).toBe('n1')
    expect(out.nodes[0].id).not.toBe('a')
    expect(out.nodes[0].position).toEqual({ x: 50, y: 60 })
    // 深拷贝:改输出不影响输入
    out.nodes[0].data.k = 999
    expect(clip.nodes[0].data.k).toBe(1)
  })

  it('remaps internal edge endpoints to the new node ids', () => {
    const clip = {
      nodes: [
        { id: 'clip', type: 'clip', data: {}, position: { x: 0, y: 0 } },
        { id: 'enc', type: 'enc', data: {}, position: { x: 100, y: 0 } },
      ],
      edges: [
        { source: 'clip', sourceHandle: 'CLIP', target: 'enc', targetHandle: 'CLIP' },
      ],
    }
    const out = buildPastedGraph(clip, 40, seqIds())
    expect(out.nodes.map((n) => n.id)).toEqual(['n1', 'n2'])
    expect(out.edges).toHaveLength(1)
    // 边端点重连到新 id(clip→n1, enc→n2),handle 保留
    expect(out.edges[0]).toMatchObject({
      source: 'n1',
      target: 'n2',
      sourceHandle: 'CLIP',
      targetHandle: 'CLIP',
    })
    // 新边自身也有新 id,不复用节点 id
    expect(out.edges[0].id).toBe('n3')
  })

  it('drops edges whose endpoints are not both in the clipboard', () => {
    const clip = {
      nodes: [{ id: 'a', type: 'x', data: {}, position: { x: 0, y: 0 } }],
      edges: [{ source: 'a', sourceHandle: 'o', target: 'missing', targetHandle: 'i' }],
    }
    const out = buildPastedGraph(clip, 0, seqIds())
    expect(out.edges).toHaveLength(0)
  })

  it('preserves width/height when present, defaults style otherwise', () => {
    const clip = {
      nodes: [
        { id: 'a', type: 'x', data: {}, position: { x: 0, y: 0 }, width: 400, height: 200 },
        { id: 'b', type: 'y', data: {}, position: { x: 0, y: 0 } },
      ],
      edges: [],
    }
    const out = buildPastedGraph(clip, 0, seqIds())
    expect(out.nodes[0].width).toBe(400)
    expect(out.nodes[0].height).toBe(200)
    expect(out.nodes[1].style).toEqual({ width: 320 })
  })
})

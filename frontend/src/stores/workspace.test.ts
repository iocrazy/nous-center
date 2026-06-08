import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

// apiFetch 走真网络 —— mock 掉,只断言「保存被触发」。
vi.mock('../api/client', () => ({
  apiFetch: vi.fn().mockResolvedValue({ id: 'db-new' }),
}))

import { useWorkspaceStore } from './workspace'
import { apiFetch } from '../api/client'

describe('workspace store — round3 #1/#2', () => {
  beforeEach(() => {
    vi.mocked(apiFetch).mockClear()
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it('getActiveWorkflow 在 activeTabId 失配时不抛(防画布白屏)', () => {
    useWorkspaceStore.setState({ activeTabId: 'does-not-exist' })
    expect(() => useWorkspaceStore.getState().getActiveWorkflow()).not.toThrow()
    const wf = useWorkspaceStore.getState().getActiveWorkflow()
    expect(wf).toBeDefined()
    expect(Array.isArray(wf.nodes)).toBe(true)
    expect(Array.isArray(wf.edges)).toBe(true)
  })

  it('removeEdge 后 undo 恢复该边(删边 undo 回归)', () => {
    const api = useWorkspaceStore.getState()
    api.addTab('undo-edge-test')
    api.setWorkflow({
      id: 'w', name: 'w',
      nodes: [
        { id: 'a', type: 'text_input', data: {}, position: { x: 0, y: 0 } },
        { id: 'b', type: 'text_output', data: {}, position: { x: 100, y: 0 } },
      ],
      edges: [{ id: 'e1', source: 'a', sourceHandle: 'text', target: 'b', targetHandle: 'text' }],
    })
    expect(useWorkspaceStore.getState().getActiveWorkflow().edges).toHaveLength(1)
    api.removeEdge('e1')
    expect(useWorkspaceStore.getState().getActiveWorkflow().edges).toHaveLength(0)
    api.undo()
    const edges = useWorkspaceStore.getState().getActiveWorkflow().edges
    expect(edges).toHaveLength(1)
    expect(edges[0].id).toBe('e1')
  })

  it('removeNode 后 undo 恢复节点 + 其连边', () => {
    const api = useWorkspaceStore.getState()
    api.addTab('undo-node-test')
    api.setWorkflow({
      id: 'w2', name: 'w2',
      nodes: [
        { id: 'a', type: 'text_input', data: {}, position: { x: 0, y: 0 } },
        { id: 'b', type: 'text_output', data: {}, position: { x: 100, y: 0 } },
      ],
      edges: [{ id: 'e1', source: 'a', sourceHandle: 'text', target: 'b', targetHandle: 'text' }],
    })
    api.removeNode('b')
    expect(useWorkspaceStore.getState().getActiveWorkflow().nodes).toHaveLength(1)
    expect(useWorkspaceStore.getState().getActiveWorkflow().edges).toHaveLength(0)
    api.undo()
    expect(useWorkspaceStore.getState().getActiveWorkflow().nodes).toHaveLength(2)
    expect(useWorkspaceStore.getState().getActiveWorkflow().edges).toHaveLength(1)
  })

  it('spliceNodeOnEdge:删原边 + 加节点 + 两条新边,单次 undo 全恢复', () => {
    const api = useWorkspaceStore.getState()
    api.addTab('splice-test')
    api.setWorkflow({
      id: 'w3', name: 'w3',
      nodes: [
        { id: 'a', type: 'text_input', data: {}, position: { x: 0, y: 0 } },
        { id: 'b', type: 'text_output', data: {}, position: { x: 200, y: 0 } },
      ],
      edges: [{ id: 'e1', source: 'a', sourceHandle: 'text', target: 'b', targetHandle: 'text' }],
    })
    api.spliceNodeOnEdge(
      { id: 'c', type: 'prompt_template', data: {}, position: { x: 100, y: 0 } },
      'e1',
      [
        { id: 'e2', source: 'a', sourceHandle: 'text', target: 'c', targetHandle: 'text' },
        { id: 'e3', source: 'c', sourceHandle: 'text', target: 'b', targetHandle: 'text' },
      ],
    )
    let wf = useWorkspaceStore.getState().getActiveWorkflow()
    expect(wf.nodes.map((n) => n.id).sort()).toEqual(['a', 'b', 'c'])
    expect(wf.edges.map((e) => e.id).sort()).toEqual(['e2', 'e3']) // e1 删,e2/e3 加
    // 单次 undo 全恢复(节点 c 去掉,原边 e1 回来)
    api.undo()
    wf = useWorkspaceStore.getState().getActiveWorkflow()
    expect(wf.nodes.map((n) => n.id).sort()).toEqual(['a', 'b'])
    expect(wf.edges.map((e) => e.id)).toEqual(['e1'])
  })

  it('per-tab autosave:编辑 tab B 不取消 tab A 的 pending save', async () => {
    vi.useFakeTimers()
    const api = useWorkspaceStore.getState()
    api.addTab('A')
    const idA = useWorkspaceStore.getState().activeTabId
    api.addTab('B')
    const idB = useWorkspaceStore.getState().activeTabId
    expect(idA).not.toBe(idB)

    // A 标脏(排 A 的定时器)→ 立刻 B 标脏(老实现会 clear 掉 A 的定时器)。
    useWorkspaceStore.getState().markDirty(idA)
    useWorkspaceStore.getState().markDirty(idB)

    await vi.advanceTimersByTimeAsync(2100)

    // 修好后:A、B 各自保存一次 = 2 次。老的单例定时器只会保存 B = 1 次。
    expect(vi.mocked(apiFetch)).toHaveBeenCalledTimes(2)
  })
})

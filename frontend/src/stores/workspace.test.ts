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

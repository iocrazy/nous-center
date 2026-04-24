import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render } from '@testing-library/react'
import CreateApiKeyDialog from './CreateApiKeyDialog'

// 与 CreateServiceDialog 同款 reset 回归测试 — 确认 useEffect deps
// 用 `.reset` 而不是 mutation 整体（不然每帧 re-render 都会被算成依赖变更）。

const resetSpy = vi.fn()
const useCreateApiKeyMock = vi.fn()
const useServicesMock = vi.fn()

vi.mock('../../api/keys', async () => {
  const actual = await vi.importActual<typeof import('../../api/keys')>('../../api/keys')
  return {
    ...actual,
    useCreateApiKey: () => useCreateApiKeyMock(),
  }
})

vi.mock('../../api/services', async () => {
  const actual = await vi.importActual<typeof import('../../api/services')>('../../api/services')
  return {
    ...actual,
    useServices: () => useServicesMock(),
  }
})

beforeEach(() => {
  resetSpy.mockReset()
  useServicesMock.mockReturnValue({ data: [] })
  useCreateApiKeyMock.mockImplementation(() => ({
    mutate: vi.fn(),
    mutateAsync: vi.fn(),
    reset: resetSpy,
    isPending: false,
    error: null,
  }))
})

describe('CreateApiKeyDialog reset effect', () => {
  it('does not loop on rerender when closed', async () => {
    const { rerender } = render(
      <CreateApiKeyDialog open={false} onClose={() => {}} />,
    )
    for (let i = 0; i < 10; i++) {
      rerender(<CreateApiKeyDialog open={false} onClose={() => {}} />)
    }
    await Promise.resolve()
    expect(resetSpy.mock.calls.length).toBeLessThanOrEqual(1)
  })

  it('does not call reset while open', async () => {
    const { rerender } = render(
      <CreateApiKeyDialog open={true} onClose={() => {}} />,
    )
    for (let i = 0; i < 5; i++) {
      rerender(<CreateApiKeyDialog open={true} onClose={() => {}} />)
    }
    await Promise.resolve()
    expect(resetSpy).not.toHaveBeenCalled()
  })
})

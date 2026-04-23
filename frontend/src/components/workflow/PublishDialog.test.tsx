import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render } from '@testing-library/react'
import PublishDialog from './PublishDialog'

const resetSpy = vi.fn()
const usePublishMock = vi.fn()

vi.mock('../../api/services', async () => {
  const actual = await vi.importActual<typeof import('../../api/services')>('../../api/services')
  return {
    ...actual,
    usePublishWorkflow: () => usePublishMock(),
  }
})

beforeEach(() => {
  resetSpy.mockReset()
  usePublishMock.mockImplementation(() => ({
    mutate: vi.fn(),
    mutateAsync: vi.fn(),
    reset: resetSpy,
    isPending: false,
    error: null,
  }))
})

describe('PublishDialog reset effect', () => {
  it('does not loop on rerender when closed (regression: previously depended on the whole useMutation result reference)', async () => {
    const { rerender } = render(
      <PublishDialog open={false} onClose={() => {}} workflowId="w1" nodes={[]} />,
    )
    for (let i = 0; i < 10; i++) {
      rerender(
        <PublishDialog open={false} onClose={() => {}} workflowId="w1" nodes={[]} />,
      )
    }
    await Promise.resolve()
    expect(resetSpy.mock.calls.length).toBeLessThanOrEqual(1)
  })

  it('does not call reset while open', async () => {
    const nodes = [{ id: 'n1', type: 'PrimitiveInput' }]
    const { rerender } = render(
      <PublishDialog open={true} onClose={() => {}} workflowId="w1" nodes={nodes} />,
    )
    for (let i = 0; i < 5; i++) {
      rerender(
        <PublishDialog open={true} onClose={() => {}} workflowId="w1" nodes={nodes} />,
      )
    }
    await Promise.resolve()
    expect(resetSpy).not.toHaveBeenCalled()
  })
})

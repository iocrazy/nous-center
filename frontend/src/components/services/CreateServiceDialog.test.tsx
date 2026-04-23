import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render } from '@testing-library/react'
import CreateServiceDialog from './CreateServiceDialog'

// Mocks for engines/services hooks. The point of this test is the
// useEffect-deps regression, not full render fidelity.

const resetSpy = vi.fn()
const useQuickProvisionMock = vi.fn()
const useEnginesMock = vi.fn()

vi.mock('../../api/services', async () => {
  const actual = await vi.importActual<typeof import('../../api/services')>('../../api/services')
  return {
    ...actual,
    useQuickProvision: () => useQuickProvisionMock(),
  }
})

vi.mock('../../api/engines', async () => {
  const actual = await vi.importActual<typeof import('../../api/engines')>('../../api/engines')
  return {
    ...actual,
    useEngines: () => useEnginesMock(),
  }
})

beforeEach(() => {
  resetSpy.mockReset()
  useEnginesMock.mockReturnValue({ data: [] })
  // Mimic React Query: each render returns a fresh result OBJECT but the
  // method identities (here: `reset`) are stable. This is the exact shape
  // that triggered the manual-gate infinite-loop bug when the effect
  // depended on the whole object instead of `.reset`.
  useQuickProvisionMock.mockImplementation(() => ({
    mutate: vi.fn(),
    mutateAsync: vi.fn(),
    reset: resetSpy,
    isPending: false,
    error: null,
  }))
})

describe('CreateServiceDialog reset effect', () => {
  it('does not loop (regression: useEffect used to depend on the whole useMutation result whose reference changes every render)', async () => {
    const { rerender } = render(<CreateServiceDialog open={false} onClose={() => {}} />)
    // Force a stack of rerenders that each return a NEW useMutation
    // result object. If the effect deps were unstable, reset() would be
    // invoked once per rerender; with the fix it fires only on the
    // initial close-pass.
    for (let i = 0; i < 10; i++) {
      rerender(<CreateServiceDialog open={false} onClose={() => {}} />)
    }
    // Allow React to flush any queued effects.
    await Promise.resolve()
    expect(resetSpy.mock.calls.length).toBeLessThanOrEqual(1)
  })

  it('does not call reset while open', async () => {
    const { rerender } = render(<CreateServiceDialog open={true} onClose={() => {}} />)
    for (let i = 0; i < 5; i++) {
      rerender(<CreateServiceDialog open={true} onClose={() => {}} />)
    }
    await Promise.resolve()
    expect(resetSpy).not.toHaveBeenCalled()
  })
})

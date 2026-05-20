import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ComponentStatusHeader } from './DeclarativeNode'
import { useComponentStateStore } from '../../api/components'

vi.mock('../../api/client', () => ({ apiFetch: vi.fn(() => Promise.resolve({ components: [] })) }))
vi.mock('../../api/useLiveChannel', () => ({ useLiveChannel: vi.fn() }))

function wrap(ui: React.ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>)
}

describe('ComponentStatusHeader', () => {
  beforeEach(() => useComponentStateStore.setState({ states: {} }))
  it('shows 未加载 (cold) by default', () => {
    wrap(<ComponentStatusHeader data={{ file: '/m/u.safe', device: 'cuda:1', dtype: 'bfloat16' }} />)
    expect(screen.getByText(/未加载/)).toBeInTheDocument()
  })
  it('shows 已加载 when store says loaded', () => {
    useComponentStateStore.getState().set('/m/u.safe|cuda:1|bfloat16|', 'loaded', null)
    wrap(<ComponentStatusHeader data={{ file: '/m/u.safe', device: 'cuda:1', dtype: 'bfloat16' }} />)
    expect(screen.getByText(/已加载/)).toBeInTheDocument()
  })
})

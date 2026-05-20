import { describe, it, expect, vi } from 'vitest'
import { render, screen, act } from '@testing-library/react'
import { ReactFlowProvider } from '@xyflow/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
vi.mock('../../api/client', () => ({ apiFetch: vi.fn(() => Promise.resolve({ components: [] })) }))
vi.mock('../../api/useLiveChannel', () => ({ useLiveChannel: vi.fn() }))
import DeclarativeNode from './DeclarativeNode'

function wrap(ui: React.ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={qc}><ReactFlowProvider>{ui}</ReactFlowProvider></QueryClientProvider>)
}

describe('cached hint', () => {
  it('shows (cached) on a cached image_generate completion', () => {
    wrap(<DeclarativeNode id="g" type="image_generate" data={{}} selected={false} {...({} as any)} />)
    act(() => {
      window.dispatchEvent(new CustomEvent('node-progress', { detail: { type: 'node_complete', node_id: 'g', duration_ms: 50, cached: true } }))
    })
    expect(screen.getByText(/\(cached\)/)).toBeInTheDocument()
  })

  it('does not show (cached) on a non-cached image_generate completion', () => {
    wrap(<DeclarativeNode id="h" type="image_generate" data={{}} selected={false} {...({} as any)} />)
    act(() => {
      window.dispatchEvent(new CustomEvent('node-progress', { detail: { type: 'node_complete', node_id: 'h', duration_ms: 1200, cached: false } }))
    })
    const doneText = screen.getByText(/完成/)
    expect(doneText.textContent).not.toContain('(cached)')
  })
})

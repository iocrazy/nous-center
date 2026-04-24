import { describe, it, expect, vi, beforeEach } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import ServicesList from './ServicesList'
import type { ServiceRow } from '../api/services'

// Hoisted mock state — vitest moves vi.mock calls to the top, so the
// module factories below close over these via the getter pattern.
const navigateSpy = vi.fn()
const useServicesMock = vi.fn()

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom')
  return { ...actual, useNavigate: () => navigateSpy }
})

vi.mock('../api/services', async () => {
  const actual = await vi.importActual<typeof import('../api/services')>('../api/services')
  return {
    ...actual,
    useServices: () => useServicesMock(),
  }
})

vi.mock('../components/services/CreateServiceDialog', () => ({
  default: () => null,
}))

function makeService(over: Partial<ServiceRow> = {}): ServiceRow {
  return {
    id: '1234567890',
    name: 'sample-svc',
    type: 'inference',
    status: 'active',
    source_type: 'workflow',
    source_id: '55',
    source_name: null,
    category: 'llm',
    meter_dim: 'tokens',
    workflow_id: '55',
    snapshot_hash: 'sha256:abc',
    snapshot_schema_version: 1,
    version: 1,
    created_at: '2026-04-23T00:00:00Z',
    updated_at: '2026-04-23T00:00:00Z',
    ...over,
  }
}

beforeEach(() => {
  navigateSpy.mockReset()
  useServicesMock.mockReset()
})

describe('ServicesList card click → navigate', () => {
  it('uses react-router useNavigate (regression: previously used window.history.pushState which never re-rendered the detail overlay)', () => {
    useServicesMock.mockReturnValue({
      data: [makeService({ id: '999', name: 'click-me' })],
      isLoading: false,
      error: null,
    })
    render(
      <MemoryRouter>
        <ServicesList />
      </MemoryRouter>,
    )
    // The card body and the explicit "详情" footer button are both wired
    // to the same navigate; click the body (named-region heuristic for
    // the click-me service).
    fireEvent.click(screen.getByText('click-me'))
    expect(navigateSpy).toHaveBeenCalledWith('/services/999')
  })

  it('falls back to onOpen prop when provided (no router involvement)', () => {
    useServicesMock.mockReturnValue({
      data: [makeService({ id: '777', name: 'override-me' })],
      isLoading: false,
      error: null,
    })
    const onOpen = vi.fn()
    render(
      <MemoryRouter>
        <ServicesList onOpen={onOpen} />
      </MemoryRouter>,
    )
    fireEvent.click(screen.getByText('override-me'))
    expect(onOpen).toHaveBeenCalledWith('777')
    expect(navigateSpy).not.toHaveBeenCalled()
  })

  it('"新建服务" split button opens a menu with both creation paths', () => {
    useServicesMock.mockReturnValue({ data: [], isLoading: false, error: null })
    render(
      <MemoryRouter>
        <ServicesList />
      </MemoryRouter>,
    )
    fireEvent.click(screen.getByRole('button', { name: /新建服务/ }))
    expect(screen.getByRole('menuitem', { name: /快速开通/ })).toBeInTheDocument()
    const fromWf = screen.getByRole('menuitem', { name: /从 Workflow 发布/ })
    expect(fromWf).toBeInTheDocument()
    fireEvent.click(fromWf)
    expect(navigateSpy).toHaveBeenCalledWith('/workflows')
  })

  it('renders the empty state CTA when there are no services', () => {
    useServicesMock.mockReturnValue({ data: [], isLoading: false, error: null })
    render(
      <MemoryRouter>
        <ServicesList />
      </MemoryRouter>,
    )
    expect(screen.getByText(/还没有服务/)).toBeInTheDocument()
  })
})

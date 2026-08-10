import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import ComfyBridgeSection from './ComfyBridgeSection'
import * as comfyTemplates from '../../api/comfyTemplates'

vi.mock('../../api/comfyTemplates', () => ({
  getComfyHealth: vi.fn(),
}))

function withQuery(ui: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return <QueryClientProvider client={qc}>{ui}</QueryClientProvider>
}

describe('ComfyBridgeSection', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })
  it('shows online status with queue depth and version', async () => {
    vi.mocked(comfyTemplates.getComfyHealth).mockResolvedValue({
      online: true,
      queue_depth: 3,
      version: '0.4.12',
      base_url: 'http://localhost:8188',
      timeout_s: 120,
    })

    render(
      withQuery(
        <ComfyBridgeSection />
      )
    )

    await waitFor(() => {
      expect(screen.getByText(/在线/)).toBeTruthy()
      expect(screen.getByText(/队列 3/)).toBeTruthy()
      expect(screen.getByText(/0.4.12/)).toBeTruthy()
    })

    expect(screen.getByText('http://localhost:8188')).toBeTruthy()
    expect(screen.getByText(/超时 120s/)).toBeTruthy()
  })

  it('shows offline status when online=false', async () => {
    vi.mocked(comfyTemplates.getComfyHealth).mockResolvedValue({
      online: false,
      queue_depth: 0,
      version: '',
      base_url: 'http://localhost:8188',
      timeout_s: 120,
    })

    render(
      withQuery(
        <ComfyBridgeSection />
      )
    )

    await waitFor(() => {
      expect(screen.getByText(/离线/)).toBeTruthy()
      expect(screen.getByText(/检查 sidecar 服务/)).toBeTruthy()
    })
  })

  it('shows error when fetch fails (network error)', async () => {
    vi.mocked(comfyTemplates.getComfyHealth).mockRejectedValue(
      new Error('Network error')
    )

    render(
      withQuery(
        <ComfyBridgeSection />
      )
    )

    await waitFor(() => {
      expect(screen.getByText('无法获取状态')).toBeTruthy()
    })
  })

  it('shows configuration note', async () => {
    vi.mocked(comfyTemplates.getComfyHealth).mockResolvedValue({
      online: true,
      queue_depth: 0,
      version: '0.4.12',
      base_url: 'http://localhost:8188',
      timeout_s: 60,
    })

    render(
      withQuery(
        <ComfyBridgeSection />
      )
    )

    await waitFor(() => {
      expect(screen.getByText(/backend\/.env/)).toBeTruthy()
      expect(screen.getByText(/NOUS_COMFY_URL/)).toBeTruthy()
      expect(screen.getByText(/NOUS_COMFY_TIMEOUT/)).toBeTruthy()
    })
  })

  it('converts timeout from seconds to hours when >= 3600', async () => {
    vi.mocked(comfyTemplates.getComfyHealth).mockResolvedValue({
      online: true,
      queue_depth: 0,
      version: '0.4.12',
      base_url: 'http://localhost:8188',
      timeout_s: 3600,
    })

    render(
      withQuery(
        <ComfyBridgeSection />
      )
    )

    await waitFor(() => {
      expect(screen.getByText(/超时 1h/)).toBeTruthy()
    })
  })

  it('preserves decimal hours for non-integer values (e.g., 5400s → 1.5h)', async () => {
    vi.mocked(comfyTemplates.getComfyHealth).mockResolvedValue({
      online: true,
      queue_depth: 0,
      version: '0.4.12',
      base_url: 'http://localhost:8188',
      timeout_s: 5400,
    })

    render(
      withQuery(
        <ComfyBridgeSection />
      )
    )

    await waitFor(() => {
      expect(screen.getByText(/超时 1\.5h/)).toBeTruthy()
    })
  })
})

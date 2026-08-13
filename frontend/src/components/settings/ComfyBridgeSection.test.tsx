import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import ComfyBridgeSection from './ComfyBridgeSection'
import * as comfyTemplates from '../../api/comfyTemplates'

vi.mock('../../api/comfyTemplates', () => ({
  getComfyHealth: vi.fn(),
  freeComfyVram: vi.fn(),
}))

const DEVICE = {
  name: 'cuda:0 NVIDIA RTX PRO 6000 Blackwell Workstation Edition : cudaMallocAsync',
  type: 'cuda',
  index: 0,
  vram_total: 102_000_000_000,
  vram_free: 37_700_000_000,
  vram_used: 64_300_000_000,      // 整卡已用(含同卡 vLLM 等别的进程)
  comfy_used: 60_000_000_000,     // ComfyUI 自占(torch reserved),「释放」能动的只有这块
  torch_vram_total: 60_000_000_000,
}

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
      devices: [],
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
      devices: [],
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
      devices: [],
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
      devices: [],
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
      devices: [],
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

  it('renders device name, used/total VRAM, and a release button', async () => {
    vi.mocked(comfyTemplates.getComfyHealth).mockResolvedValue({
      online: true,
      queue_depth: 0,
      version: '0.4.12',
      base_url: 'http://localhost:8188',
      timeout_s: 120,
      devices: [DEVICE],
    })

    render(withQuery(<ComfyBridgeSection />))

    await waitFor(() => {
      expect(screen.getByText(/NVIDIA RTX PRO 6000/)).toBeTruthy()
      expect(screen.getByText(/64\.3GB \/ 102\.0GB/)).toBeTruthy()
      expect(screen.getByText('释放显存')).toBeTruthy()
    })
  })

  it('clicking release button calls freeComfyVram and refetches health', async () => {
    vi.mocked(comfyTemplates.getComfyHealth).mockResolvedValue({
      online: true,
      queue_depth: 0,
      version: '0.4.12',
      base_url: 'http://localhost:8188',
      timeout_s: 120,
      devices: [DEVICE],
    })
    vi.mocked(comfyTemplates.freeComfyVram).mockResolvedValue({
      ok: true,
      settled: true,
      freed_bytes: 20_500_000_000,
      devices: [{ ...DEVICE, vram_free: 100_000_000_000, vram_used: 2_000_000_000 }],
    })

    render(withQuery(<ComfyBridgeSection />))

    const button = await screen.findByText('释放显存')
    fireEvent.click(button)

    await waitFor(() => {
      expect(comfyTemplates.freeComfyVram).toHaveBeenCalledTimes(1)
      // getComfyHealth: once on mount, once after the invalidated refetch
      expect(comfyTemplates.getComfyHealth).toHaveBeenCalledTimes(2)
    })
    // 落定时报出实际归还量,而不是让用户自己比对数字
    expect(await screen.findByText(/已释放 20\.5GB/)).toBeInTheDocument()
  })

  it('unsettled release says it is still unloading, not that it failed', async () => {
    // ComfyUI 的 /free 只设 flag,卸载晚几秒才发生;后端 6s 轮询没等到就 settled=false。
    // 这不是失败 —— UI 必须照实说"仍在进行",否则用户以为按钮坏了。
    vi.mocked(comfyTemplates.getComfyHealth).mockResolvedValue({
      online: true,
      queue_depth: 0,
      version: '0.4.12',
      base_url: 'http://localhost:8188',
      timeout_s: 120,
      devices: [DEVICE],
    })
    vi.mocked(comfyTemplates.freeComfyVram).mockResolvedValue({
      ok: true,
      settled: false,
      freed_bytes: 0,
      devices: [DEVICE],
    })

    render(withQuery(<ComfyBridgeSection />))
    fireEvent.click(await screen.findByText('释放显存'))

    expect(await screen.findByText(/仍在卸载/)).toBeInTheDocument()
  })

  it('区分 ComfyUI 自占与整卡已用,不把别人的占用算在 ComfyUI 头上', async () => {
    // 实机误导:整卡 43.8G 里 39.3G 是 nous 的 qwen3_6_35b_a3b_fp8,ComfyUI 只占 0.1G。
    vi.mocked(comfyTemplates.getComfyHealth).mockResolvedValue({
      online: true,
      queue_depth: 0,
      version: '0.31.0',
      base_url: 'http://localhost:8888',
      timeout_s: 14400,
      devices: [{ ...DEVICE, vram_used: 43_800_000_000, comfy_used: 100_000_000 }],
    })

    render(withQuery(<ComfyBridgeSection />))

    // 两个数都要出现,且 ComfyUI 那个是小的那个
    expect(await screen.findByText(/ComfyUI 0\.1GB/)).toBeInTheDocument()
    expect(screen.getByText(/整卡 43\.8GB/)).toBeInTheDocument()
  })

  it('hides devices and release button when offline', async () => {
    vi.mocked(comfyTemplates.getComfyHealth).mockResolvedValue({
      online: false,
      queue_depth: 0,
      version: '',
      base_url: 'http://localhost:8188',
      timeout_s: 120,
      devices: [],
    })

    render(withQuery(<ComfyBridgeSection />))

    await waitFor(() => {
      expect(screen.getByText(/离线/)).toBeTruthy()
    })
    expect(screen.queryByText('释放显存')).toBeFalsy()
  })

  it('shows an inline error when releasing VRAM fails', async () => {
    vi.mocked(comfyTemplates.getComfyHealth).mockResolvedValue({
      online: true,
      queue_depth: 0,
      version: '0.4.12',
      base_url: 'http://localhost:8188',
      timeout_s: 120,
      devices: [DEVICE],
    })
    vi.mocked(comfyTemplates.freeComfyVram).mockRejectedValue(new Error('释放显存失败(HTTP 502)'))

    render(withQuery(<ComfyBridgeSection />))

    const button = await screen.findByText('释放显存')
    fireEvent.click(button)

    await waitFor(() => {
      expect(screen.getByText(/释放显存失败/)).toBeTruthy()
    })
  })
})

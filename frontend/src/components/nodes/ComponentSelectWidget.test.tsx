import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

vi.mock('../../api/client', () => ({ apiFetch: vi.fn() }))
vi.mock('../../api/useLiveChannel', () => ({ useLiveChannel: vi.fn() }))
// ComponentSelectWidget 现在读 useAllComponentStates 标「已加载」—— stub 成空,保留
// useComponents/loadedStateByFile 真实(否则全局 apiFetch mock 把 {components} 当 states 喂崩)。
vi.mock('../../api/components', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../../api/components')>()),
  useAllComponentStates: () => ({ data: [] }),
}))
import { apiFetch } from '../../api/client'
import { ComponentSelectWidget } from './DeclarativeNode'

function wrap(ui: React.ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>)
}

describe('ComponentSelectWidget', () => {
  beforeEach(() => vi.clearAllMocks())
  // 迁移到 NodeSelectPopover 后:选项是 role=option 的按钮,只在打开浮层时渲染;选中回传 abs_path。
  it('lists components for the role by abs_path (popover)', async () => {
    vi.mocked(apiFetch).mockResolvedValue({ components: [
      { filename: 'flux-unet.safetensors', abs_path: '/m/flux-unet.safetensors', size_mb: 18000, quant_type: 'bf16', mtime: 0 },
    ] })
    const onChange = vi.fn()
    wrap(<ComponentSelectWidget value="" onChange={onChange} role="diffusion_models" />)
    // 点触发按钮打开浮层
    fireEvent.click(screen.getByRole('button'))
    // 选项出现(数据 query 异步 → findByRole 重试等待),点选回传 abs_path
    const opt = await screen.findByRole('option', { name: /flux-unet/ })
    fireEvent.click(opt)
    expect(onChange).toHaveBeenCalledWith('/m/flux-unet.safetensors')
  })
})

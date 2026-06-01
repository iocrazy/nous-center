import { describe, it, expect, beforeAll } from 'vitest'
import { render, screen, act } from '@testing-library/react'
import { ReactFlowProvider } from '@xyflow/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { vi } from 'vitest'
vi.mock('../../api/client', () => ({ apiFetch: vi.fn(() => Promise.resolve({ components: [] })) }))
vi.mock('../../api/useLiveChannel', () => ({ useLiveChannel: vi.fn() }))
import DeclarativeNode from './DeclarativeNode'
import { DECLARATIVE_NODES } from '../../models/nodeRegistry'
import { NODE_DEFS } from '../../models/workflow'

// flux2_ksampler 是插件节点;测试里手动登记让 DeclarativeNode 能渲染。
beforeAll(() => {
  DECLARATIVE_NODES['flux2_ksampler'] = {
    type: 'flux2_ksampler', label: 'KSampler', category: 'image',
    badge: 'Sampler', badgeColor: 'var(--accent-2)', widgets: [],
  }
  NODE_DEFS['flux2_ksampler'] = {
    type: 'flux2_ksampler', label: 'KSampler',
    inputs: [{ id: 'model', type: 'any', label: 'MODEL' }, { id: 'conditioning', type: 'any', label: 'CONDITIONING' }],
    outputs: [{ id: 'latent', type: 'any', label: 'LATENT' }],
  }
})

function wrap(ui: React.ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={qc}><ReactFlowProvider>{ui}</ReactFlowProvider></QueryClientProvider>)
}

describe('node denoise progress', () => {
  // 真机用 chrome-devtools 抓到:后端 progress_tracker 发的 detail 是 "dit_denoise N/T"
  // (stage 名前缀,非 "step")。早先 DeclarativeNode 正则写死 /step\s+\d+\/\d+/ → 对不上
  // dit_denoise → 「RUNNING 无运行进度」复发 bug。本测试钉死:dit_denoise 文案能解析出进度。
  it('parses "dit_denoise N/T" detail into a step progress bar', () => {
    wrap(<DeclarativeNode id="ksm" type="flux2_ksampler" data={{}} selected={false} {...({} as any)} />)
    act(() => {
      window.dispatchEvent(new CustomEvent('node-progress', {
        detail: { type: 'node_progress', node_id: 'ksm', detail: 'dit_denoise 7/25', progress: 0.28 },
      }))
    })
    expect(screen.getByText(/7\/25/)).toBeInTheDocument()
    expect(screen.getByText(/28%/)).toBeInTheDocument()
  })

  it('still parses legacy "step N/T" detail', () => {
    wrap(<DeclarativeNode id="ksm2" type="flux2_ksampler" data={{}} selected={false} {...({} as any)} />)
    act(() => {
      window.dispatchEvent(new CustomEvent('node-progress', {
        detail: { type: 'node_progress', node_id: 'ksm2', detail: 'step 3/25', progress: 0.12 },
      }))
    })
    expect(screen.getByText(/3\/25/)).toBeInTheDocument()
  })
})

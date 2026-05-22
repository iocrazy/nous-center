import { describe, it, expect, vi, beforeAll } from 'vitest'
import { render, screen, act } from '@testing-library/react'
import { ReactFlowProvider } from '@xyflow/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
vi.mock('../../api/client', () => ({ apiFetch: vi.fn(() => Promise.resolve({ components: [] })) }))
vi.mock('../../api/useLiveChannel', () => ({ useLiveChannel: vi.fn() }))
import DeclarativeNode from './DeclarativeNode'
import { DECLARATIVE_NODES } from '../../models/nodeRegistry'
import { NODE_DEFS } from '../../models/workflow'

// flux2_vae_decode 是插件节点(运行时 loadPluginDefinitions 注册),测试里手动登记
// 让 DeclarativeNode 能渲染(imageStage cached 提示挂在它上)。
beforeAll(() => {
  DECLARATIVE_NODES['flux2_vae_decode'] = {
    type: 'flux2_vae_decode', label: 'VAE Decode', category: 'image',
    badge: 'Decoder', badgeColor: 'var(--err)', widgets: [],
  }
  NODE_DEFS['flux2_vae_decode'] = {
    type: 'flux2_vae_decode', label: 'VAE Decode',
    inputs: [{ id: 'vae', type: 'any', label: 'VAE' }, { id: 'latent', type: 'any', label: 'LATENT' }],
    outputs: [{ id: 'image', type: 'image', label: '图像' }],
  }
})

function wrap(ui: React.ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={qc}><ReactFlowProvider>{ui}</ReactFlowProvider></QueryClientProvider>)
}

describe('cached hint', () => {
  it('shows (cached) on a cached flux2_vae_decode completion', () => {
    wrap(<DeclarativeNode id="g" type="flux2_vae_decode" data={{}} selected={false} {...({} as any)} />)
    act(() => {
      window.dispatchEvent(new CustomEvent('node-progress', { detail: { type: 'node_complete', node_id: 'g', duration_ms: 50, cached: true } }))
    })
    expect(screen.getByText(/\(cached\)/)).toBeInTheDocument()
  })

  it('does not show (cached) on a non-cached flux2_vae_decode completion', () => {
    wrap(<DeclarativeNode id="h" type="flux2_vae_decode" data={{}} selected={false} {...({} as any)} />)
    act(() => {
      window.dispatchEvent(new CustomEvent('node-progress', { detail: { type: 'node_complete', node_id: 'h', duration_ms: 1200, cached: false } }))
    })
    const doneText = screen.getByText(/完成/)
    expect(doneText.textContent).not.toContain('(cached)')
  })
})

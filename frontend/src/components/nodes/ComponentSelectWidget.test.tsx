import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

vi.mock('../../api/client', () => ({ apiFetch: vi.fn() }))
vi.mock('../../api/useLiveChannel', () => ({ useLiveChannel: vi.fn() }))
import { apiFetch } from '../../api/client'
import { ComponentSelectWidget } from './DeclarativeNode'

function wrap(ui: React.ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>)
}

describe('ComponentSelectWidget', () => {
  beforeEach(() => vi.clearAllMocks())
  it('lists components for the role by abs_path', async () => {
    vi.mocked(apiFetch).mockResolvedValue({ components: [
      { filename: 'flux-unet.safetensors', abs_path: '/m/flux-unet.safetensors', size_mb: 18000, quant_type: 'bf16', mtime: 0 },
    ] })
    wrap(<ComponentSelectWidget value="" onChange={() => {}} role="diffusion_models" />)
    await waitFor(() => expect(screen.getByRole('option', { name: /flux-unet/ })).toBeInTheDocument())
    const opt = screen.getByRole('option', { name: /flux-unet/ }) as HTMLOptionElement
    expect(opt.value).toBe('/m/flux-unet.safetensors')
  })
})

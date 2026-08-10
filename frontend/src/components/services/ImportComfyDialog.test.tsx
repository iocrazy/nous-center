import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import ImportComfyDialog from './ImportComfyDialog'

vi.mock('../../api/comfyTemplates', () => ({
  createComfyTemplate: vi.fn(async () => ({ id: 1, service_name: 'minimax-h3-r2v', node_count: 2 })),
}))

describe('ImportComfyDialog', () => {
  it('拒绝非 API 格式 JSON', async () => {
    render(<ImportComfyDialog open onClose={() => {}} onImported={() => {}} />)
    const file = new File(['{"nodes":[]}'], 'ui.json', { type: 'application/json' })
    fireEvent.change(screen.getByTestId('comfy-file-input'), { target: { files: [file] } })
    await waitFor(() => expect(screen.getByText(/API 格式/)).toBeInTheDocument())
  })
  it('合法 JSON + 名称 → 提交回调服务名', async () => {
    const onImported = vi.fn()
    render(<ImportComfyDialog open onClose={() => {}} onImported={onImported} />)
    const file = new File([JSON.stringify({ '1': { class_type: 'X', inputs: {} } })], 'wf.json',
      { type: 'application/json' })
    fireEvent.change(screen.getByTestId('comfy-file-input'), { target: { files: [file] } })
    fireEvent.change(await screen.findByPlaceholderText(/服务名/), { target: { value: 'minimax-h3-r2v' } })
    fireEvent.click(screen.getByRole('button', { name: /导入/ }))
    await waitFor(() => expect(onImported).toHaveBeenCalledWith('minimax-h3-r2v'))
  })
})

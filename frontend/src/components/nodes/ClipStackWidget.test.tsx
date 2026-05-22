import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

// clip_stack 行内嵌 ComponentSelectWidget(useComponents)+ 状态点(useComponentState)
vi.mock('../../api/components', () => ({
  useComponents: () => ({ data: [
    { filename: 'clipL.safetensors', abs_path: '/m/clipL.safetensors', size_mb: 200, quant_type: 'bf16', mtime: 0 },
    { filename: 't5xxl.safetensors', abs_path: '/m/t5xxl.safetensors', size_mb: 9000, quant_type: 'fp8mixed', mtime: 0 },
  ] }),
  useComponentState: () => ({ state: 'cold' }),
  componentStateKey: (x: { file?: string }) => `${x.file ?? ''}|`,
}))

import { ClipStackWidget } from './DeclarativeNode'

function wrap(ui: React.ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>)
}

describe('ClipStackWidget (PR-3 动态多 CLIP)', () => {
  beforeEach(() => vi.clearAllMocks())

  it('渲染已有 CLIP 条目(每行 file 下拉 + 精度)', () => {
    wrap(<ClipStackWidget value={[{ file: '/m/clipL.safetensors', weight_dtype: 'bfloat16' }]} onChange={() => {}} />)
    const opt = screen.getByRole('option', { name: /clipL/ }) as HTMLOptionElement
    expect(opt.value).toBe('/m/clipL.safetensors')
  })

  it('点「添加 CLIP」追加一行', () => {
    const onChange = vi.fn()
    wrap(<ClipStackWidget value={[{ file: '/m/clipL.safetensors', weight_dtype: 'bfloat16' }]} onChange={onChange} />)
    fireEvent.click(screen.getByText('添加 CLIP'))
    expect(onChange).toHaveBeenCalledTimes(1)
    expect(onChange.mock.calls[0][0]).toHaveLength(2)
  })

  it('点删除移除该行', () => {
    const onChange = vi.fn()
    wrap(<ClipStackWidget
      value={[
        { file: '/m/clipL.safetensors', weight_dtype: 'bfloat16' },
        { file: '/m/t5xxl.safetensors', weight_dtype: 'fp8_e4m3' },
      ]}
      onChange={onChange} />)
    const delButtons = screen.getAllByRole('button', { name: /删除 CLIP/ })
    expect(delButtons).toHaveLength(2)
    fireEvent.click(delButtons[0])
    expect(onChange.mock.calls[0][0]).toEqual([{ file: '/m/t5xxl.safetensors', weight_dtype: 'fp8_e4m3' }])
  })

  it('空值时渲染「添加 CLIP」按钮(0 行)', () => {
    wrap(<ClipStackWidget value={[]} onChange={() => {}} />)
    expect(screen.getByText('添加 CLIP')).toBeInTheDocument()
    expect(screen.queryAllByRole('button', { name: /删除 CLIP/ })).toHaveLength(0)
  })
})

import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import AppEditorNodePopup from './AppEditorNodePopup'
import type { ExposableRow } from './appEditorSchema'
import type { ExposedParam } from '../../api/services'

function row(input_name: string, label: string, over: Partial<ExposedParam> = {}): ExposableRow {
  return {
    input_name,
    label,
    param: { node_id: 'n', key: input_name, input_name, label, type: 'string', default: 'dv', ...over },
  }
}

function setup(exposedParams: ExposedParam[] = []) {
  const handlers = { onToggle: vi.fn(), onRename: vi.fn(), onChangeKind: vi.fn(), onClose: vi.fn() }
  const exposed = new Map(exposedParams.map((p) => [`${p.node_id}::${p.input_name}`, p]))
  render(
    <AppEditorNodePopup
      title="文本编码"
      sub="CLIPTextEncode · #50"
      rows={[row('text', '提示词'), row('strength', '强度', { type: 'number' })]}
      exposed={exposed}
      {...handlers}
    />,
  )
  return handlers
}

describe('AppEditorNodePopup', () => {
  it('renders node title + rows with default value', () => {
    setup()
    expect(screen.getByText('文本编码')).toBeTruthy()
    expect(screen.getByText('提示词')).toBeTruthy()
    expect(screen.getAllByText(/默认值/).length).toBe(2)
  })

  it('toggles exposure when a row is clicked', () => {
    const h = setup()
    fireEvent.click(screen.getByText('提示词'))
    expect(h.onToggle).toHaveBeenCalledWith('text')
  })

  it('name/type controls are disabled until exposed', () => {
    setup() // nothing exposed
    const names = screen.getAllByPlaceholderText('显示名') as HTMLInputElement[]
    expect(names[0].disabled).toBe(true)
  })

  it('renames + changes widget type for an exposed param', () => {
    const exposed: ExposedParam = { node_id: 'n', key: 'text', input_name: 'text', label: '提示词', type: 'string' }
    const h = setup([exposed])
    const name = screen.getAllByPlaceholderText('显示名')[0]
    fireEvent.change(name, { target: { value: '正向提示' } })
    expect(h.onRename).toHaveBeenCalledWith('text', '正向提示')

    // 控件类型 = NodeSelectPopover:type:string 无 single_line → 当前「多行文本」;
    // 点触发打开浮层 → 选「文本」→ onChangeKind('text','text')。
    fireEvent.click(screen.getByText('多行文本'))
    fireEvent.click(screen.getByRole('option', { name: '文本' }))
    expect(h.onChangeKind).toHaveBeenCalledWith('text', 'text')
  })

  it('close button fires onClose', () => {
    const h = setup()
    fireEvent.click(screen.getByTitle('关闭'))
    expect(h.onClose).toHaveBeenCalled()
  })
})

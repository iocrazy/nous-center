import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import WorkflowAppEditor, { type AppEditorValue } from './WorkflowAppEditor'
import type { ExposedParam } from '../../api/services'

function inp(node_id: string, input_name: string, label: string): ExposedParam {
  return { node_id, key: input_name, input_name, label, type: 'string' }
}

function setup(inputs: ExposedParam[]) {
  const onChange = vi.fn()
  const value: AppEditorValue = { inputs, outputs: [] }
  render(
    <WorkflowAppEditor
      nodes={[{ id: 'a', type: 'llm', data: {} }]}
      edges={[]}
      value={value}
      onChange={onChange}
    />,
  )
  return { onChange }
}

describe('WorkflowAppEditor field list (rename / reorder)', () => {
  it('reorders inputs when 下移 is clicked', () => {
    const { onChange } = setup([inp('a', 'f1', '字段一'), inp('a', 'f2', '字段二')])
    fireEvent.click(screen.getAllByTitle('下移')[0])
    expect(onChange).toHaveBeenCalledTimes(1)
    const next = onChange.mock.calls[0][0] as AppEditorValue
    expect(next.inputs.map((p) => p.input_name)).toEqual(['f2', 'f1'])
  })

  it('renames an input label', () => {
    const { onChange } = setup([inp('a', 'f1', '字段一')])
    fireEvent.change(screen.getByPlaceholderText('f1'), { target: { value: '新名' } })
    const next = onChange.mock.calls[0][0] as AppEditorValue
    expect(next.inputs[0].label).toBe('新名')
    expect(next.inputs[0].input_name).toBe('f1') // slot 不变
  })

  it('removes an input', () => {
    const { onChange } = setup([inp('a', 'f1', '字段一'), inp('a', 'f2', '字段二')])
    fireEvent.click(screen.getAllByTitle('移除暴露')[0])
    const next = onChange.mock.calls[0][0] as AppEditorValue
    expect(next.inputs.map((p) => p.input_name)).toEqual(['f2'])
  })

  it('first row 上移 is disabled (no-op)', () => {
    const { onChange } = setup([inp('a', 'f1', '字段一'), inp('a', 'f2', '字段二')])
    const up = screen.getAllByTitle('上移')[0] as HTMLButtonElement
    expect(up.disabled).toBe(true)
  })
})

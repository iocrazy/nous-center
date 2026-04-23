import { describe, it, expect, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import SchemaDrivenForm from './SchemaDrivenForm'
import type { ExposedParam } from '../../api/services'

function setup(inputs: ExposedParam[]) {
  const onSubmit = vi.fn()
  render(<SchemaDrivenForm inputs={inputs} onSubmit={onSubmit} />)
  return { onSubmit }
}

describe('SchemaDrivenForm', () => {
  it('renders empty-state hint when no inputs', () => {
    setup([])
    expect(screen.getByText(/没有暴露入参/)).toBeInTheDocument()
  })

  it('renders a string input and submits its value', () => {
    const { onSubmit } = setup([
      { node_id: 'in_1', key: 'prompt', input_name: 'value', label: '提示词', type: 'string', required: true },
    ])
    const inp = screen.getByRole('textbox')
    fireEvent.change(inp, { target: { value: 'hello' } })
    fireEvent.click(screen.getByText(/▶ 运行/))
    expect(onSubmit).toHaveBeenCalledWith({ prompt: 'hello' })
  })

  it('renders a multiline textarea for type=string_multiline', () => {
    setup([
      {
        node_id: 'in_2',
        key: 'script',
        input_name: 'value',
        label: '脚本',
        type: 'string_multiline',
      },
    ])
    expect(screen.getByRole('textbox').tagName).toBe('TEXTAREA')
  })

  it('renders a select when constraints.enum is present', () => {
    setup([
      {
        node_id: 'in_3',
        key: 'voice',
        input_name: 'value',
        label: '音色',
        type: 'string',
        constraints: { enum: ['alice', 'bob'] },
      },
    ])
    const sel = screen.getByRole('combobox') as HTMLSelectElement
    expect(sel).toBeInTheDocument()
    expect(Array.from(sel.options).map((o) => o.value)).toEqual(['alice', 'bob'])
  })

  it('renders a checkbox for boolean and forwards the typed value on submit', () => {
    const { onSubmit } = setup([
      { node_id: 'in_4', key: 'stream', input_name: 'value', label: '流式', type: 'boolean' },
    ])
    const cb = screen.getByRole('checkbox')
    fireEvent.click(cb)
    fireEvent.click(screen.getByText(/▶ 运行/))
    expect(onSubmit).toHaveBeenCalledWith({ stream: true })
  })

  it('parses number input as a Number on submit', () => {
    const { onSubmit } = setup([
      { node_id: 'in_5', key: 'temperature', input_name: 'value', label: '温度', type: 'number' },
    ])
    fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '0.7' } })
    fireEvent.click(screen.getByText(/▶ 运行/))
    expect(onSubmit).toHaveBeenCalledWith({ temperature: 0.7 })
  })

  it('honors legacy aliases (api_name + param_key) on backfilled rows', () => {
    const { onSubmit } = setup([
      { node_id: 'in_6', api_name: 'prompt', param_key: 'value', label: '提示', type: 'string' },
    ])
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'hi' } })
    fireEvent.click(screen.getByText(/▶ 运行/))
    expect(onSubmit).toHaveBeenCalledWith({ prompt: 'hi' })
  })
})

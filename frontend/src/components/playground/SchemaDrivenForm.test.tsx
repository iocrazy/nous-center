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

  it('renders a slider (range + number) when numeric with min/max', () => {
    const { onSubmit } = setup([
      {
        node_id: 'in_s', key: 'steps', input_name: 'steps', label: '步数',
        type: 'number', constraints: { min: 1, max: 100, step: 1 }, default: 20,
      },
    ])
    const range = screen.getByRole('slider') as HTMLInputElement
    expect(range).toBeInTheDocument()
    expect(range.min).toBe('1')
    expect(range.max).toBe('100')
    fireEvent.change(range, { target: { value: '30' } })
    fireEvent.click(screen.getByText(/▶ 运行/))
    expect(onSubmit).toHaveBeenCalledWith({ steps: 30 })
  })

  it('shows a randomize button for seed-like numeric fields and sets a number', () => {
    const { onSubmit } = setup([
      { node_id: 'in_seed', key: 'seed', input_name: 'seed', label: '种子', type: 'integer' },
    ])
    fireEvent.click(screen.getByLabelText('随机种子'))
    fireEvent.click(screen.getByText(/▶ 运行/))
    expect(onSubmit).toHaveBeenCalledTimes(1)
    const arg = onSubmit.mock.calls[0][0] as Record<string, unknown>
    expect(typeof arg.seed).toBe('number')
  })

  it('uses enum_labels for select option text', () => {
    setup([
      {
        node_id: 'in_e', key: 'mode', input_name: 'mode', label: '模式', type: 'string',
        constraints: { enum: ['a', 'b'], enum_labels: { a: '甲', b: '乙' } },
      },
    ])
    const sel = screen.getByRole('combobox') as HTMLSelectElement
    expect(Array.from(sel.options).map((o) => o.textContent)).toEqual(['甲', '乙'])
    expect(Array.from(sel.options).map((o) => o.value)).toEqual(['a', 'b'])
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

import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import OptionThumbGrid from './OptionThumbGrid'
import { parseValues } from './fieldKind'

const OPTS = [
  { value: 'sai-anime', label: 'SAI-动漫', image: 'https://x/a.jpg' },
  { value: 'Fooocus Enhance', label: 'Fooocus-优化增强', image: 'https://x/b.jpg' },
  { value: 'no-thumb' },
]

describe('parseValues', () => {
  it('空串给空数组(而不是含一个空字符串的数组)', () => {
    expect(parseValues('')).toEqual([])
  })
  it('去空白并丢掉空项', () => {
    expect(parseValues('a, b ,,c')).toEqual(['a', 'b', 'c'])
  })
})

describe('OptionThumbGrid', () => {
  it('渲染缩略图,没有 image 的项退化成文字卡不留空白', () => {
    render(<OptionThumbGrid options={OPTS} value="" onChange={() => {}} />)
    const imgs = screen.getAllByRole('presentation', { hidden: true }) as HTMLImageElement[]
    expect(imgs.map((i) => i.src)).toEqual(['https://x/a.jpg', 'https://x/b.jpg'])
    // 无图项仍然可选,且显示名出现两次(卡面 + 卡底标题)
    expect(screen.getAllByText('no-thumb').length).toBeGreaterThan(0)
  })

  it('单选:点击即选中,再点同一项取消', () => {
    const onChange = vi.fn()
    const { rerender } = render(
      <OptionThumbGrid options={OPTS} value="" onChange={onChange} />)
    fireEvent.click(screen.getByTitle('sai-anime'))
    expect(onChange).toHaveBeenCalledWith('sai-anime')

    rerender(<OptionThumbGrid options={OPTS} value="sai-anime" onChange={onChange} />)
    fireEvent.click(screen.getByTitle('sai-anime'))
    expect(onChange).toHaveBeenLastCalledWith('')
  })

  it('多选:拼成逗号串,且按 options 原序而非点击顺序', () => {
    const onChange = vi.fn()
    // 先点第二项,再(在已选第二项的状态下)点第一项 —— 输出仍要是 options 的顺序
    const { rerender } = render(
      <OptionThumbGrid options={OPTS} value="" multiple onChange={onChange} />)
    fireEvent.click(screen.getByTitle('Fooocus Enhance'))
    expect(onChange).toHaveBeenCalledWith('Fooocus Enhance')

    rerender(
      <OptionThumbGrid options={OPTS} value="Fooocus Enhance" multiple onChange={onChange} />)
    fireEvent.click(screen.getByTitle('sai-anime'))
    expect(onChange).toHaveBeenLastCalledWith('sai-anime,Fooocus Enhance')
  })

  it('多选:再点已选项从串里摘掉', () => {
    const onChange = vi.fn()
    render(
      <OptionThumbGrid
        options={OPTS} value="sai-anime,Fooocus Enhance" multiple onChange={onChange} />)
    fireEvent.click(screen.getByTitle('sai-anime'))
    expect(onChange).toHaveBeenCalledWith('Fooocus Enhance')
  })

  it('搜索按值和显示名两边匹配', () => {
    render(<OptionThumbGrid options={OPTS} value="" onChange={() => {}} />)
    const box = screen.getByLabelText('搜索选项')
    // 中文显示名
    fireEvent.change(box, { target: { value: '动漫' } })
    expect(screen.getByTitle('sai-anime')).toBeTruthy()
    expect(screen.queryByTitle('Fooocus Enhance')).toBeNull()
    // 英文值
    fireEvent.change(box, { target: { value: 'fooocus' } })
    expect(screen.getByTitle('Fooocus Enhance')).toBeTruthy()
    expect(screen.queryByTitle('sai-anime')).toBeNull()
  })

  it('搜索无结果给提示而不是空白', () => {
    render(<OptionThumbGrid options={OPTS} value="" onChange={() => {}} />)
    fireEvent.change(screen.getByLabelText('搜索选项'), { target: { value: 'zzz' } })
    expect(screen.getByText(/没有匹配/)).toBeTruthy()
  })

  it('已选时给出清空按钮', () => {
    const onChange = vi.fn()
    render(<OptionThumbGrid options={OPTS} value="a,b" multiple onChange={onChange} />)
    fireEvent.click(screen.getByTitle('清空选择'))
    expect(onChange).toHaveBeenCalledWith('')
  })
})

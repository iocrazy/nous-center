import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import SchemaDrivenOutput from './SchemaDrivenOutput'
import type { ExposedParam } from '../../api/services'

function out(p: Partial<ExposedParam> & { node_id: string; key: string }): ExposedParam {
  return { ...p }
}

describe('SchemaDrivenOutput', () => {
  it('shows a placeholder when there is no result yet', () => {
    render(<SchemaDrivenOutput outputs={[]} result={null} />)
    expect(screen.getByText(/运行后输出会显示在这里/)).toBeInTheDocument()
  })

  it('renders a string result as text content', () => {
    render(
      <SchemaDrivenOutput
        outputs={[out({ node_id: 'out_1', key: 'echo', label: 'Echo', type: 'string' })]}
        result={{ echo: 'hello world' }}
      />,
    )
    expect(screen.getByText('hello world')).toBeInTheDocument()
  })

  it('renders an audio element for type=audio when value is a URL', () => {
    const { container } = render(
      <SchemaDrivenOutput
        outputs={[out({ node_id: 'out_2', key: 'voice', label: '语音', type: 'audio' })]}
        result={{ voice: 'https://cdn.example/x.wav' }}
      />,
    )
    const audio = container.querySelector('audio')
    expect(audio).not.toBeNull()
    expect(audio?.getAttribute('src')).toBe('https://cdn.example/x.wav')
  })

  it('renders a video element for type=video', () => {
    const { container } = render(
      <SchemaDrivenOutput
        outputs={[out({ node_id: 'out_3', key: 'clip', label: '短视频', type: 'video' })]}
        result={{ clip: 'https://cdn.example/x.mp4' }}
      />,
    )
    expect(container.querySelector('video')).not.toBeNull()
  })

  it('renders an image element for type=image', () => {
    const { container } = render(
      <SchemaDrivenOutput
        outputs={[out({ node_id: 'out_4', key: 'pic', label: '图', type: 'image' })]}
        result={{ pic: '/static/x.png' }}
      />,
    )
    expect(container.querySelector('img')).not.toBeNull()
  })

  it('falls back to JSON dump when there are no declared outputs', () => {
    render(<SchemaDrivenOutput outputs={[]} result={{ raw: { ok: true } }} />)
    expect(screen.getByText(/"raw"/)).toBeInTheDocument()
  })

  it('shows an error box when error prop is set', () => {
    render(<SchemaDrivenOutput outputs={[]} result={null} error="boom" />)
    expect(screen.getByText('boom')).toBeInTheDocument()
  })
})

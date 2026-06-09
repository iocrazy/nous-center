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

  // 真机回归:image 服务输出常是 type=string + slot=image_url(如 img-flux2),
  // 必须按值/slot 推断出 <img>,否则把图渲成 URL 文本。
  it('infers <img> from a .png URL even when type=string (real img-flux2 case)', () => {
    const { container } = render(
      <SchemaDrivenOutput
        outputs={[out({ node_id: 'dec', key: 'output_1', input_name: 'image_url', label: '图像', type: 'string' })]}
        result={{ outputs: { dec: { image_url: '/files/images/2026/abc.png?token=x&expires=1' } } }}
      />,
    )
    const img = container.querySelector('img')
    expect(img).not.toBeNull()
    expect(img?.getAttribute('src')).toContain('/files/images/2026/abc.png')
  })

  it('infers <img> from a data:image URI when type=string', () => {
    const { container } = render(
      <SchemaDrivenOutput
        outputs={[out({ node_id: 'd', key: 'image', input_name: 'image', type: 'string' })]}
        result={{ image: 'data:image/png;base64,iVBORw0KGgo=' }}
      />,
    )
    expect(container.querySelector('img')).not.toBeNull()
  })

  it('does NOT turn a plain text field into an image', () => {
    const { container } = render(
      <SchemaDrivenOutput
        outputs={[out({ node_id: 't', key: 'caption', input_name: 'text', type: 'string' })]}
        result={{ caption: 'a description of the image' }}
      />,
    )
    expect(container.querySelector('img')).toBeNull()
    expect(screen.getByText('a description of the image')).toBeInTheDocument()
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

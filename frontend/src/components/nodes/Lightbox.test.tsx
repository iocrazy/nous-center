import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, act } from '@testing-library/react'
import Lightbox from './Lightbox'
import { useLightboxStore } from '../../stores/lightbox'

const URL1 = 'data:image/png;base64,iVBORw0KGgo='

function scaleOf(transform: string): number {
  const m = transform.match(/scale\(([\d.]+)\)/)
  return m ? parseFloat(m[1]) : 1
}

describe('Lightbox zoom/pan wiring', () => {
  beforeEach(() => {
    act(() => useLightboxStore.setState({ open: false, images: [], metas: [], index: 0 }))
  })

  it('renders meta panel (prompt + fields + rerun) when meta present', () => {
    const onRerun = vi.fn()
    render(<Lightbox />)
    act(() => useLightboxStore.setState({
      open: true, images: [URL1], index: 0,
      metas: [{ prompt: 'a corgi astronaut', fields: [{ label: 'seed', value: '42' }], durationMs: 2000, onRerun }],
    }))
    expect(screen.getByText('a corgi astronaut')).toBeTruthy()
    expect(screen.getByText('seed')).toBeTruthy()
    expect(screen.getByText(/耗时 2\.0s/)).toBeTruthy()
    fireEvent.click(screen.getByText(/重跑/))
    expect(onRerun).toHaveBeenCalled()
  })

  it('no meta panel when meta empty', () => {
    render(<Lightbox />)
    act(() => useLightboxStore.setState({ open: true, images: [URL1], metas: [undefined], index: 0 }))
    expect(screen.queryByText(/重跑/)).toBeNull()
  })

  it('does not render when closed', () => {
    render(<Lightbox />)
    expect(screen.queryByAltText('preview')).toBeNull()
  })

  it('wheel up zooms in; double-click resets', () => {
    render(<Lightbox />)
    act(() => useLightboxStore.setState({ open: true, images: [URL1], index: 0 }))
    const img = screen.getByAltText('preview') as HTMLImageElement
    expect(scaleOf(img.style.transform)).toBeCloseTo(1)

    // 滚轮向上 → 放大
    fireEvent.wheel(img.parentElement!, { deltaY: -300, clientX: 700, clientY: 450 })
    expect(scaleOf(img.style.transform)).toBeGreaterThan(1)

    // 双击 → 复位
    fireEvent.dblClick(img)
    expect(scaleOf(img.style.transform)).toBeCloseTo(1)
  })

  it('wheel down past 1 stays clamped at 1', () => {
    render(<Lightbox />)
    act(() => useLightboxStore.setState({ open: true, images: [URL1], index: 0 }))
    const img = screen.getByAltText('preview') as HTMLImageElement
    fireEvent.wheel(img.parentElement!, { deltaY: 300, clientX: 700, clientY: 450 })
    expect(scaleOf(img.style.transform)).toBeCloseTo(1)
  })
})

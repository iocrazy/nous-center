import { describe, it, expect, beforeEach } from 'vitest'
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
    act(() => useLightboxStore.setState({ open: false, images: [], index: 0 }))
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

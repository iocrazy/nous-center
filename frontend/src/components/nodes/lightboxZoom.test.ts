import { describe, it, expect } from 'vitest'
import { clampScale, zoomAt, clampPan, IDENTITY, MIN_SCALE, MAX_SCALE } from './lightboxZoom'

describe('lightboxZoom.clampScale', () => {
  it('clamps to [1,6]', () => {
    expect(clampScale(0.2)).toBe(MIN_SCALE)
    expect(clampScale(99)).toBe(MAX_SCALE)
    expect(clampScale(2.5)).toBe(2.5)
  })
})

describe('lightboxZoom.zoomAt', () => {
  it('zooms in and out, returning identity at min scale', () => {
    const zin = zoomAt(IDENTITY, 1.5, 0, 0)
    expect(zin.scale).toBeCloseTo(1.5)
    // zooming back below 1 → identity (reset)
    expect(zoomAt({ scale: 1.1, tx: 20, ty: 5 }, 0.5, 0, 0)).toEqual(IDENTITY)
  })
  it('keeps the cursor-anchored point fixed', () => {
    // point under cursor (cx,cy) in screen space = (cx - tx)/scale; must be invariant.
    const before = { scale: 1, tx: 0, ty: 0 }
    const cx = 100, cy = -40
    const after = zoomAt(before, 2, cx, cy)
    const pBefore = { x: (cx - before.tx) / before.scale, y: (cy - before.ty) / before.scale }
    const pAfter = { x: (cx - after.tx) / after.scale, y: (cy - after.ty) / after.scale }
    expect(pAfter.x).toBeCloseTo(pBefore.x)
    expect(pAfter.y).toBeCloseTo(pBefore.y)
  })
})

describe('lightboxZoom.clampPan', () => {
  it('no pan at scale 1', () => {
    expect(clampPan({ scale: 1, tx: 50, ty: 50 }, 400, 300)).toEqual({ scale: 1, tx: 0, ty: 0 })
  })
  it('clamps within half the overflow', () => {
    // scale 2, w=400 → maxX = 400*(2-1)/2 = 200
    const r = clampPan({ scale: 2, tx: 999, ty: -999 }, 400, 300)
    expect(r.tx).toBe(200)
    expect(r.ty).toBe(-150)
  })
})

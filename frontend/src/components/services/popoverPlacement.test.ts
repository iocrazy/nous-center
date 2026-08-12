import { describe, it, expect } from 'vitest'
import { placePopover, type RectLike } from './popoverPlacement'

function rect(left: number, top: number, right: number, bottom: number): RectLike {
  return { left, top, right, bottom }
}

const CONTAINER = rect(0, 0, 900, 600)

describe('placePopover', () => {
  it('默认贴节点右侧 12px,右侧放得下就不翻', () => {
    const nodeRect = rect(100, 50, 290, 100) // width 190, well within container
    const pos = placePopover({
      nodeRect, containerRect: CONTAINER, scrollLeft: 0, scrollTop: 0,
      clientW: 900, clientH: 600, popW: 380, popH: 400,
    })
    expect(pos.left).toBe(290 + 12) // nodeRect.right + gap
    expect(pos.top).toBe(50) // aligns with node top
  })

  it('右侧放不下 → 翻到节点左侧', () => {
    // 节点贴容器右边界:right 侧放 380 宽弹窗会超出容器
    const nodeRect = rect(700, 50, 890, 100)
    const pos = placePopover({
      nodeRect, containerRect: CONTAINER, scrollLeft: 0, scrollTop: 0,
      clientW: 900, clientH: 600, popW: 380, popH: 400,
    })
    expect(pos.left).toBe(700 - 380 - 12) // nodeRect.left - popW - gap
  })

  it('翻到左侧后仍溢出容器左边界 → 夹在容器左边界 + 8px', () => {
    const nodeRect = rect(50, 50, 240, 100) // 靠近左边界,左翻会变负数
    const pos = placePopover({
      nodeRect, containerRect: CONTAINER, scrollLeft: 0, scrollTop: 0,
      clientW: 300, clientH: 600, popW: 380, popH: 400, // 窄容器强制翻转
    })
    expect(pos.left).toBe(8)
  })

  it('顶部对齐会导致底部溢出 → 上移,最多移到 maxTop', () => {
    const nodeRect = rect(100, 550, 290, 590) // 节点在容器底部附近
    const pos = placePopover({
      nodeRect, containerRect: CONTAINER, scrollLeft: 0, scrollTop: 0,
      clientW: 900, clientH: 600, popW: 380, popH: 400,
    })
    // maxTop = 0 + 600 - 400 - 8 = 192
    expect(pos.top).toBe(192)
  })

  it('容器有滚动偏移时,内容坐标系下的定位结果与未滚动时一致(popover 是滚动内容的子元素)', () => {
    // 与 case 1 是同一个节点(内容坐标 left:100..290, top:50..100),但容器向右/下滚了
    // (50,30)——同一节点的屏幕矩形(getBoundingClientRect)会相应左移/上移 50/30。
    const nodeRect = rect(100 - 50, 50 - 30, 290 - 50, 100 - 30)
    const pos = placePopover({
      nodeRect, containerRect: CONTAINER, scrollLeft: 50, scrollTop: 30,
      clientW: 900, clientH: 600, popW: 380, popH: 400,
    })
    expect(pos.left).toBe(290 + 12) // 内容坐标系下与未滚动时(case 1)相同
    expect(pos.top).toBe(50)
  })
})

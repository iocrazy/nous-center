// 节点配置弹窗定位算法(Task 9 UI 返工,对齐 Infinite-Canvas 参考实现
// static/js/comfyui-settings.js 的 openNodePopup,约 L707-728)。纯函数,不碰 DOM —— 调用方
// (ComfyTemplateEditor)在点击节点时读一次 getBoundingClientRect()/scrollLeft/scrollTop
// 传进来,这里只做算术,方便单测三种情形(贴右/翻左/顶部溢出夹底部)不用真的渲染 DOM。
//
// 坐标系:所有输出(left/top)相对图容器可滚动内容原点(scrollLeft=0, scrollTop=0 处),
// 与 IC 的 `wrap.scrollLeft`/`wrap.scrollTop` 换算方式一致 —— 这样弹窗定位在容器发生原生
// 滚动时也跟着内容走,不会跟丢节点(我们的 React Flow 容器目前不原生滚动,scrollLeft/Top
// 恒为 0,但公式保留这两个参数,不写死 0,保持可测 & 未来容器改滚动语义时不用重写这层算法)。

export interface RectLike {
  left: number
  top: number
  right: number
  bottom: number
}

export interface PlacePopoverArgs {
  /** 被点击节点的屏幕矩形(getBoundingClientRect()) */
  nodeRect: RectLike
  /** 图容器的屏幕矩形 */
  containerRect: RectLike
  scrollLeft: number
  scrollTop: number
  /** 容器可视区宽/高(clientWidth/clientHeight,不含滚动条外的溢出部分) */
  clientW: number
  clientH: number
  /** 弹窗固定宽度 */
  popW: number
  /** 弹窗最大高度(调用方按 min(容器高-40, 视口高*0.7) 算好传入) */
  popH: number
}

export interface PopoverPosition {
  left: number
  top: number
}

const EDGE_GAP = 12
const CLAMP_MARGIN = 8

/** 默认贴节点右侧 12px;放不下(超出容器右边界)翻到左侧;再放不下夹在容器左边界。
 *  纵向:默认与节点顶部对齐;超出容器底部就上移,最多移到 `maxTop`。 */
export function placePopover({
  nodeRect, containerRect, scrollLeft, scrollTop, clientW, clientH, popW, popH,
}: PlacePopoverArgs): PopoverPosition {
  let left = nodeRect.right - containerRect.left + scrollLeft + EDGE_GAP
  if (left + popW > scrollLeft + clientW - CLAMP_MARGIN) {
    left = nodeRect.left - containerRect.left + scrollLeft - popW - EDGE_GAP
  }
  if (left < scrollLeft + CLAMP_MARGIN) left = scrollLeft + CLAMP_MARGIN

  let top = nodeRect.top - containerRect.top + scrollTop
  const maxTop = scrollTop + clientH - popH - CLAMP_MARGIN
  if (top > maxTop) top = Math.max(scrollTop + CLAMP_MARGIN, maxTop)

  return { left, top }
}

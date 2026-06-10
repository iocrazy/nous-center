// 灯箱缩放/平移纯数学(移植 Infinite-Canvas image-preview.js)。抽成纯函数便于单测。
// 变换为 `translate(tx,ty) scale(scale)`,transform-origin = 元素中心;tx/ty/cx/cy 均为
// 相对中心的屏幕像素。
export const MIN_SCALE = 1
export const MAX_SCALE = 6

export interface ZoomState {
  scale: number
  tx: number
  ty: number
}

export const IDENTITY: ZoomState = { scale: 1, tx: 0, ty: 0 }

export function clampScale(s: number): number {
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, s))
}

/** 以光标点(cx,cy,相对中心)为锚缩放 factor 倍;回到 1 时归位。 */
export function zoomAt(st: ZoomState, factor: number, cx: number, cy: number): ZoomState {
  const ns = clampScale(st.scale * factor)
  if (ns === MIN_SCALE) return { ...IDENTITY }
  const k = ns / st.scale
  return { scale: ns, tx: cx - (cx - st.tx) * k, ty: cy - (cy - st.ty) * k }
}

/** 把平移夹到能看到放大后图的边缘、又不让图飞出(w/h = scale=1 时图的渲染尺寸)。 */
export function clampPan(st: ZoomState, w: number, h: number): ZoomState {
  const maxX = Math.max(0, (w * (st.scale - 1)) / 2)
  const maxY = Math.max(0, (h * (st.scale - 1)) / 2)
  return {
    scale: st.scale,
    tx: Math.min(maxX, Math.max(-maxX, st.tx)),
    ty: Math.min(maxY, Math.max(-maxY, st.ty)),
  }
}

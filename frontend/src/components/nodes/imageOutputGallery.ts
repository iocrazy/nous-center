// image_output 累积画廊的纯数据逻辑(从组件文件拆出 —— react-refresh 要求组件文件只导出组件)。

/** 一张累积出图(对齐 Infinite-Canvas OUTPUT:节点是「你这条流所有生成的画廊」)。 */
export interface OutImg {
  url: string
  seed?: number | null
  steps?: number | null
  cfg?: number | null
  width?: number | null
  height?: number | null
  durationMs?: number | null
}

export const MAX_IMAGES = 60  // 防止跑太多次撑爆节点 data;超出丢最旧。

/** 累积一张出图(dedup by url + 截断最旧)。同 url → 返回原数组引用(调用方据此跳过写回)。 */
export function appendOutput(prev: OutImg[], item: OutImg, max = MAX_IMAGES): OutImg[] {
  if (prev.some((x) => x.url === item.url)) return prev
  return [...prev, item].slice(-max)
}

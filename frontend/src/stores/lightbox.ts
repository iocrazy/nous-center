import { create } from 'zustand'
import { useWorkspaceStore } from './workspace'

// 全局图片预览 lightbox(对齐 Infinite-Canvas Z/方向键切图)。
// 画布上任意图像节点点图 → 收集当前工作流全部图 url,定位到点击的那张,
// ←/→ 跨图切换,Esc/点击关闭。

// 节点 data 里可能承载图片 url 的字段(ImageOutput / ImageCompare / 输入图等)。
const IMAGE_KEYS = ['image_url', 'image_a_url', 'image_b_url', 'image', 'dataUrl', 'url']

/** 按节点顺序收集当前工作流全部图片 url(去重)。 */
export function collectWorkflowImages(): string[] {
  const wf = useWorkspaceStore.getState().getActiveWorkflow()
  const urls: string[] = []
  const push = (v: unknown) => { if (typeof v === 'string' && v && !urls.includes(v)) urls.push(v) }
  for (const n of wf.nodes) {
    const d = n.data as Record<string, unknown> | undefined
    if (!d) continue
    for (const k of IMAGE_KEYS) push(d[k])
    // 多模态输入:data.images 是 url 数组,逐个收。
    if (Array.isArray(d.images)) d.images.forEach(push)
  }
  return urls
}

interface LightboxState {
  open: boolean
  images: string[]
  index: number
  /** 从某张图 url 打开:自动收集同工作流全部图并定位。 */
  openFromUrl: (url: string) => void
  /** 显式给定图集打开。 */
  openAt: (images: string[], index: number) => void
  close: () => void
  next: () => void
  prev: () => void
}

export const useLightboxStore = create<LightboxState>((set, get) => ({
  open: false,
  images: [],
  index: 0,
  openFromUrl: (url) => {
    const all = collectWorkflowImages()
    let images = all
    let index = all.indexOf(url)
    if (index < 0) { images = [url]; index = 0 }
    set({ open: true, images, index })
  },
  openAt: (images, index) => set({ open: true, images, index: Math.max(0, Math.min(index, images.length - 1)) }),
  close: () => set({ open: false }),
  next: () => { const { images, index } = get(); if (images.length) set({ index: (index + 1) % images.length }) },
  prev: () => { const { images, index } = get(); if (images.length) set({ index: (index - 1 + images.length) % images.length }) },
}))

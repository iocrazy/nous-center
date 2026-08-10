import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, fireEvent } from '@testing-library/react'
import { useLightboxStore } from '../../stores/lightbox'

// mock 数据 hooks + 路由
const task = {
  id: 1,
  output_thumbnails: ['http://x/a.png'],
  input_json: { 提示词: 'a corgi astronaut riding a rocket through space', 宽度: 1024 },
  workflow_id: 5,
  workflow_name: 'wf',
  created_at: new Date('2026-06-10T00:00:00Z').toISOString(),
  duration_ms: 2000,
}
// 视频产物 task(Task 11):result 走 comfy bridge 节点 envelope 形状
// `{outputs:{node_id:{video_url, items:[{kind:'video',url}], thumbnails:[...]}}}`,
// 顶层 output_thumbnails 是 execution_task_serialize._image_urls 落的 ffmpeg 首帧 PNG。
const videoTask = {
  id: 2,
  output_thumbnails: ['http://x/frame.png'],
  input_json: { 提示词: 'a corgi running on the beach at sunset' },
  workflow_id: 5,
  workflow_name: 'wf-video',
  created_at: new Date('2026-06-10T00:00:00Z').toISOString(),
  duration_ms: 4000,
  result: {
    outputs: {
      n1: {
        video_url: 'http://x/clip.mp4',
        thumbnails: ['http://x/frame.png'],
        items: [{ url: 'http://x/clip.mp4', kind: 'video', filename: 'clip.mp4' }],
      },
    },
  },
}
vi.mock('../../api/tasks', () => ({
  useImageTasks: () => ({ data: [task, videoTask], isLoading: false }),
  useDeleteTask: () => ({ mutate: vi.fn() }),
}))
vi.mock('../../api/services', () => ({
  useServices: () => ({ data: [{ id: 9, workflow_id: 5, name: 'svc' }] }),
}))
vi.mock('react-router-dom', () => ({ useNavigate: () => vi.fn() }))
// panel store 模块加载时读 localStorage(Node24 测试环境 localStorage 异常)→ mock 绕开。
vi.mock('../../stores/panel', () => ({ usePanelStore: (sel: (s: { setOverlay: () => void }) => unknown) => sel({ setOverlay: () => {} }) }))

import HistoryOverlay from './HistoryOverlay'

describe('HistoryOverlay → 共享灯箱', () => {
  beforeEach(() => { useLightboxStore.setState({ open: false, images: [], metas: [], kinds: [], index: 0 }) })

  it('点卡片用 openItems 打开共享灯箱,meta 带 prompt + 重跑', () => {
    const { container } = render(<HistoryOverlay />)
    const img = container.querySelector('img')!
    fireEvent.click(img.parentElement!)
    const st = useLightboxStore.getState()
    expect(st.open).toBe(true)
    // 画廊是跨 task 的扁平图集(见组件注释),第二个 task 是视频 → 追加视频条目。
    expect(st.images).toEqual(['http://x/a.png', 'http://x/clip.mp4'])
    expect(st.metas[0]?.prompt).toContain('corgi astronaut')
    expect(typeof st.metas[0]?.onRerun).toBe('function') // 有对应服务 → 可重跑
    expect(st.metas[0]?.fields?.some((f) => f.label === '宽度')).toBe(true)
  })

  it('视频任务卡片渲染 ▶ 角标 + 工作流名标签,点开传给灯箱的是视频 url + kind video', () => {
    const { container, getByLabelText } = render(<HistoryOverlay />)
    // ▶ 角标(aria-label="视频")存在
    expect(getByLabelText('视频')).toBeTruthy()
    // 服务/工作流名标签
    expect(container.textContent).toContain('wf-video')

    const imgs = container.querySelectorAll('img')
    expect(imgs.length).toBe(2)
    const videoCardThumb = imgs[1] // 第二张卡片 = videoTask,缩略图仍是 PNG 首帧
    expect(videoCardThumb.getAttribute('src')).toBe('http://x/frame.png')

    fireEvent.click(videoCardThumb.parentElement!)
    const st = useLightboxStore.getState()
    expect(st.open).toBe(true)
    // 灯箱条目是视频 url(不是缩略 PNG),kind 标记为 video
    expect(st.images).toContain('http://x/clip.mp4')
    const idx = st.images.indexOf('http://x/clip.mp4')
    expect(st.kinds[idx]).toBe('video')
  })
})

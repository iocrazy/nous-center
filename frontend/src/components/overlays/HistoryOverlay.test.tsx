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
vi.mock('../../api/tasks', () => ({
  useImageTasks: () => ({ data: [task], isLoading: false }),
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
  beforeEach(() => { useLightboxStore.setState({ open: false, images: [], metas: [], index: 0 }) })

  it('点卡片用 openItems 打开共享灯箱,meta 带 prompt + 重跑', () => {
    const { container } = render(<HistoryOverlay />)
    const img = container.querySelector('img')!
    fireEvent.click(img.parentElement!)
    const st = useLightboxStore.getState()
    expect(st.open).toBe(true)
    expect(st.images).toEqual(['http://x/a.png'])
    expect(st.metas[0]?.prompt).toContain('corgi astronaut')
    expect(typeof st.metas[0]?.onRerun).toBe('function') // 有对应服务 → 可重跑
    expect(st.metas[0]?.fields?.some((f) => f.label === '宽度')).toBe(true)
  })
})

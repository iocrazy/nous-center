import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import NodeEditor from './NodeEditor'
import { usePanelStore } from '../../stores/panel'

// ServicesList 的 chunk 永远不 resolve —— 模拟「刷新 /services,chunk 还在路上」
// 这个窗口。overlay 挂起期间 Suspense fallback 一直显示,断言随便做。
vi.mock('../../pages/ServicesList', () => new Promise<never>(() => {}))

describe('NodeEditor overlay 懒加载占位', () => {
  beforeEach(() => {
    usePanelStore.setState({ activeOverlay: null, activePanel: null })
  })

  it('chunk 未到达时用不透明满屏占位盖住画布,而不是露出画布', async () => {
    usePanelStore.setState({ activeOverlay: 'services', activePanel: null })
    render(<NodeEditor />)

    const placeholder = await screen.findByTestId('overlay-loading')

    // 满屏 + 不透明 + 主题底色 —— 这三条合起来才等于「画布看不见」。
    expect(placeholder).toHaveClass('absolute', 'inset-0')
    expect(placeholder.style.background).toBe('var(--bg)')
    expect(placeholder.style.opacity).toBe('') // 没有透明度,不会透出下面

    // 画布仍挂载(切回来要保留视口),但占位层在它之后 → 盖在它上面。
    const canvas = screen.getByTestId('workflow-canvas')
    expect(
      canvas.compareDocumentPosition(placeholder) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
  })

  it('「加载中…」延迟淡入,秒开不闪 loading 框', async () => {
    usePanelStore.setState({ activeOverlay: 'services', activePanel: null })
    render(<NodeEditor />)

    const text = await screen.findByText('加载中…')
    // fill-mode both + 180ms delay:延迟期间停在 opacity:0,快的加载压根不露脸。
    expect(text.style.animation).toContain('nous-delayed-fade-in')
    expect(text.style.animation).toContain('180ms')
    expect(text.style.animation).toContain('both')
  })

  it('画布路由(activeOverlay=null)不挂占位层,画布直接渲染', () => {
    render(<NodeEditor />)

    expect(screen.queryByTestId('overlay-loading')).toBeNull()
    expect(screen.getByTestId('workflow-canvas')).toBeInTheDocument()
  })
})

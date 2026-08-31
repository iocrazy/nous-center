// 回归护栏:节点卡必须带 source/target <Handle>。
// 这个组件在 ComfyTemplateEditor.test.tsx 里是被 vi.mock 掉的,所以之前没有任何测试
// 覆盖它 —— 结果是「连线看不见」被反复报了三次、前两次都当成配色问题去调 stroke,
// 而真因是卡片里根本没有 <Handle>:xyflow 的边是 source handle → target handle 连的,
// 找不到 handle 就在渲染前整条丢掉(控制台 error#008),DOM 里连
// .react-flow__edge-path 都不会出现,再怎么改颜色都没用。
// jsdom 里没有真实布局,xyflow 量不出节点尺寸、边本身画不出来,所以这里不断言边,
// 只钉住「handle 在」这个前提条件 —— 边的真实可见性由 Playwright 截图验证。
import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ComfyTemplateGraph from './ComfyTemplateGraph'

const NODES = [
  { id: '1', class_type: 'CheckpointLoaderSimple', x: 0, y: 0, usedCount: 0 },
  { id: '2', class_type: 'KSampler', x: 250, y: 0, usedCount: 1 },
]
const EDGES = [{ source: '1', target: '2' }]

describe('ComfyTemplateGraph', () => {
  it('每个节点卡都渲染 source + target handle(没有 handle 则所有边会被 xyflow 丢弃)', () => {
    const { container } = render(
      <ComfyTemplateGraph nodes={NODES} edges={EDGES} activeNodeId={null} onNodeClick={() => {}} />,
    )
    const cards = container.querySelectorAll('.react-flow__node')
    expect(cards.length).toBe(NODES.length)
    cards.forEach((card) => {
      expect(card.querySelector('.react-flow__handle.source')).not.toBeNull()
      expect(card.querySelector('.react-flow__handle.target')).not.toBeNull()
    })
  })
})

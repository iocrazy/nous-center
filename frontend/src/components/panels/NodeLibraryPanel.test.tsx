import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import NodeLibraryPanel from './NodeLibraryPanel'

describe('NodeLibraryPanel', () => {
  it('收敛后:图像组保留,Family B「组件加载」分类已删', () => {
    render(<NodeLibraryPanel />)
    expect(screen.getByText('图像')).toBeInTheDocument()
    // Family B 收敛删除:不再有「组件加载」分类 + UNET 加载等节点
    expect(screen.queryByText('组件加载')).not.toBeInTheDocument()
    expect(screen.queryByText('UNET 加载')).not.toBeInTheDocument()
    expect(screen.queryByText('图像生成')).not.toBeInTheDocument()
  })
})

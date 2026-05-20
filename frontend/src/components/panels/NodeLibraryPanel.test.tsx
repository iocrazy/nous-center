import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import NodeLibraryPanel from './NodeLibraryPanel'

describe('NodeLibraryPanel', () => {
  it('shows the 组件加载 category with loader nodes', () => {
    render(<NodeLibraryPanel />)
    expect(screen.getByText('组件加载')).toBeInTheDocument()
    expect(screen.getByText('UNET 加载')).toBeInTheDocument()
  })
})

import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import NodePropertyPanel from './NodePropertyPanel'
import { useSelectionStore } from '../../stores/selection'
import { useWorkspaceStore } from '../../stores/workspace'

// 没选节点 → 空态；选了节点 → 渲染 widgets；改值 → 走 updateNode。

vi.mock('../../api/agents', () => ({
  useAgents: () => ({ data: [] }),
}))

function withQuery(ui: React.ReactNode) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return <QueryClientProvider client={qc}>{ui}</QueryClientProvider>
}

beforeEach(() => {
  // reset stores between tests
  useSelectionStore.setState({ selectedNodeId: null })
  useWorkspaceStore.setState({
    tabs: [
      {
        id: 't1',
        name: 'test',
        workflow: {
          id: 'wf1',
          name: 'test',
          nodes: [
            {
              id: 'llm-1',
              type: 'llm',
              position: { x: 0, y: 0 },
              data: { system: 'hello', temperature: 0.5 },
            },
          ],
          edges: [],
        },
        isDirty: false,
        dbId: null,
        past: [],
        future: [],
      },
    ],
    activeTabId: 't1',
  })
})

describe('NodePropertyPanel', () => {
  it('shows empty state when nothing selected', () => {
    render(withQuery(<NodePropertyPanel />))
    expect(screen.getByText(/点击画布上的一个节点/)).toBeTruthy()
  })

  it('renders widgets for the selected LLM node', () => {
    useSelectionStore.setState({ selectedNodeId: 'llm-1' })
    render(withQuery(<NodePropertyPanel />))
    // header label
    expect(screen.getByText('LLM')).toBeTruthy()
    // 至少 system_prompt + temperature 两个字段渲染出来（label）
    expect(screen.getByText('系统提示')).toBeTruthy()
    expect(screen.getByText('温度')).toBeTruthy()
  })

  it('updates workspace store when a field changes', () => {
    useSelectionStore.setState({ selectedNodeId: 'llm-1' })
    render(withQuery(<NodePropertyPanel />))
    // 系统提示是 textarea，找它然后改值
    const textarea = document.querySelector('textarea') as HTMLTextAreaElement
    expect(textarea).toBeTruthy()
    fireEvent.change(textarea, { target: { value: 'updated prompt' } })
    const node = useWorkspaceStore.getState().tabs[0].workflow.nodes[0]
    expect(node.data.system).toBe('updated prompt')
  })
})

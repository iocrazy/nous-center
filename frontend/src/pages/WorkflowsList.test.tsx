import { describe, it, expect, vi, beforeEach } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import WorkflowsList from './WorkflowsList'
import type { WorkflowSummary } from '../api/workflows'
import type { ServiceRow } from '../api/services'

const navigateSpy = vi.fn()
const useWorkflowsMock = vi.fn()
const useServicesMock = vi.fn()
const createMutateMock = vi.fn()

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom')
  return { ...actual, useNavigate: () => navigateSpy }
})

vi.mock('../api/workflows', async () => {
  const actual = await vi.importActual<typeof import('../api/workflows')>('../api/workflows')
  return {
    ...actual,
    useWorkflows: () => useWorkflowsMock(),
    useCreateWorkflow: () => ({
      mutate: createMutateMock,
      isPending: false,
    }),
    // 卡片菜单的删除入口
    useDeleteWorkflow: () => ({ mutate: vi.fn(), isPending: false }),
  }
})

vi.mock('../api/services', async () => {
  const actual = await vi.importActual<typeof import('../api/services')>('../api/services')
  return {
    ...actual,
    useServices: () => useServicesMock(),
  }
})

function makeWf(over: Partial<WorkflowSummary> = {}): WorkflowSummary {
  return {
    id: 'wf-1',
    name: '基础合成',
    description: null,
    is_template: false,
    status: 'draft',
    auto_generated: false,
    generated_for_service_id: null,
    nodes: [{ id: 'n1' }, { id: 'n2' }, { id: 'n3' }],
    edges: [],
    created_at: '2026-04-23T00:00:00Z',
    updated_at: '2026-04-23T00:00:00Z',
    ...over,
  }
}

function makeSvc(over: Partial<ServiceRow> = {}): ServiceRow {
  return {
    id: 'svc-1',
    name: 'podcast-tts',
    type: 'inference',
    status: 'active',
    source_type: 'workflow',
    source_id: 'wf-1',
    source_name: null,
    category: 'tts',
    meter_dim: 'chars',
    workflow_id: 'wf-1',
    workflow_name: 'demo',
    snapshot_hash: 'sha256:x',
    snapshot_schema_version: 1,
    version: 1,
    created_at: '2026-04-23T00:00:00Z',
    updated_at: '2026-04-23T00:00:00Z',
    ...over,
  }
}

beforeEach(() => {
  navigateSpy.mockReset()
  createMutateMock.mockReset()
  useServicesMock.mockReturnValue({ data: [] })
})

describe('WorkflowsList', () => {
  it('clicking a card navigates to the canvas /workflows/:id', () => {
    useWorkflowsMock.mockReturnValue({
      data: [makeWf({ id: 'wf-99', name: 'click-me' })],
      isLoading: false,
      error: null,
    })
    render(
      <MemoryRouter>
        <WorkflowsList />
      </MemoryRouter>,
    )
    fireEvent.click(screen.getByText('click-me'))
    expect(navigateSpy).toHaveBeenCalledWith('/workflows/wf-99')
  })

  it('shows the linked service + bump-version button when an associated service exists', () => {
    useWorkflowsMock.mockReturnValue({
      data: [makeWf({ id: 'wf-1', name: 'pod', status: 'published' })],
      isLoading: false,
      error: null,
    })
    useServicesMock.mockReturnValue({
      data: [makeSvc({ id: 'svc-1', workflow_id: 'wf-1', version: 1 })],
    })
    render(
      <MemoryRouter>
        <WorkflowsList />
      </MemoryRouter>,
    )
    expect(screen.getByText('podcast-tts')).toBeInTheDocument()
    expect(screen.getByText(/v1/)).toBeInTheDocument()
    expect(screen.getByText(/↻ v2/)).toBeInTheDocument()
  })

  it('shows "未关联服务" + 发布 CTA when no service is linked', () => {
    useWorkflowsMock.mockReturnValue({
      data: [makeWf({ id: 'wf-2', name: 'draft', status: 'draft' })],
      isLoading: false,
      error: null,
    })
    render(
      <MemoryRouter>
        <WorkflowsList />
      </MemoryRouter>,
    )
    expect(screen.getByText(/未关联服务/)).toBeInTheDocument()
    expect(screen.getByText('发布为服务')).toBeInTheDocument()
  })

  it('已发布 workflow 但未挂服务时显示"已发布 · 未关联服务"', () => {
    useWorkflowsMock.mockReturnValue({
      data: [makeWf({ id: 'wf-3', name: 'pub-no-svc', status: 'published' })],
      isLoading: false,
      error: null,
    })
    render(
      <MemoryRouter>
        <WorkflowsList />
      </MemoryRouter>,
    )
    expect(screen.getByText(/已发布 · 未关联服务/)).toBeInTheDocument()
  })

  it('点击"发布为服务"按钮 navigate to /workflows/:id?publish=1（不冒泡到 onOpen）', () => {
    useWorkflowsMock.mockReturnValue({
      data: [makeWf({ id: 'wf-2', name: 'draft', status: 'draft' })],
      isLoading: false,
      error: null,
    })
    render(
      <MemoryRouter>
        <WorkflowsList />
      </MemoryRouter>,
    )
    fireEvent.click(screen.getByText('发布为服务'))
    expect(navigateSpy).toHaveBeenLastCalledWith('/workflows/wf-2?publish=1')
  })

  it('templates tab filters out non-template rows', () => {
    useWorkflowsMock.mockReturnValue({
      data: [
        makeWf({ id: 'a', name: 'mine-1', is_template: false }),
        makeWf({ id: 'b', name: 'tpl-1', is_template: true }),
      ],
      isLoading: false,
      error: null,
    })
    render(
      <MemoryRouter>
        <WorkflowsList />
      </MemoryRouter>,
    )
    fireEvent.click(screen.getByText(/模板\s+1/))
    expect(screen.queryByText('mine-1')).not.toBeInTheDocument()
    expect(screen.getByText('tpl-1')).toBeInTheDocument()
  })

  it('empty state CTA creates a new workflow', () => {
    useWorkflowsMock.mockReturnValue({ data: [], isLoading: false, error: null })
    render(
      <MemoryRouter>
        <WorkflowsList />
      </MemoryRouter>,
    )
    fireEvent.click(screen.getByText('新建 Workflow'))
    expect(createMutateMock).toHaveBeenCalled()
  })
})

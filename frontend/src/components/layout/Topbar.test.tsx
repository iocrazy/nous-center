import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Topbar from './Topbar'
import { useExecutionStore } from '../../stores/execution'

function renderTopbar() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>
        <Topbar />
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

const toastAdd = vi.fn()
vi.mock('../../stores/toast', () => ({
  useToastStore: (sel: (s: { add: typeof toastAdd }) => unknown) => sel({ add: toastAdd }),
}))

// executeWorkflow 现在返回 202 契约的 { task_id }（Lane S）。
const executeWorkflow = vi.fn()
vi.mock('../../utils/workflowExecutor', () => ({
  executeWorkflow: (...a: unknown[]) => executeWorkflow(...a),
}))

vi.mock('../../api/workflows', () => ({ useUnpublishWorkflow: () => ({ mutate: vi.fn() }) }))

const requestPermission = vi.fn().mockResolvedValue(undefined)
vi.mock('../../stores/notifications', () => ({
  useNotificationStore: (sel?: (s: { requestPermission: typeof requestPermission }) => unknown) =>
    sel ? sel({ requestPermission }) : { requestPermission },
}))

describe('Topbar handleRun — async 202 feedback', () => {
  beforeEach(() => {
    toastAdd.mockClear()
    executeWorkflow.mockReset()
    requestPermission.mockClear()
    useExecutionStore.setState({ taskIconBadge: 0, taskPanelOpen: false, isRunning: false })
  })

  it('on Run: shows enqueued toast, bumps badge, does NOT open the panel', async () => {
    executeWorkflow.mockResolvedValue({ task_id: 'wf_exec_9z' })
    renderTopbar()
    fireEvent.click(screen.getByText(/Run/))
    await waitFor(() => expect(executeWorkflow).toHaveBeenCalled())
    expect(toastAdd).toHaveBeenCalledWith(expect.stringContaining('入队'), 'info')
    expect(useExecutionStore.getState().taskIconBadge).toBe(1)
    // 面板不自动打开 —— DD4
    expect(useExecutionStore.getState().taskPanelOpen).toBe(false)
  })

  it('first Run triggers the browser notification permission request', async () => {
    executeWorkflow.mockResolvedValue({ task_id: 'wf_x' })
    renderTopbar()
    fireEvent.click(screen.getByText(/Run/))
    await waitFor(() => expect(requestPermission).toHaveBeenCalled())
  })
})

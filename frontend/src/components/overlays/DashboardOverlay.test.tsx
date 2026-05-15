import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import DashboardOverlay from './DashboardOverlay'

// DashboardOverlay 拉一堆 api —— 全部 mock 成空/最小数据，只验 GpuCard runner 标签。
vi.mock('../../api/dashboard', () => ({ useDashboardSummary: () => ({ data: undefined }) }))
vi.mock('../../api/observability', () => ({ useRuntimeMetrics: () => ({ data: undefined, isLoading: false, error: null }) }))
vi.mock('../../api/vllm', () => ({
  useVLLMMetrics: () => ({ data: { instances: [] }, isLoading: false, error: null }),
  useUpdateLaunchParams: () => ({ mutate: vi.fn(), isPending: false }),
}))
vi.mock('../../api/engines', () => ({ useEngines: () => ({ data: [] }) }))
vi.mock('../../api/system', () => ({
  useSysGpus: () => ({
    data: {
      count: 1,
      gpus: [{
        index: 2, name: 'RTX Pro 6000', utilization_gpu: 40, utilization_memory: 30,
        temperature: 55, fan_speed: 30, power_draw_w: 200, power_limit_w: 600,
        memory_used_mb: 20000, memory_total_mb: 98000, memory_free_mb: 78000, processes: [],
      }],
    },
  }),
  useSysStats: () => ({ data: undefined }),
  useSysProcesses: () => ({ data: undefined }),
  useKillProcess: () => ({ mutate: vi.fn() }),
}))
vi.mock('../../api/runners', () => ({
  useRunners: () => ({
    data: [
      { id: 'runner-i', label: 'Runner-I', role: 'image', state: 'busy',
        current_task: null, queue: [], restart_attempt: null, load_error: null, gpus: [2] },
    ],
  }),
}))

describe('DashboardOverlay GpuCard — runner label (DD3)', () => {
  it('shows the owning runner label on the GPU card', () => {
    render(<DashboardOverlay />)
    // 系统状态默认收起，需要先点开
    const sysToggle = screen.getByText('系统状态')
    fireEvent.click(sysToggle)
    // GPU index 2 属于 runner-i → 卡片上应有 "Runner-I (image)"
    expect(screen.getByText(/Runner-I \(image\)/)).toBeTruthy()
  })
})

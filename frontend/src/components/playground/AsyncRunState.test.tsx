import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, act } from '@testing-library/react'
import AsyncRunState from './AsyncRunState'
import * as api from '../../api/predictions'
import type { Prediction } from '../../api/predictions'

vi.mock('../../api/predictions', () => ({
  getPrediction: vi.fn(),
  cancelPrediction: vi.fn(),
  createPredictionAsync: vi.fn(),
}))

function pred(overrides: Partial<Prediction>): Prediction {
  return {
    id: 'p1',
    service: 'svc',
    status: 'starting',
    input: {},
    output: null,
    error: null,
    progress: { nodes_done: 0, nodes_total: 2 },
    metrics: {},
    created_at: new Date().toISOString(),
    started_at: null,
    completed_at: null,
    ...overrides,
  }
}

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
  vi.clearAllMocks()
})

describe('AsyncRunState', () => {
  it('轮询推进 已提交/排队中 → 运行中 → 完成,succeeded 时把 output 交给 onDone', async () => {
    const onDone = vi.fn()
    const onCancel = vi.fn()
    vi.mocked(api.getPrediction)
      .mockResolvedValueOnce(pred({ status: 'starting' }))
      .mockResolvedValueOnce(pred({ status: 'processing', progress: { nodes_done: 1, nodes_total: 2 } }))
      .mockResolvedValueOnce(pred({
        status: 'succeeded',
        output: { outputs: { '92': { video_url: '/x.mp4' } } },
      }))

    render(<AsyncRunState predictionId="p1" onDone={onDone} onCancel={onCancel} />)

    await act(async () => { await Promise.resolve() })
    expect(screen.getByText('排队中')).toBeInTheDocument()
    expect(onDone).not.toHaveBeenCalled()

    await act(async () => { await vi.advanceTimersByTimeAsync(2000) })
    expect(screen.getByText('运行中')).toBeInTheDocument()
    expect(onDone).not.toHaveBeenCalled()

    await act(async () => { await vi.advanceTimersByTimeAsync(2000) })
    expect(onDone).toHaveBeenCalledTimes(1)
    expect(onDone).toHaveBeenCalledWith({ outputs: { '92': { video_url: '/x.mp4' } } })
  })

  it('取消按钮调用 cancelPrediction 再触发 onCancel', async () => {
    const onDone = vi.fn()
    const onCancel = vi.fn()
    vi.mocked(api.getPrediction).mockResolvedValue(pred({ status: 'processing' }))
    vi.mocked(api.cancelPrediction).mockResolvedValue(pred({ status: 'canceled' }))

    render(<AsyncRunState predictionId="p1" onDone={onDone} onCancel={onCancel} />)
    await act(async () => { await Promise.resolve() })

    fireEvent.click(screen.getByRole('button', { name: /取消/ }))
    await act(async () => { await Promise.resolve() })

    expect(api.cancelPrediction).toHaveBeenCalledWith('p1')
    expect(onCancel).toHaveBeenCalledTimes(1)
  })

  it('failed 终态展示后端 error 文案', async () => {
    vi.mocked(api.getPrediction).mockResolvedValue(pred({ status: 'failed', error: '节点 92 崩了' }))

    render(<AsyncRunState predictionId="p1" onDone={vi.fn()} onCancel={vi.fn()} />)
    await act(async () => { await Promise.resolve() })

    expect(screen.getByText('节点 92 崩了')).toBeInTheDocument()
  })

  it('卸载后清定时器,不再继续轮询', async () => {
    vi.mocked(api.getPrediction).mockResolvedValue(pred({ status: 'starting' }))

    const { unmount } = render(<AsyncRunState predictionId="p1" onDone={vi.fn()} onCancel={vi.fn()} />)
    await act(async () => { await Promise.resolve() })
    const callsBeforeUnmount = vi.mocked(api.getPrediction).mock.calls.length

    unmount()
    await act(async () => { await vi.advanceTimersByTimeAsync(6000) })

    expect(vi.mocked(api.getPrediction).mock.calls.length).toBe(callsBeforeUnmount)
  })
})

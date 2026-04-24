import { describe, it, expect, vi, beforeEach } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import UsagePage from './UsagePage'
import type { UsageSummary, Timeseries, TopKeys } from '../api/usage'

const useSummaryMock = vi.fn()
const useSeriesMock = vi.fn()
const useTopKeysMock = vi.fn()

vi.mock('../api/usage', async () => {
  const actual = await vi.importActual<typeof import('../api/usage')>('../api/usage')
  return {
    ...actual,
    useUsageSummary: (days: number) => useSummaryMock(days),
    useUsageTimeseries: (days: number) => useSeriesMock(days),
    useUsageTopKeys: (days: number, limit?: number) => useTopKeysMock(days, limit),
  }
})

// recharts pulls in canvas-y dependencies that jsdom doesn't fully cover
// (ResizeObserver, getBoundingClientRect for SVG layout). The chart
// itself isn't what these tests assert, so swap it for a no-op.
vi.mock('recharts', () => ({
  Bar: () => null,
  BarChart: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
  CartesianGrid: () => null,
  Legend: () => null,
  ResponsiveContainer: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
  Tooltip: () => null,
  XAxis: () => null,
  YAxis: () => null,
}))

function summary(over: Partial<UsageSummary> = {}): UsageSummary {
  return {
    days: 7,
    period_start: '2026-04-17T00:00:00Z',
    period_end: '2026-04-24T00:00:00Z',
    total_calls: 1234,
    total_tokens: 56789,
    prompt_tokens: 20000,
    completion_tokens: 36789,
    tts_characters: 999,
    avg_latency_ms: 1400,
    error_rate: null,
    prev_total_calls: 1000,
    prev_total_tokens: 50000,
    ...over,
  }
}

function series(): Timeseries {
  return {
    days: 7,
    points: [
      { date: '2026-04-23', by_service: { 'svc-a': 5, 'svc-b': 3 } },
      { date: '2026-04-24', by_service: { 'svc-a': 7 } },
    ],
    top_services: ['svc-a', 'svc-b'],
  }
}

function topKeys(rows: TopKeys['rows'] = []): TopKeys {
  return { days: 7, rows }
}

beforeEach(() => {
  useSummaryMock.mockReset()
  useSeriesMock.mockReset()
  useTopKeysMock.mockReset()
  // sensible defaults — individual tests override summary as needed
  useSummaryMock.mockReturnValue({ data: summary(), isLoading: false, error: null })
  useSeriesMock.mockReturnValue({ data: series(), isLoading: false, error: null })
  useTopKeysMock.mockReturnValue({ data: topKeys(), isLoading: false, error: null })
})

describe('UsagePage', () => {
  it('renders all 4 stat cards with formatted values', () => {
    render(<UsagePage />)
    // total calls 1.2K (1234 → /1000 → 1.2K)
    expect(screen.getByText('1.2K')).toBeInTheDocument()
    // total tokens 56.8K
    expect(screen.getByText('56.8K')).toBeInTheDocument()
    // avg latency 1.40s
    expect(screen.getByText('1.40s')).toBeInTheDocument()
    // error rate null → "—"
    expect(screen.getByText('（暂未采集）')).toBeInTheDocument()
  })

  it('shows the period-over-period delta', () => {
    render(<UsagePage />)
    // (1234 - 1000) / 1000 = 23.4% ↑
    expect(screen.getByText(/↑\s*23\.4%/)).toBeInTheDocument()
  })

  it('falls back to "—" when avg_latency_ms is null', () => {
    useSummaryMock.mockReturnValue({
      data: summary({ avg_latency_ms: null }),
      isLoading: false,
      error: null,
    })
    render(<UsagePage />)
    expect(screen.getAllByText('—').length).toBeGreaterThan(0)
  })

  it('changing the days range refetches with the new value', () => {
    render(<UsagePage />)
    expect(useSummaryMock).toHaveBeenLastCalledWith(7)
    fireEvent.change(screen.getByRole('combobox'), { target: { value: '30' } })
    expect(useSummaryMock).toHaveBeenLastCalledWith(30)
    expect(useSeriesMock).toHaveBeenLastCalledWith(30)
    expect(useTopKeysMock).toHaveBeenLastCalledWith(30, 10)
  })

  it('renders an empty-state row in the top-keys table when nothing was called', () => {
    render(<UsagePage />)
    expect(screen.getByText(/该窗口内暂无调用/)).toBeInTheDocument()
  })

  it('renders mode badges + numeric columns for each top-key row', () => {
    useTopKeysMock.mockReturnValue({
      data: topKeys([
        {
          api_key_id: 1,
          label: 'mn-key',
          key_prefix: 'sk-mn1234',
          mode: 'm:n',
          calls: 4182,
          tokens: 1_200_000,
          avg_latency_ms: 2100,
        },
        {
          api_key_id: 2,
          label: 'legacy-key',
          key_prefix: 'sk-leg5678',
          mode: 'legacy',
          calls: 2105,
          tokens: 847_000,
          avg_latency_ms: 3800,
        },
      ]),
      isLoading: false,
      error: null,
    })
    render(<UsagePage />)
    expect(screen.getByText('mn-key')).toBeInTheDocument()
    expect(screen.getByText('legacy-key')).toBeInTheDocument()
    expect(screen.getByText('M:N')).toBeInTheDocument()
    expect(screen.getByText('Legacy')).toBeInTheDocument()
    expect(screen.getByText('1.2M')).toBeInTheDocument()
    expect(screen.getByText('2.10s')).toBeInTheDocument()
  })

  it('shows an error message when the timeseries query fails', () => {
    useSeriesMock.mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error('boom'),
    })
    render(<UsagePage />)
    expect(screen.getByText('boom')).toBeInTheDocument()
  })
})

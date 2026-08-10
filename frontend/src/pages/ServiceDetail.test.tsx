import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, act } from '@testing-library/react'
import { formatHms, AsrSegments, buildAsrCurl, PlaygroundTab } from './ServiceDetail'
import type { ServiceDetail as ServiceDetailT } from '../api/services'
import * as predictionsApi from '../api/predictions'
import type { Prediction } from '../api/predictions'

vi.mock('../api/predictions', () => ({
  createPredictionAsync: vi.fn(),
  getPrediction: vi.fn(),
  cancelPrediction: vi.fn(),
}))

describe('buildAsrCurl', () => {
  it('默认格式:multipart /v1/audio/transcriptions + timestamps=true(平台增强段)', () => {
    const curl = buildAsrCurl('moss-asr', false)
    expect(curl).toContain("POST 'https://YOUR_HOST/v1/audio/transcriptions'")
    expect(curl).toContain("-F 'file=@audio.wav'")
    expect(curl).toContain("-F 'model=moss-asr'")
    expect(curl).toContain("-F 'timestamps=true'")
    expect(curl).not.toContain('verbose_json')
  })
  it('verbose_json:走 response_format=verbose_json(OpenAI-Whisper SDK 直连)', () => {
    const curl = buildAsrCurl('moss-asr', true)
    expect(curl).toContain("-F 'response_format=verbose_json'")
    expect(curl).toContain("-F 'model=moss-asr'")
    expect(curl).not.toContain('timestamps=true')
  })
})

describe('formatHms', () => {
  it('mm:ss below 1h(分/秒两位)', () => {
    expect(formatHms(0)).toBe('00:00')
    expect(formatHms(5)).toBe('00:05')
    expect(formatHms(65)).toBe('01:05')
  })
  it('边界:3599s → 59:59(仍在 1h 内)', () => {
    expect(formatHms(3599)).toBe('59:59')
  })
  it('边界:3600s → 1:00:00(进位到 h:mm:ss)', () => {
    expect(formatHms(3600)).toBe('1:00:00')
  })
  it('h:mm:ss above 1h', () => {
    expect(formatHms(3661)).toBe('1:01:01')
    expect(formatHms(7325)).toBe('2:02:05')
  })
  it('float 秒向下取整(后端契约是秒 float)', () => {
    expect(formatHms(3.9)).toBe('00:03')
    expect(formatHms(3599.99)).toBe('59:59')
  })
  it('非有限/负值兜底为 0', () => {
    expect(formatHms(NaN)).toBe('00:00')
    expect(formatHms(-10)).toBe('00:00')
    expect(formatHms(Infinity)).toBe('00:00')
  })
})

describe('AsrSegments', () => {
  const segs = [
    { start: 0.28, end: 3.24, speaker: 'S01', text: '大家好' },
    { start: 3.24, end: 65.5, speaker: 'S02', text: '你好' },
    { start: 65.5, end: 70, speaker: 'S01', text: '再说一句' },
  ]

  it('requested=false 时不渲染(维持纯文本现状,不回归)', () => {
    const { container } = render(<AsrSegments segments={segs} requested={false} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('无 segments 时不渲染', () => {
    const { container } = render(<AsrSegments segments={[]} requested={true} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('渲染每段时间轴 + 说话人徽标 + 文本', () => {
    render(<AsrSegments segments={segs} requested={true} />)
    // mm:ss 时间轴(秒 float 取整)
    expect(screen.getByText('[00:00]')).toBeInTheDocument()
    expect(screen.getByText('[01:05]')).toBeInTheDocument()
    // 段文本
    expect(screen.getByText('大家好')).toBeInTheDocument()
    expect(screen.getByText('再说一句')).toBeInTheDocument()
    // 说话人徽标(S01 出现两段)
    expect(screen.getAllByText('S01')).toHaveLength(2)
    expect(screen.getByText('S02')).toBeInTheDocument()
  })

  it('顶部小结:总时长 + 段数 + 说话人数', () => {
    render(<AsrSegments segments={segs} requested={true} />)
    expect(screen.getByText(/总时长\s*01:10/)).toBeInTheDocument()
    expect(screen.getByText('3 段')).toBeInTheDocument()
    expect(screen.getByText('2 位说话人')).toBeInTheDocument()
  })

  it('同一说话人全程同色(稳定轮转分配)', () => {
    render(<AsrSegments segments={segs} requested={true} />)
    const s01 = screen.getAllByText('S01')
    expect(s01[0].getAttribute('style')).toBe(s01[1].getAttribute('style'))
    // 不同说话人不同色
    const s02 = screen.getByText('S02')
    expect(s02.getAttribute('style')).not.toBe(s01[0].getAttribute('style'))
  })

  it('speaker null 分支:无徽标纯文本行,仍带时间轴', () => {
    const withNull = [
      { start: 0, end: 2, speaker: null, text: '无归属的一段' },
      { start: 2, end: 4, speaker: 'S01', text: '有归属' },
    ]
    render(<AsrSegments segments={withNull} requested={true} />)
    expect(screen.getByText('无归属的一段')).toBeInTheDocument()
    expect(screen.getByText('[00:00]')).toBeInTheDocument()
    // 只有一位说话人入表
    expect(screen.getByText('1 位说话人')).toBeInTheDocument()
    // null 段不产生徽标 → 全局只有一个 S01 徽标
    expect(screen.getAllByText('S01')).toHaveLength(1)
  })

  it('全 null 段:不显示说话人数小结', () => {
    const allNull = [
      { start: 0, end: 2, speaker: null, text: 'a' },
      { start: 2, end: 4, speaker: null, text: 'b' },
    ]
    render(<AsrSegments segments={allNull} requested={true} />)
    expect(screen.getByText('2 段')).toBeInTheDocument()
    expect(screen.queryByText(/位说话人/)).not.toBeInTheDocument()
  })
})

// Task 10 review 修复(Critical):comfy_template 服务走 respond-async 提交后,PlaygroundTab
// 的 running/asyncPredictionId 全靠 AsyncRunState 的 onDone/onError/onCancel 回调复位——
// 之前只接了 onDone/onCancel,failed 终态没有出口,running 永久 true、提交按钮永久禁用。
// 这里直接渲染 PlaygroundTab(不经 react-query/react-router 的 ServiceDetailPage 外壳,
// 那层跟这个 bug 无关,只会让测试更重),走真实的 submit → createPredictionAsync →
// AsyncRunState 轮询 → 回调收尾这条完整生产路径。
describe('PlaygroundTab · comfy_template respond-async 集成', () => {
  function comfyService(overrides: Partial<ServiceDetailT> = {}): ServiceDetailT {
    return {
      id: 'svc-1',
      name: 'comfy-svc',
      type: 'app',
      status: 'active',
      source_type: 'comfy_template',
      source_id: '7',
      source_name: 'tpl',
      category: 'image',
      meter_dim: null,
      workflow_id: null,
      workflow_name: null,
      snapshot_hash: null,
      snapshot_schema_version: 1,
      version: 1,
      models: [],
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      workflow_snapshot: { nodes: [{ id: '138', type: 'T' }], edges: [] },
      exposed_inputs: [],
      exposed_outputs: [],
      ...overrides,
    }
  }

  function predictionFixture(overrides: Partial<Prediction>): Prediction {
    return {
      id: '42',
      service: 'comfy-svc',
      status: 'starting',
      input: {},
      output: null,
      error: null,
      progress: { nodes_done: 0, nodes_total: 1 },
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

  it('完整周期 提交→轮询→succeeded:output 走 SchemaDrivenOutput,按钮复位可再次运行', async () => {
    vi.mocked(predictionsApi.createPredictionAsync).mockResolvedValue(predictionFixture({}))
    vi.mocked(predictionsApi.getPrediction)
      .mockResolvedValueOnce(predictionFixture({ status: 'processing' }))
      .mockResolvedValueOnce(predictionFixture({
        status: 'succeeded',
        output: { outputs: { '92': { video_url: '/race-cat.mp4' } } },
      }))

    render(<PlaygroundTab svc={comfyService()} />)

    fireEvent.click(screen.getByText(/▶ 运行/))
    await act(async () => { await Promise.resolve() })
    expect(predictionsApi.createPredictionAsync).toHaveBeenCalledWith('comfy-svc', {})
    // 提交中:按钮禁用(running=true),不能重复点。
    expect(screen.getByText(/运行中…/)).toBeDisabled()

    await act(async () => { await vi.advanceTimersByTimeAsync(2000) })
    await act(async () => { await vi.advanceTimersByTimeAsync(2000) })

    // succeeded → onDone 把 output 塞进既有 SchemaDrivenOutput 渲染路径(exposed_outputs=[]
    // 时兜底转储原始 JSON,断言原文出现即证明走的是同一条渲染代码,不是另起一套展示)。
    expect(screen.getByText(/race-cat\.mp4/)).toBeInTheDocument()
    // 按钮解锁,可以再次提交 —— 不是被永久锁死的「运行中…」。
    const runBtn = screen.getByText('▶ 运行')
    expect(runBtn).not.toBeDisabled()
  })

  it('完整周期 提交→轮询→failed:展示 error 文案,按钮解锁(回归 Critical:之前会永久锁死)', async () => {
    vi.mocked(predictionsApi.createPredictionAsync).mockResolvedValue(predictionFixture({}))
    // 首次轮询就拿到 failed(边界:提交即失败)—— mockResolvedValue(非 Once)让每次调用都
    // 立即落终态,精确复现「onError 是唯一出口」这条路径,不依赖多轮 timer 推进。
    vi.mocked(predictionsApi.getPrediction).mockResolvedValue(
      predictionFixture({ status: 'failed', error: 'ComfyUI 节点 92 崩了' }),
    )

    render(<PlaygroundTab svc={comfyService()} />)

    fireEvent.click(screen.getByText(/▶ 运行/))
    await act(async () => { await Promise.resolve() })
    await act(async () => { await Promise.resolve() })

    // error 文案展示(经 onError → setError/setStatus('failed') → 复用同步路径的
    // SchemaDrivenOutput error 渲染)。
    expect(screen.getByText('ComfyUI 节点 92 崩了')).toBeInTheDocument()
    // Critical 回归点:按钮必须解锁,不能停在「运行中…」再也点不动。
    const runBtn = screen.getByText('▶ 运行')
    expect(runBtn).not.toBeDisabled()
  })
})

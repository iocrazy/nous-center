import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { formatHms, AsrSegments } from './ServiceDetail'

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

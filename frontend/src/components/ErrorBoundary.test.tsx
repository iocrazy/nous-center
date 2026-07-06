import { render } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { ErrorBoundary } from './ErrorBoundary'

// mock 上报,避免测试打 sendBeacon
vi.mock('../utils/errorReporter', () => ({ reportReactError: vi.fn() }))

function Boom(): React.ReactElement {
  throw new Error('kaboom')
}

describe('ErrorBoundary', () => {
  it('渲染错误时显示降级 UI + 重载按钮,不白屏', () => {
    // 抑制 React 打印的错误噪声
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {})
    const { getByRole, getByText } = render(
      <ErrorBoundary><Boom /></ErrorBoundary>,
    )
    expect(getByRole('alert')).toBeTruthy()
    expect(getByText('kaboom')).toBeTruthy()
    expect(getByText('重新加载')).toBeTruthy()
    spy.mockRestore()
  })

  it('正常子组件原样渲染', () => {
    const { getByText } = render(
      <ErrorBoundary><div>hello</div></ErrorBoundary>,
    )
    expect(getByText('hello')).toBeTruthy()
  })
})

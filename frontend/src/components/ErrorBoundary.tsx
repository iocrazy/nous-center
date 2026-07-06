import { Component, type ErrorInfo, type ReactNode } from 'react'
import { reportReactError } from '../utils/errorReporter'

interface Props {
  children: ReactNode
}
interface State {
  hasError: boolean
  message: string
}

// 根级错误边界(先进性 A1):此前 main.tsx 只包 QueryClientProvider,任一组件渲染
// 抛错 → 整个 SPA 白屏。这里降级成可读错误页 + 重载按钮,并上报到 /logs/frontend。
export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, message: '' }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, message: error?.message || 'Unexpected error' }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    reportReactError(error, info.componentStack || undefined)
  }

  handleReload = () => {
    // 清错误态后整页重载 —— 单管理员场景最省事的恢复。
    this.setState({ hasError: false, message: '' })
    window.location.reload()
  }

  render() {
    if (!this.state.hasError) return this.props.children
    return (
      <div
        role="alert"
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '100vh',
          gap: 16,
          padding: 24,
          background: 'var(--bg, #111)',
          color: 'var(--text, #eee)',
          fontFamily: 'system-ui, sans-serif',
        }}
      >
        <div style={{ fontSize: 18, fontWeight: 600 }}>页面出错了</div>
        <div style={{ fontSize: 13, opacity: 0.7, maxWidth: 480, textAlign: 'center', wordBreak: 'break-word' }}>
          {this.state.message}
        </div>
        <button
          onClick={this.handleReload}
          style={{
            marginTop: 8,
            padding: '8px 20px',
            borderRadius: 6,
            border: '1px solid var(--border, #444)',
            background: 'var(--accent, #2563eb)',
            color: '#fff',
            cursor: 'pointer',
            fontSize: 14,
          }}
        >
          重新加载
        </button>
      </div>
    )
  }
}

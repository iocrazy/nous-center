const REPORT_URL = '/api/v1/logs/frontend'

function report(type: string, message: string, page: string, stack?: string) {
  try {
    navigator.sendBeacon(REPORT_URL, JSON.stringify({ type, message, page, stack: stack || null }))
  } catch {
    // Fallback to fetch
    fetch(REPORT_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type, message, page, stack: stack || null }),
    }).catch(() => {})
  }
}

export function installErrorReporter() {
  // JS errors
  window.addEventListener('error', (e) => {
    report('error', e.message || 'Unknown error', window.location.pathname, e.error?.stack)
  })

  // Unhandled promise rejections
  window.addEventListener('unhandledrejection', (e) => {
    const msg = e.reason?.message || String(e.reason) || 'Unhandled rejection'
    report('unhandled_rejection', msg, window.location.pathname, e.reason?.stack)
  })

  // Patch fetch for network errors
  const originalFetch = window.fetch
  window.fetch = async (...args) => {
    try {
      const resp = await originalFetch(...args)
      if (!resp.ok && resp.status >= 500) {
        const url = typeof args[0] === 'string' ? args[0] : (args[0] as Request).url
        // round3 #3:5xx 分支也要排除日志端点自身。否则 report() 的 fallback fetch
        // (sendBeacon 不可用时)打到 /logs/、若该端点返回 500 → 再 report → 再 fetch
        // → 无限循环狂刷后端。catch 分支早有此守卫,5xx 分支漏了。
        if (!url.includes('/api/v1/logs/')) {
          report('network', `${resp.status} ${resp.statusText} — ${url}`, window.location.pathname)
        }
      }
      return resp
    } catch (err: any) {
      const url = typeof args[0] === 'string' ? args[0] : (args[0] as Request).url
      // Don't report errors for the log endpoint itself (avoid recursion)
      if (!url.includes('/api/v1/logs/')) {
        report('network', `${err.message} — ${url}`, window.location.pathname)
      }
      throw err
    }
  }
}

// React 渲染错误(ErrorBoundary 用)—— window.onerror 抓不到 React 组件树内的
// 渲染异常,需 ErrorBoundary.componentDidCatch 显式上报到同一 /logs/frontend。
export function reportReactError(error: Error, componentStack?: string) {
  report(
    'react_error_boundary',
    error.message || String(error),
    window.location.pathname,
    (error.stack || '') + (componentStack ? '\n--- component stack ---\n' + componentStack : ''),
  )
}

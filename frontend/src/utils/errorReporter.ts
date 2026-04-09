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
        report('network', `${resp.status} ${resp.statusText} — ${url}`, window.location.pathname)
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

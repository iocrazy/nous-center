import './utils/cryptoPolyfill' // 必须最先执行:非安全上下文(明文 HTTP 内网)补 crypto.randomUUID
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import App from './App'
import { ErrorBoundary } from './components/ErrorBoundary'
import { installErrorReporter } from './utils/errorReporter'
import { setUnauthorizedHandler } from './api/client'
import { ADMIN_ME_KEY } from './api/admin'
import { shouldRetry } from './api/retryPolicy'
import './index.css'

installErrorReporter()

// 4xx 不重试(见 api/retryPolicy.ts):React Query 默认对任何失败重试 3 次,而 4xx
// 重发一百遍还是同样的错 —— 白等 4 轮,期间界面一直显示「加载中…」。
const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: shouldRetry },
    mutations: { retry: shouldRetry },
  },
})

// Cookie expired or admin pulled the rug → flip AuthGate back to Login.
// Plugin definitions are loaded inside <AuthGate> after authentication so we
// don't fire /api/* requests before the user has had a chance to log in.
setUnauthorizedHandler(() => {
  queryClient.invalidateQueries({ queryKey: ADMIN_ME_KEY })
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>
    </ErrorBoundary>
  </StrictMode>,
)

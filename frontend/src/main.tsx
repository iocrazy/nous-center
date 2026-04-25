import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import App from './App'
import { installErrorReporter } from './utils/errorReporter'
import { setUnauthorizedHandler } from './api/client'
import { ADMIN_ME_KEY } from './api/admin'
import './index.css'

installErrorReporter()

const queryClient = new QueryClient()

// Cookie expired or admin pulled the rug → flip AuthGate back to Login.
// Plugin definitions are loaded inside <AuthGate> after authentication so we
// don't fire /api/* requests before the user has had a chance to log in.
setUnauthorizedHandler(() => {
  queryClient.invalidateQueries({ queryKey: ADMIN_ME_KEY })
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
)

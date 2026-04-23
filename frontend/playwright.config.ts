import { defineConfig, devices } from 'playwright/test'

const BASE_URL = process.env.PW_BASE_URL ?? 'http://localhost:5173'

/**
 * v3 Playground e2e specs.
 *
 * These specs need a real backend (the Python API + a DB with at least one
 * loaded engine) and the Vite dev server. They're intentionally NOT in the
 * default `npm test` run so CI stays self-contained.
 *
 * Local one-shot:
 *   (terminal A) cd backend && uvicorn src.api.main:app
 *   (terminal B) cd frontend && npm run dev
 *   (terminal C) cd frontend && npm run test:e2e
 */
export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  fullyParallel: false,
  retries: 0,
  use: {
    baseURL: BASE_URL,
    trace: 'on-first-retry',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
})

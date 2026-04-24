import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import wasm from 'vite-plugin-wasm'

export default defineConfig({
  plugins: [react(), tailwindcss(), wasm()],
  server: {
    proxy: {
      '/api/': 'http://localhost:8000',
      '/v1': 'http://localhost:8000',
      '/sys': 'http://localhost:8001',
      '/ws': { target: 'ws://localhost:8000', ws: true },
    },
  },
  build: {
    rollupOptions: {
      output: {
        // Split heavy node_modules into focused vendor chunks so the
        // initial app-only chunk stays small. Keeps the workflow canvas
        // (xyflow) and usage charts (recharts) out of the critical path
        // for users who just hit /services or /dashboard.
        manualChunks(id) {
          if (!id.includes('node_modules')) return
          if (
            id.includes('/react/') ||
            id.includes('/react-dom/') ||
            id.includes('/react-router-dom/') ||
            id.includes('/react-router/') ||
            id.includes('/scheduler/')
          ) {
            return 'vendor-react'
          }
          if (id.includes('@tanstack')) return 'vendor-query'
          if (id.includes('@xyflow')) return 'vendor-xyflow'
          // d3-* is shared by both recharts and @xyflow's zoom/selection
          // helpers — pull it into its own chunk so the two consumers
          // don't form a circular dependency through the shared code.
          if (id.includes('/d3-')) return 'vendor-d3'
          if (id.includes('recharts') || id.includes('/victory-')) return 'vendor-charts'
          if (id.includes('lucide-react')) return 'vendor-icons'
          if (id.includes('zustand')) return 'vendor-state'
        },
      },
    },
  },
})

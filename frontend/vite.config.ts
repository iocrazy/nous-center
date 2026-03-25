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
})

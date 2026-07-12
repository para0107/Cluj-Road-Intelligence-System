import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true,   // live-mode WebSocket (/api/live/ws)
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        // Stable vendor chunks: the app shell loads only "react"; maps,
        // charts and animation libraries arrive with the page that needs
        // them. The assistant stack (@mlc-ai/web-llm, @langchain/*,
        // @huggingface/transformers) is deliberately NOT listed — it is
        // reached through dynamic import() only and gets its own lazy chunks.
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined
          if (id.includes('react-leaflet') || id.includes('node_modules/leaflet/')) return 'leaflet'
          if (id.includes('recharts') || id.includes('victory-vendor') || id.includes('d3-')) return 'charts'
          if (id.includes('gsap') || id.includes('node_modules/ogl/')
              || id.includes('node_modules/motion') || id.includes('framer-motion')) return 'anim'
          if (id.includes('react-router') || id.includes('node_modules/react/')
              || id.includes('react-dom') || id.includes('node_modules/scheduler/')) return 'react'
          return undefined
        },
      },
    },
  },
})

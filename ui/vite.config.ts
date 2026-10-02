import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// The Rust binary serves the built assets under /_ui/, and mounts the index at
// /. `base` must match so the emitted asset URLs resolve.
export default defineConfig({
  base: '/_ui/',
  plugins: [vue()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  server: {
    // `bun run dev` proxies the API to a locally running `r2t2 serve`.
    proxy: {
      '/api': 'http://127.0.0.1:8272',
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        // The console.
        index: fileURLToPath(new URL('./index.html', import.meta.url)),
        // The caption overlay, kept separate so its bundle stays small: OBS
        // loads this as a browser source and nothing else needs to come with
        // it.
        live: fileURLToPath(new URL('./live.html', import.meta.url)),
      },
    },
  },
})

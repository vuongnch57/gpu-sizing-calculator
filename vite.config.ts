/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv, type Plugin } from 'vite'

// Cloudflare Web Analytics: cookieless page-view counts.
// The beacon is only added when VITE_CF_BEACON_TOKEN is set (see .env.production), so dev and
// test builds send nothing. The token is public by design: it ends up in the page HTML.
// `spa: false` because the app rewrites the query string with replaceState on every input
// change; we want one page view per visit, not one per keystroke.
function cloudflareWebAnalytics(token: string | undefined): Plugin {
  return {
    name: 'cloudflare-web-analytics',
    transformIndexHtml() {
      if (!token) return []
      return [
        {
          tag: 'script',
          attrs: {
            defer: true,
            src: 'https://static.cloudflareinsights.com/beacon.min.js',
            'data-cf-beacon': JSON.stringify({ token, spa: false }),
          },
          injectTo: 'body',
        },
      ]
    },
  }
}

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), 'VITE_')
  return {
    plugins: [react(), tailwindcss(), cloudflareWebAnalytics(env.VITE_CF_BEACON_TOKEN)],
    test: {
      include: ['src/**/*.test.ts'],
    },
  }
})

# GPU Sizing Calculator

Estimate the GPU memory needed to run an LLM for inference (weights + KV cache + overhead) and see which GPU setups fit. Results are estimates, shown in GB (10⁹ bytes).

## Development

Requires Node 20.19+ (22 recommended, see `.node-version`).

```sh
npm install
npm run dev          # local dev server
npm test             # unit tests (Vitest)
npm run typecheck    # TypeScript
npm run lint         # oxlint
npm run format       # Prettier
npm run build        # production build into dist/
```

## Deployment

Deployed on Cloudflare Workers (static assets) via Workers Builds, Cloudflare's GitHub integration. `wrangler.jsonc` runs `npm run build` and serves `dist/`; no server code is involved. Pushes to `main` deploy to production; other branches get preview builds.

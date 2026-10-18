# GPU Sizing Calculator

Estimate the GPU memory needed to run an LLM for inference (weights + KV cache + overhead) and see which GPU setups fit. Results are estimates, shown in GB (10⁹ bytes).

Live: https://vram.builtbyglenn.com

## How the estimate works

All formulas live in [`src/lib/calc/`](src/lib/calc/) as pure, unit-tested functions, each with a comment explaining why it works that way. The app's "Show the math" panel shows them with your numbers plugged in.

| Part               | Formula                                                |
| ------------------ | ------------------------------------------------------ |
| Weights            | `parameters × bytes per parameter`                     |
| KV cache per token | `2 × layers × KV heads × head dim × bytes per element` |
| KV cache           | `per token × context length × concurrent users`        |
| Overhead           | `max((weights + KV cache) × overhead %, 1 GB)`         |
| Total              | `weights + KV cache + overhead`                        |

- **Bytes per element:** FP32 4, FP16/BF16 2, FP8/INT8 1, INT4 0.5.
- **The 2 in the KV cache** is one key plus one value vector per token, per layer, per KV head.
- **KV heads, not attention heads:** with grouped-query attention (GQA) several query heads share one K/V head, so only KV heads are cached. Using attention heads would overestimate Llama 3.1 8B's cache by 4×.
- **Overhead** covers the CUDA context, activations and fragmentation. The 1 GB floor reflects the fixed per-GPU CUDA context cost.

### Multiple GPUs (tensor parallelism)

Splitting a model across N GPUs gives each GPU `weights / N` and `max(1, ceil(KV heads / N))` KV heads' worth of cache. Each GPU keeps its own overhead (with the 1 GB floor), so total memory grows with N. A split is only offered when attention heads divide by N and KV heads divide by N or N is a multiple of them. Recommendations try 1, 2, 4 and 8 GPUs.

### Limitations

These are estimates for inference only. Not modelled: training or fine-tuning memory, throughput, sliding-window attention (Gemma 2's KV cache is overestimated above 4K context), paged-attention block rounding, and quantization scale overhead (covered loosely by the overhead margin).

## Data sources

- **Models** ([`src/data/models.ts`](src/data/models.ts)): architecture values come from each model's `config.json` on Hugging Face, linked per preset. Gated Llama and Gemma configs come from `unsloth/` mirrors with identical configs. Parameter counts are computed from the config and match the published totals.
- **GPUs** ([`src/data/gpus.ts`](src/data/gpus.ts)): memory sizes from NVIDIA's product pages, linked per GPU.

## Sharing

Every input is stored in the URL, e.g. `?model=llama-3.1-70b&weights=fp8&kv=fp8&ctx=32768&users=8&overhead=15`. Custom models add `params` (billions), `layers`, `kvHeads` and `headDim`.

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

## Analytics

Page views are counted with [Cloudflare Web Analytics](https://developers.cloudflare.com/web-analytics/) (cookieless, no personal data). `vite.config.ts` adds the beacon script to `index.html` only when `VITE_CF_BEACON_TOKEN` is set at build time (in `.env.production` or as a Workers Builds build variable), so local dev sends nothing.

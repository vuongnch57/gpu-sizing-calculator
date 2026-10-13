import { useState } from 'react'
import { InputForm } from './components/InputForm'
import { GpuRecommendations } from './components/GpuRecommendations'
import { MemoryBreakdown } from './components/MemoryBreakdown'
import { GPU_COUNTS, GPUS, type GpuSpec } from './data/gpus'
import { DEFAULT_INPUTS, findPreset, resolveModel, type CalculatorInputs } from './inputs'
import {
  estimateMemory,
  recommendGpus,
  type GpuFit,
  type MemoryBreakdown as Breakdown,
} from './lib/calc'

function App() {
  const [inputs, setInputs] = useState<CalculatorInputs>(DEFAULT_INPUTS)
  const result = calculate(inputs)

  return (
    <main className="mx-auto max-w-5xl px-4 py-10">
      <header className="mb-8">
        <h1 className="text-3xl font-semibold tracking-tight">GPU Sizing Calculator</h1>
        <p className="mt-2 text-slate-600">
          Estimate how much GPU memory an LLM needs for inference, and which GPU setups fit.
        </p>
      </header>

      <div className="grid gap-8 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <InputForm inputs={inputs} onChange={(patch) => setInputs({ ...inputs, ...patch })} />
        </div>
        <div className="flex flex-col gap-8 md:self-start">
          {result.ok ? (
            <>
              <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                <MemoryBreakdown breakdown={result.breakdown} />
              </div>
              <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                <GpuRecommendations fits={result.fits} maxCount={Math.max(...GPU_COUNTS)} />
              </div>
            </>
          ) : (
            <p
              role="alert"
              className="rounded-xl border border-slate-200 bg-white p-5 text-sm text-red-700 shadow-sm"
            >
              {result.error}
            </p>
          )}
        </div>
      </div>
    </main>
  )
}

/** Maps the calc module's parameter names (used in its error messages) to form labels. */
const FIELD_LABELS: Record<string, string> = {
  paramCount: 'Parameters',
  numLayers: 'Layers',
  numKvHeads: 'KV heads',
  headDim: 'Head dim',
  contextLength: 'Context length',
  batchSize: 'Concurrent users',
  overheadPercent: 'Overhead margin',
}

type CalcResult =
  { ok: true; breakdown: Breakdown; fits: GpuFit<GpuSpec>[] } | { ok: false; error: string }

/** The calc functions throw on invalid input (e.g. a cleared field); show that instead of crashing. */
function calculate(inputs: CalculatorInputs): CalcResult {
  try {
    // Attention heads limit which GPU counts a model can be split across; custom models skip that check.
    const model = {
      ...resolveModel(inputs),
      numAttentionHeads: findPreset(inputs.modelId)?.numAttentionHeads,
    }
    return {
      ok: true,
      breakdown: estimateMemory(model, inputs),
      fits: recommendGpus(model, inputs, GPUS, GPU_COUNTS),
    }
  } catch (e) {
    if (e instanceof RangeError) {
      const field = Object.keys(FIELD_LABELS).find((k) => e.message.startsWith(k))
      const label = field ? FIELD_LABELS[field] : 'a field'
      return { ok: false, error: `Enter a valid value for ${label} to see the estimate.` }
    }
    throw e
  }
}

export default App

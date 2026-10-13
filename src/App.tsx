import { useState } from 'react'
import { InputForm } from './components/InputForm'
import { MemoryBreakdown } from './components/MemoryBreakdown'
import { DEFAULT_INPUTS, resolveModel, type CalculatorInputs } from './inputs'
import { estimateMemory, type MemoryBreakdown as Breakdown } from './lib/calc'

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
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm md:self-start">
          {result.ok ? (
            <MemoryBreakdown breakdown={result.breakdown} />
          ) : (
            <p role="alert" className="text-sm text-red-700">
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

type CalcResult = { ok: true; breakdown: Breakdown } | { ok: false; error: string }

/** The calc functions throw on invalid input (e.g. a cleared field); show that instead of crashing. */
function calculate(inputs: CalculatorInputs): CalcResult {
  try {
    return { ok: true, breakdown: estimateMemory(resolveModel(inputs), inputs) }
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
